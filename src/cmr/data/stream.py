from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


@dataclass(frozen=True)
class CrossModalData:
    images: np.ndarray
    texts: np.ndarray
    labels: np.ndarray
    ids: np.ndarray | None


@dataclass(frozen=True)
class StreamStep:
    idx: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class DatasetFiles:
    dataset_dir: Path
    feature_mat: Path
    train_stream_mat: Path
    test_stream_mat: Path


@dataclass(frozen=True)
class StreamLabelAlignment:
    enabled: bool
    reordered: bool
    stream_to_feature: tuple[int, ...]
    train_accuracy: float
    test_accuracy: float


def resolve_dataset_files(
    dataset: str,
    data_root: Path,
    feature_mat: str | Path | None = None,
    train_stream_mat: str | Path | None = None,
    test_stream_mat: str | Path | None = None,
) -> DatasetFiles:
    dataset_dir = data_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    stream_dir = dataset_dir / "data-stream"
    if not stream_dir.exists():
        raise FileNotFoundError(f"Stream directory not found: {stream_dir}")

    feature_path = Path(feature_mat) if feature_mat else _find_feature_mat(dataset_dir)
    train_path = Path(train_stream_mat) if train_stream_mat else _find_stream_mat(stream_dir, train=True)
    test_path = Path(test_stream_mat) if test_stream_mat else _find_stream_mat(stream_dir, train=False)

    return DatasetFiles(
        dataset_dir=dataset_dir,
        feature_mat=_resolve_path(feature_path),
        train_stream_mat=_resolve_path(train_path),
        test_stream_mat=_resolve_path(test_path),
    )


def load_cross_modal_mat(path: Path) -> CrossModalData:
    try:
        data = _load_cross_modal_mat_hdf5(path)
    except OSError:
        data = _load_cross_modal_mat_scipy(path)

    if data.images.shape[0] != data.texts.shape[0] or data.images.shape[0] != data.labels.shape[0]:
        raise ValueError(
            "Image, text, and label sample counts differ: "
            f"{data.images.shape[0]}, {data.texts.shape[0]}, {data.labels.shape[0]}"
        )
    return data


def load_stream_steps(path: Path, experiment: int = 0, variable_name: str | None = None) -> list[StreamStep]:
    try:
        steps = _load_stream_steps_hdf5(path, experiment=experiment, variable_name=variable_name)
    except OSError:
        steps = _load_stream_steps_scipy(path, experiment=experiment, variable_name=variable_name)

    if not steps:
        raise ValueError(f"No stream steps with idx found in {path}")
    return steps


def count_stream_experiments(path: Path, variable_name: str | None = None) -> int:
    """Return the number of experiment splits stored in a stream MAT file."""
    try:
        count = _count_stream_experiments_hdf5(path, variable_name=variable_name)
    except OSError:
        count = _count_stream_experiments_scipy(path, variable_name=variable_name)

    if count <= 0:
        raise ValueError(f"No experiment splits found in {path}")
    return count


def _count_stream_experiments_hdf5(path: Path, variable_name: str | None = None) -> int:
    with h5py.File(path, "r") as f:
        var = variable_name or _first_mat_variable(f.keys())
        top = f[var]
        if not isinstance(top, h5py.Dataset) or top.dtype != object:
            return 1

        refs = [ref for ref in top[()].reshape(-1) if ref]
        if not refs:
            return 0

        # A cell array may either contain experiment cells or directly contain
        # stream-step structs. The latter represents one experiment.
        first = f[refs[0]]
        if isinstance(first, h5py.Group) and "idx" in first:
            return 1
        return len(refs)


def _count_stream_experiments_scipy(path: Path, variable_name: str | None = None) -> int:
    import scipy.io as sio

    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    var = variable_name or _first_mat_variable(mat.keys())
    top = mat[var]
    if not isinstance(top, np.ndarray):
        return 1
    if top.size == 0:
        return 0
    if _get_scipy_field(top.reshape(-1)[0], "idx") is not None:
        return 1
    if top.dtype == object:
        return int(top.size)
    return 1


def _load_cross_modal_mat_hdf5(path: Path) -> CrossModalData:
    with h5py.File(path, "r") as f:
        keys = [k for k in f.keys() if not k.startswith("#")]
        image_key = _pick_key(keys, ["Images", "Image", "image", "I", "X_I"])
        text_key = _pick_key(keys, ["Texts", "Text", "text", "T", "X_T"])
        label_key = _pick_key(keys, ["Labels", "Label", "labels", "L"])
        id_key = _pick_optional_key(keys, ["Idx", "idx", "index", "indices"])

        ids = None
        if id_key is not None:
            ids = np.asarray(f[id_key][()]).reshape(-1).astype(np.int64)
            n_samples = int(ids.shape[0])
        else:
            raw = np.asarray(f[image_key])
            n_samples = max(raw.shape)

        images = _orient_samples(np.asarray(f[image_key][()], dtype=np.float32), n_samples)
        texts = _orient_samples(np.asarray(f[text_key][()], dtype=np.float32), n_samples)
        labels = _orient_samples(np.asarray(f[label_key][()], dtype=np.float32), n_samples)
        labels = (labels > 0).astype(np.float32)
    return CrossModalData(images=images, texts=texts, labels=labels, ids=ids)


def _load_cross_modal_mat_scipy(path: Path) -> CrossModalData:
    import scipy.io as sio

    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    image_key = _pick_key(keys, ["Images", "Image", "image", "I", "X_I"])
    text_key = _pick_key(keys, ["Texts", "Text", "text", "T", "X_T"])
    label_key = _pick_key(keys, ["Labels", "Label", "labels", "L"])
    id_key = _pick_optional_key(keys, ["Idx", "idx", "index", "indices"])

    ids = None
    if id_key is not None:
        ids = np.asarray(mat[id_key]).reshape(-1).astype(np.int64)
        n_samples = int(ids.shape[0])
    else:
        n_samples = max(np.asarray(mat[image_key]).shape)

    images = _orient_samples(np.asarray(mat[image_key], dtype=np.float32), n_samples)
    texts = _orient_samples(np.asarray(mat[text_key], dtype=np.float32), n_samples)
    labels = _orient_samples(np.asarray(mat[label_key], dtype=np.float32), n_samples)
    labels = (labels > 0).astype(np.float32)
    return CrossModalData(images=images, texts=texts, labels=labels, ids=ids)


def _load_stream_steps_hdf5(
    path: Path,
    experiment: int = 0,
    variable_name: str | None = None,
) -> list[StreamStep]:
    with h5py.File(path, "r") as f:
        var = variable_name or _first_mat_variable(f.keys())
        top = f[var]
        experiment_obj = _select_experiment(f, top, experiment)
        refs = _extract_object_refs(experiment_obj)

        steps: list[StreamStep] = []
        for ref in refs:
            obj = f[ref]
            if not isinstance(obj, h5py.Group) or "idx" not in obj:
                continue
            idx = np.asarray(obj["idx"][()]).reshape(-1).astype(np.int64)
            if "data" in obj:
                labels = _orient_samples(np.asarray(obj["data"][()], dtype=np.float32), idx.shape[0])
                labels = (labels > 0).astype(np.float32)
            else:
                labels = np.empty((idx.shape[0], 0), dtype=np.float32)
            steps.append(StreamStep(idx=idx, labels=labels))

    return steps


def _load_stream_steps_scipy(
    path: Path,
    experiment: int = 0,
    variable_name: str | None = None,
) -> list[StreamStep]:
    import scipy.io as sio

    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    var = variable_name or _first_mat_variable(mat.keys())
    top = mat[var]
    experiment_obj = _select_scipy_experiment(top, experiment)
    candidates = _flatten_scipy_cells(experiment_obj)
    if _get_scipy_field(experiment_obj, "idx") is not None:
        candidates = [experiment_obj]

    steps: list[StreamStep] = []
    for obj in candidates:
        idx_obj = _get_scipy_field(obj, "idx")
        if idx_obj is None:
            continue
        idx = np.asarray(idx_obj).reshape(-1).astype(np.int64)
        data_obj = _get_scipy_field(obj, "data")
        if data_obj is None:
            labels = np.empty((idx.shape[0], 0), dtype=np.float32)
        else:
            labels = _orient_samples(np.asarray(data_obj, dtype=np.float32), idx.shape[0])
            labels = (labels > 0).astype(np.float32)
        steps.append(StreamStep(idx=idx, labels=labels))
    return steps


def materialize_steps(
    data: CrossModalData,
    steps: Iterable[StreamStep],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    texts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for step in steps:
        rows = indices_to_rows(data.ids, step.idx, data.images.shape[0])
        images.append(data.images[rows])
        texts.append(data.texts[rows])
        labels.append(step.labels if step.labels.size else data.labels[rows])
        indices.append(step.idx)

    return (
        np.concatenate(images, axis=0),
        np.concatenate(texts, axis=0),
        np.concatenate(labels, axis=0).astype(np.float32),
        np.concatenate(indices, axis=0),
    )


def align_stream_labels_to_feature_order(
    data: CrossModalData,
    train_steps: list[StreamStep],
    test_steps: list[StreamStep],
) -> tuple[list[StreamStep], list[StreamStep], StreamLabelAlignment]:
    """Reorder stream label columns to match ``data.labels`` exactly.

    The mapping is inferred from training samples only. Test labels are used
    solely to verify that the same schema mapping holds; they do not influence
    the inferred permutation.
    """

    width = int(data.labels.shape[1])
    labeled_train = [step for step in train_steps if step.labels.size]
    if not labeled_train:
        identity = tuple(range(width))
        return train_steps, test_steps, StreamLabelAlignment(False, False, identity, 1.0, 1.0)

    _validate_stream_label_width(labeled_train, width, "train")
    _validate_stream_label_width([step for step in test_steps if step.labels.size], width, "test")

    stream_train, feature_train = _paired_stream_and_feature_labels(data, labeled_train)
    mapping = _infer_exact_stream_to_feature_permutation(stream_train, feature_train)
    aligned_train = _reorder_stream_steps(train_steps, mapping)
    aligned_test = _reorder_stream_steps(test_steps, mapping)

    train_accuracy = _stream_feature_alignment_accuracy(data, aligned_train)
    test_accuracy = _stream_feature_alignment_accuracy(data, aligned_test)
    if train_accuracy != 1.0:
        raise ValueError(f"Aligned train stream labels do not match feature labels exactly: {train_accuracy:.8f}")
    if test_accuracy != 1.0:
        raise ValueError(
            "Train-derived stream label permutation does not exactly align test labels: "
            f"{test_accuracy:.8f}"
        )

    identity = tuple(range(width))
    return aligned_train, aligned_test, StreamLabelAlignment(
        enabled=True,
        reordered=mapping != identity,
        stream_to_feature=mapping,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
    )


def _validate_stream_label_width(steps: list[StreamStep], width: int, split: str) -> None:
    for step_idx, step in enumerate(steps):
        if step.labels.shape[1] != width:
            raise ValueError(
                f"{split} stream step {step_idx} has {step.labels.shape[1]} label columns; expected {width}"
            )


def _paired_stream_and_feature_labels(
    data: CrossModalData,
    steps: list[StreamStep],
) -> tuple[np.ndarray, np.ndarray]:
    stream_values: list[np.ndarray] = []
    feature_values: list[np.ndarray] = []
    for step in steps:
        rows = indices_to_rows(data.ids, step.idx, data.images.shape[0])
        stream_values.append((step.labels > 0).astype(np.float32))
        feature_values.append((data.labels[rows] > 0).astype(np.float32))
    return np.concatenate(stream_values, axis=0), np.concatenate(feature_values, axis=0)


def _infer_exact_stream_to_feature_permutation(
    stream_labels: np.ndarray,
    feature_labels: np.ndarray,
) -> tuple[int, ...]:
    if stream_labels.shape != feature_labels.shape:
        raise ValueError(
            "Stream and feature labels must have the same shape to infer a column permutation: "
            f"{stream_labels.shape} vs {feature_labels.shape}"
        )
    n_samples, width = stream_labels.shape
    positive_matches = stream_labels.T @ feature_labels
    negative_matches = (1.0 - stream_labels).T @ (1.0 - feature_labels)
    exact = (positive_matches + negative_matches) == float(n_samples)
    candidate_counts = exact.sum(axis=1)
    if not np.all(candidate_counts == 1):
        ambiguous = np.flatnonzero(candidate_counts != 1).tolist()
        raise ValueError(
            "Could not infer a unique exact stream-to-feature label permutation; "
            f"ambiguous stream columns: {ambiguous}, candidate counts: {candidate_counts.tolist()}"
        )
    mapping = tuple(int(value) for value in exact.argmax(axis=1).tolist())
    if len(set(mapping)) != width:
        raise ValueError(f"Stream-to-feature label mapping is not one-to-one: {mapping}")
    return mapping


def _reorder_stream_steps(steps: list[StreamStep], mapping: tuple[int, ...]) -> list[StreamStep]:
    reordered: list[StreamStep] = []
    for step in steps:
        if not step.labels.size:
            reordered.append(step)
            continue
        labels = np.empty_like(step.labels)
        labels[:, np.asarray(mapping, dtype=np.int64)] = step.labels
        reordered.append(StreamStep(idx=step.idx, labels=labels))
    return reordered


def _stream_feature_alignment_accuracy(data: CrossModalData, steps: list[StreamStep]) -> float:
    labeled = [step for step in steps if step.labels.size]
    if not labeled:
        return 1.0
    stream_values, feature_values = _paired_stream_and_feature_labels(data, labeled)
    return float((stream_values == feature_values).mean())


def indices_to_rows(ids: np.ndarray | None, idx: np.ndarray, n_samples: int) -> np.ndarray:
    idx = np.asarray(idx).reshape(-1).astype(np.int64)

    if ids is not None:
        ids = np.asarray(ids).reshape(-1).astype(np.int64)
        sequential_zero_based = ids.shape[0] == n_samples and np.array_equal(ids, np.arange(n_samples))
        if not sequential_zero_based:
            id_to_row = {int(sample_id): row for row, sample_id in enumerate(ids.tolist())}
            try:
                return np.asarray([id_to_row[int(sample_id)] for sample_id in idx], dtype=np.int64)
            except KeyError:
                pass

    if idx.size == 0:
        return idx
    if int(idx.min()) >= 0 and int(idx.max()) < n_samples:
        return idx
    if int(idx.min()) >= 1 and int(idx.max()) <= n_samples:
        return idx - 1
    raise IndexError(
        f"Cannot map stream indices to feature rows: min={idx.min()}, max={idx.max()}, n={n_samples}"
    )


def pad_label_width(
    data: CrossModalData,
    train_steps: list[StreamStep],
    test_steps: list[StreamStep],
) -> tuple[CrossModalData, list[StreamStep], list[StreamStep]]:
    width = data.labels.shape[1]
    for step in train_steps + test_steps:
        if step.labels.size:
            width = max(width, step.labels.shape[1])
    labels = _pad_matrix(data.labels, width)
    data = CrossModalData(images=data.images, texts=data.texts, labels=labels, ids=data.ids)
    return data, [_pad_step(step, width) for step in train_steps], [_pad_step(step, width) for step in test_steps]


def _resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(path)


def _find_feature_mat(dataset_dir: Path) -> Path:
    mats = [p for p in dataset_dir.glob("*.mat") if p.is_file()]
    if not mats:
        raise FileNotFoundError(f"No feature .mat file found in {dataset_dir}")
    return sorted(mats, key=lambda p: p.stat().st_size, reverse=True)[0]


def _find_stream_mat(stream_dir: Path, train: bool) -> Path:
    mats = [p for p in stream_dir.glob("*.mat") if p.is_file()]
    if not mats:
        raise FileNotFoundError(f"No stream .mat files found in {stream_dir}")

    lowered = [(p, p.name.lower()) for p in mats]
    needles = ["tr", "train"] if train else ["te", "test"]
    matches = [p for p, name in lowered if any(needle in name for needle in needles)]
    if not matches:
        kind = "train" if train else "test"
        raise FileNotFoundError(f"No {kind} stream .mat found in {stream_dir}")
    return sorted(matches)[0]


def _pick_key(keys: list[str], candidates: list[str]) -> str:
    key = _pick_optional_key(keys, candidates)
    if key is None:
        raise KeyError(f"Could not find any of {candidates}; available keys: {keys}")
    return key


def _pick_optional_key(keys: list[str], candidates: list[str]) -> str | None:
    lower_to_key = {k.lower(): k for k in keys}
    for candidate in candidates:
        if candidate.lower() in lower_to_key:
            return lower_to_key[candidate.lower()]
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for key in keys:
            if candidate_lower in key.lower():
                return key
    return None


def _orient_samples(array: np.ndarray, n_samples: int) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(n_samples, 1)
    if array.shape[0] == n_samples:
        return array
    if array.shape[1] == n_samples:
        return array.T
    if array.shape[0] < array.shape[1]:
        return array.T
    return array


def _first_mat_variable(keys: Iterable[str]) -> str:
    for key in keys:
        if not key.startswith("#") and not key.startswith("__"):
            return key
    raise KeyError("No MATLAB variable found in stream file")


def _select_experiment(f: h5py.File, top: h5py.Dataset | h5py.Group, experiment: int) -> h5py.Dataset | h5py.Group:
    if isinstance(top, h5py.Dataset) and top.dtype == object:
        refs = top[()]
        if refs.ndim == 2 and refs.shape[0] == 1:
            ref = refs[0, experiment]
        elif refs.ndim == 2 and refs.shape[1] == 1:
            ref = refs[experiment, 0]
        else:
            ref = refs.reshape(-1)[experiment]
        return f[ref]
    return top


def _extract_object_refs(obj: h5py.Dataset | h5py.Group) -> list[h5py.Reference]:
    if isinstance(obj, h5py.Dataset) and obj.dtype == object:
        return [ref for ref in obj[()].reshape(-1) if ref]
    if isinstance(obj, h5py.Group):
        if "idx" in obj:
            return []
        refs: list[h5py.Reference] = []
        for value in obj.values():
            if isinstance(value, h5py.Dataset) and value.dtype == object:
                refs.extend([ref for ref in value[()].reshape(-1) if ref])
        return refs
    return []


def _select_scipy_experiment(obj: object, experiment: int) -> object:
    if isinstance(obj, np.ndarray):
        if obj.size > 0 and _get_scipy_field(obj.reshape(-1)[0], "idx") is not None:
            return obj
        if obj.dtype == object:
            if obj.ndim == 0:
                return obj.item()
            if obj.ndim == 2 and obj.shape[0] == 1:
                return obj[0, experiment]
            if obj.ndim == 2 and obj.shape[1] == 1:
                return obj[experiment, 0]
            return obj.reshape(-1)[experiment]
        if obj.ndim == 0:
            return obj.item()
    return obj


def _flatten_scipy_cells(obj: object) -> list[object]:
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [item for item in obj.reshape(-1).tolist()]
        if obj.dtype.names:
            return [item for item in obj.reshape(-1).tolist()]
    return [obj]


def _get_scipy_field(obj: object, name: str) -> object | None:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            return _get_scipy_field(obj.item(), name)
        if obj.dtype.names and name in obj.dtype.names:
            return obj[name]
    return None


def _pad_step(step: StreamStep, width: int) -> StreamStep:
    if not step.labels.size:
        return step
    return StreamStep(idx=step.idx, labels=_pad_matrix(step.labels, width))


def _pad_matrix(matrix: np.ndarray, width: int) -> np.ndarray:
    if matrix.shape[1] == width:
        return matrix.astype(np.float32)
    if matrix.shape[1] > width:
        raise ValueError(f"Matrix width {matrix.shape[1]} exceeds target width {width}")
    output = np.zeros((matrix.shape[0], width), dtype=np.float32)
    output[:, : matrix.shape[1]] = matrix.astype(np.float32)
    return output
