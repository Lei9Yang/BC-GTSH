from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import h5py
import numpy as np


@dataclass(frozen=True)
class ArrayFieldSpec:
    field: str
    sample_axis: int | str = "auto"

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("Dataset array field names must be non-empty")
        if self.sample_axis not in {"auto", 0, 1}:
            raise ValueError("sample_axis must be 'auto', 0, or 1")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: str
    path: str
    modalities: Mapping[str, ArrayFieldSpec]
    labels: ArrayFieldSpec | None
    ids: ArrayFieldSpec | None = None
    class_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DatasetSpec.name must be non-empty")
        if self.loader not in {"mat", "hdf5", "npz"}:
            raise ValueError("DatasetSpec.loader must be 'mat', 'hdf5', or 'npz'")
        if not self.path:
            raise ValueError("DatasetSpec.path must be non-empty")
        if not self.modalities:
            raise ValueError("DatasetSpec.modalities must not be empty")
        if any(not name for name in self.modalities):
            raise ValueError("Dataset modality names must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        def field_dict(value: ArrayFieldSpec | None) -> dict[str, Any] | None:
            if value is None:
                return None
            return {"field": value.field, "sample_axis": value.sample_axis}

        return {
            "name": self.name,
            "loader": self.loader,
            "path": self.path,
            "modalities": {
                name: field_dict(value) for name, value in self.modalities.items()
            },
            "labels": field_dict(self.labels),
            "ids": field_dict(self.ids),
            "class_names": list(self.class_names),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CanonicalDataset:
    ids: np.ndarray
    modalities: Mapping[str, np.ndarray]
    ground_truth: np.ndarray | None
    class_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids)
        if ids.ndim != 1:
            raise ValueError("CanonicalDataset.ids must be one-dimensional")
        if ids.size == 0:
            raise ValueError("CanonicalDataset must contain at least one sample")
        if np.unique(ids).size != ids.size:
            raise ValueError("CanonicalDataset.ids must be unique")
        if not self.modalities:
            raise ValueError("CanonicalDataset.modalities must not be empty")
        for name, values in self.modalities.items():
            array = np.asarray(values)
            if array.ndim != 2 or array.shape[0] != ids.size:
                raise ValueError(
                    f"Modality '{name}' must be two-dimensional with {ids.size} rows"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Modality '{name}' contains non-finite values")
        if self.ground_truth is not None:
            labels = np.asarray(self.ground_truth)
            if labels.ndim != 2 or labels.shape[0] != ids.size:
                raise ValueError(
                    f"ground_truth must be two-dimensional with {ids.size} rows"
                )
            if self.class_names and len(self.class_names) != labels.shape[1]:
                raise ValueError(
                    "class_names length must match the ground-truth label width"
                )

    @property
    def sample_count(self) -> int:
        return int(self.ids.size)

    @property
    def label_width(self) -> int:
        return 0 if self.ground_truth is None else int(self.ground_truth.shape[1])


DatasetLoader = Callable[[DatasetSpec, Path], CanonicalDataset]


class DatasetAdapterRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, DatasetLoader] = {}

    def register(self, name: str, loader: DatasetLoader) -> None:
        key = name.strip().casefold()
        if not key:
            raise ValueError("Dataset adapter names must be non-empty")
        if key in self._loaders:
            raise ValueError(f"Dataset adapter '{key}' is already registered")
        self._loaders[key] = loader

    def load(self, spec: DatasetSpec, workspace_root: Path) -> CanonicalDataset:
        try:
            loader = self._loaders[spec.loader.casefold()]
        except KeyError as error:
            available = ", ".join(sorted(self._loaders))
            raise KeyError(
                f"Unknown dataset adapter '{spec.loader}'. Available: {available}"
            ) from error
        path = _resolve_path(spec.path, workspace_root)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return loader(spec, path)


def dataset_spec_from_mapping(payload: Mapping[str, Any]) -> DatasetSpec:
    allowed = {
        "name",
        "loader",
        "path",
        "modalities",
        "labels",
        "ids",
        "class_names",
        "metadata",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"DatasetSpec.{unknown[0]} is not a recognized field")
    modalities_raw = payload.get("modalities")
    if not isinstance(modalities_raw, Mapping):
        raise ValueError("DatasetSpec.modalities must be an object")
    modalities = {
        str(name): _array_field(value, f"DatasetSpec.modalities.{name}")
        for name, value in modalities_raw.items()
    }
    labels_raw = payload.get("labels")
    ids_raw = payload.get("ids")
    class_names_raw = payload.get("class_names", ())
    if not isinstance(class_names_raw, (list, tuple)):
        raise ValueError("DatasetSpec.class_names must be an array")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("DatasetSpec.metadata must be an object")
    return DatasetSpec(
        name=str(payload.get("name", "")).strip(),
        loader=str(payload.get("loader", "")).strip().casefold(),
        path=str(payload.get("path", "")).strip(),
        modalities=modalities,
        labels=None
        if labels_raw is None
        else _array_field(labels_raw, "DatasetSpec.labels"),
        ids=None if ids_raw is None else _array_field(ids_raw, "DatasetSpec.ids"),
        class_names=tuple(str(value) for value in class_names_raw),
        metadata=dict(metadata),
    )


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Dataset specification root must be an object")
    return dataset_spec_from_mapping(payload)


def default_image_text_spec(
    *,
    name: str,
    path: str,
    class_names: tuple[str, ...] = (),
) -> DatasetSpec:
    suffix = Path(path).suffix.casefold()
    loader = "npz" if suffix == ".npz" else "mat"
    return DatasetSpec(
        name=name,
        loader=loader,
        path=path,
        modalities={
            "image": ArrayFieldSpec("Images"),
            "text": ArrayFieldSpec("Texts"),
        },
        labels=ArrayFieldSpec("Labels"),
        ids=ArrayFieldSpec("Idx"),
        class_names=class_names,
    )


def _array_field(value: Any, path: str) -> ArrayFieldSpec:
    if isinstance(value, str):
        return ArrayFieldSpec(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a string or object")
    unknown = sorted(set(value) - {"field", "sample_axis"})
    if unknown:
        raise ValueError(f"{path}.{unknown[0]} is not a recognized field")
    return ArrayFieldSpec(
        field=str(value.get("field", "")).strip(),
        sample_axis=value.get("sample_axis", "auto"),
    )


def _load_npz(spec: DatasetSpec, path: Path) -> CanonicalDataset:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    return _canonicalize(spec, path, arrays)


def _load_hdf5(spec: DatasetSpec, path: Path) -> CanonicalDataset:
    with h5py.File(path, "r") as payload:
        fields = _required_fields(spec)
        arrays = {name: np.asarray(payload[name][()]) for name in fields}
    return _canonicalize(spec, path, arrays)


def _load_mat(spec: DatasetSpec, path: Path) -> CanonicalDataset:
    try:
        return _load_hdf5(spec, path)
    except OSError:
        import scipy.io as sio

        payload = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
        arrays = {
            name: np.asarray(payload[name])
            for name in _required_fields(spec)
            if name in payload
        }
        return _canonicalize(spec, path, arrays)


def _required_fields(spec: DatasetSpec) -> tuple[str, ...]:
    fields = [value.field for value in spec.modalities.values()]
    if spec.labels is not None:
        fields.append(spec.labels.field)
    if spec.ids is not None:
        fields.append(spec.ids.field)
    return tuple(dict.fromkeys(fields))


def _canonicalize(
    spec: DatasetSpec, source_path: Path, arrays: Mapping[str, np.ndarray]
) -> CanonicalDataset:
    missing = [name for name in _required_fields(spec) if name not in arrays]
    if missing:
        raise ValueError(
            f"Dataset file {source_path} is missing field(s): {', '.join(missing)}"
        )
    raw_ids = None if spec.ids is None else np.asarray(arrays[spec.ids.field]).reshape(-1)
    if raw_ids is not None:
        ids = raw_ids.astype(np.int64, copy=False)
        sample_count = int(ids.size)
    else:
        reference = (
            np.asarray(arrays[spec.labels.field])
            if spec.labels is not None
            else np.asarray(arrays[next(iter(spec.modalities.values())).field])
        )
        sample_count = int(max(reference.shape))
        ids = np.arange(sample_count, dtype=np.int64)

    modalities = {
        name: _orient(
            np.asarray(arrays[field_spec.field], dtype=np.float32),
            sample_count,
            field_spec.sample_axis,
            f"modality '{name}'",
        )
        for name, field_spec in spec.modalities.items()
    }
    ground_truth = None
    if spec.labels is not None:
        ground_truth = _orient(
            np.asarray(arrays[spec.labels.field], dtype=np.float32),
            sample_count,
            spec.labels.sample_axis,
            "labels",
        )
        ground_truth = (ground_truth > 0).astype(np.float32)
    class_names = spec.class_names
    if (
        not class_names
        and ground_truth is not None
        and spec.metadata.get("anonymous_class_names") is True
    ):
        width = max(3, len(str(ground_truth.shape[1] - 1)))
        class_names = tuple(
            f"class_{index:0{width}d}" for index in range(ground_truth.shape[1])
        )
    return CanonicalDataset(
        ids=ids,
        modalities=modalities,
        ground_truth=ground_truth,
        class_names=class_names,
        metadata={
            **dict(spec.metadata),
            "dataset_name": spec.name,
            "source_path": str(source_path.resolve()),
            "loader": spec.loader,
            "generated_ids": spec.ids is None,
        },
    )


def _orient(
    values: np.ndarray,
    sample_count: int,
    sample_axis: int | str,
    name: str,
) -> np.ndarray:
    values = np.squeeze(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional after squeezing")
    if sample_axis == "auto":
        candidates = [axis for axis, size in enumerate(values.shape) if size == sample_count]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot infer sample axis for {name} with shape {values.shape} "
                f"and sample_count={sample_count}"
            )
        sample_axis = candidates[0]
    if values.shape[int(sample_axis)] != sample_count:
        raise ValueError(
            f"{name} sample axis has {values.shape[int(sample_axis)]} rows; "
            f"expected {sample_count}"
        )
    return values if sample_axis == 0 else values.T


def _resolve_path(value: str | Path, workspace_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


dataset_adapters = DatasetAdapterRegistry()
dataset_adapters.register("mat", _load_mat)
dataset_adapters.register("hdf5", _load_hdf5)
dataset_adapters.register("npz", _load_npz)
