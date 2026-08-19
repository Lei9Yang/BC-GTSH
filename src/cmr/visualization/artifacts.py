from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ARTIFACT_VERSION = 1
DIRECTIONAL_ARTIFACT_VERSION = 2
ARRAY_FIELDS = (
    "database_indices",
    "query_indices",
    "database_labels",
    "query_labels",
    "database_image_codes",
    "database_text_codes",
    "query_image_codes",
    "query_text_codes",
)


@dataclass(frozen=True)
class VisualizationArtifact:
    method: str
    dataset: str
    bits: int
    seed: int
    stage: int
    database_indices: np.ndarray
    query_indices: np.ndarray
    database_labels: np.ndarray
    query_labels: np.ndarray
    database_image_codes: np.ndarray
    database_text_codes: np.ndarray
    query_image_codes: np.ndarray
    query_text_codes: np.ndarray
    database_image_embeddings: np.ndarray | None = None
    database_text_embeddings: np.ndarray | None = None
    query_image_embeddings: np.ndarray | None = None
    query_text_embeddings: np.ndarray | None = None
    source_path: Path | None = None


@dataclass(frozen=True)
class DirectionalVisualizationArtifact:
    """Visualization payload for source-disjoint directional galleries."""

    method: str
    dataset: str
    bits: int
    seed: int
    stage: int
    gallery_image_ids: np.ndarray
    gallery_text_ids: np.ndarray
    query_ids: np.ndarray
    gallery_image_labels: np.ndarray
    gallery_text_labels: np.ndarray
    query_labels: np.ndarray
    gallery_image_codes: np.ndarray
    gallery_text_codes: np.ndarray
    query_image_codes: np.ndarray
    query_text_codes: np.ndarray
    gallery_image_embeddings: np.ndarray | None = None
    gallery_text_embeddings: np.ndarray | None = None
    query_image_embeddings: np.ndarray | None = None
    query_text_embeddings: np.ndarray | None = None
    source_path: Path | None = None


def _scalar(value: Any) -> Any:
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"Expected scalar metadata, got shape {value.shape}")
    result = value.reshape(-1)[0]
    if isinstance(result, bytes):
        return result.decode("utf-8")
    return result.item() if isinstance(result, np.generic) else result


def _field(container: Any, name: str) -> Any:
    if isinstance(container, Mapping):
        return container[name]
    if hasattr(container, name):
        return getattr(container, name)
    if isinstance(container, np.ndarray) and container.dtype.names and name in container.dtype.names:
        return container[name]
    raise KeyError(name)


def normalize_binary_codes(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values).squeeze()
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D code matrix, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    unique = set(np.unique(array).tolist())
    if unique <= {0, 1}:
        array = np.where(array > 0, 1, -1)
    elif not unique <= {-1, 1}:
        raise ValueError(f"{name} must contain only {{-1,+1}} or {{0,1}}, got {sorted(unique)}")
    return array.astype(np.int8, copy=False)


def artifact_from_mapping(payload: Any, source_path: Path | None = None) -> VisualizationArtifact:
    kwargs: dict[str, Any] = {
        "method": str(_scalar(_field(payload, "method"))),
        "dataset": str(_scalar(_field(payload, "dataset"))).lower(),
        "bits": int(_scalar(_field(payload, "bits"))),
        "seed": int(_scalar(_field(payload, "seed"))),
        "stage": int(_scalar(_field(payload, "stage"))),
        "source_path": source_path,
    }
    for name in ("database_indices", "query_indices"):
        kwargs[name] = np.asarray(_field(payload, name), dtype=np.int64).reshape(-1)
    for name in ("database_labels", "query_labels"):
        value = np.asarray(_field(payload, name))
        if value.ndim != 2:
            raise ValueError(f"{name} must be a 2-D label matrix, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
        kwargs[name] = (value > 0).astype(np.uint8)
    for name in (
        "database_image_codes",
        "database_text_codes",
        "query_image_codes",
        "query_text_codes",
    ):
        kwargs[name] = normalize_binary_codes(_field(payload, name), name=name)
    for name in (
        "database_image_embeddings",
        "database_text_embeddings",
        "query_image_embeddings",
        "query_text_embeddings",
    ):
        try:
            value = np.asarray(_field(payload, name), dtype=np.float32)
        except KeyError:
            value = None
        kwargs[name] = value
    artifact = VisualizationArtifact(**kwargs)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: VisualizationArtifact) -> None:
    if not artifact.method.strip():
        raise ValueError("Artifact method is empty")
    if not artifact.dataset.strip():
        raise ValueError("Artifact dataset is empty")
    if artifact.bits <= 0:
        raise ValueError("Artifact bit length must be positive")
    if artifact.stage <= 0:
        raise ValueError("Artifact stage must be positive")
    n_database = artifact.database_indices.size
    n_query = artifact.query_indices.size
    if len(np.unique(artifact.database_indices)) != n_database:
        raise ValueError(f"Duplicate database indices in {artifact.method}")
    if len(np.unique(artifact.query_indices)) != n_query:
        raise ValueError(f"Duplicate query indices in {artifact.method}")
    if np.intersect1d(artifact.database_indices, artifact.query_indices).size:
        raise ValueError(f"Database/query indices overlap in {artifact.method}")
    database_arrays = (
        artifact.database_labels,
        artifact.database_image_codes,
        artifact.database_text_codes,
    )
    query_arrays = (
        artifact.query_labels,
        artifact.query_image_codes,
        artifact.query_text_codes,
    )
    if any(array.shape[0] != n_database for array in database_arrays):
        raise ValueError(f"Database row-count mismatch in {artifact.method}")
    if any(array.shape[0] != n_query for array in query_arrays):
        raise ValueError(f"Query row-count mismatch in {artifact.method}")
    if artifact.database_labels.shape[1] != artifact.query_labels.shape[1]:
        raise ValueError(f"Database/query label dimensions differ in {artifact.method}")
    for name in (
        "database_image_codes",
        "database_text_codes",
        "query_image_codes",
        "query_text_codes",
    ):
        if getattr(artifact, name).shape[1] != artifact.bits:
            raise ValueError(f"{name} is not {artifact.bits}-bit in {artifact.method}")
    for name, expected_rows in (
        ("database_image_embeddings", n_database),
        ("database_text_embeddings", n_database),
        ("query_image_embeddings", n_query),
        ("query_text_embeddings", n_query),
    ):
        values = getattr(artifact, name)
        if values is not None and (values.ndim != 2 or values.shape[0] != expected_rows):
            raise ValueError(f"{name} row-count mismatch in {artifact.method}")


def save_visualization_npz(path: str | Path, artifact: VisualizationArtifact) -> Path:
    validate_artifact(artifact)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "artifact_version": np.asarray(ARTIFACT_VERSION, dtype=np.int64),
        "method": np.asarray(artifact.method),
        "dataset": np.asarray(artifact.dataset),
        "bits": np.asarray(artifact.bits, dtype=np.int64),
        "seed": np.asarray(artifact.seed, dtype=np.int64),
        "stage": np.asarray(artifact.stage, dtype=np.int64),
        **{name: getattr(artifact, name) for name in ARRAY_FIELDS},
    }
    for name in (
        "database_image_embeddings",
        "database_text_embeddings",
        "query_image_embeddings",
        "query_text_embeddings",
    ):
        values = getattr(artifact, name)
        if values is not None:
            payload[name] = values
    np.savez_compressed(target, **payload)
    return target


def validate_directional_artifact(artifact: DirectionalVisualizationArtifact) -> None:
    if not artifact.method.strip() or not artifact.dataset.strip():
        raise ValueError("Directional visualization method and dataset must be non-empty")
    if artifact.bits <= 0 or artifact.stage <= 0:
        raise ValueError("Directional visualization bits and stage must be positive")
    sizes = {
        "gallery_image": artifact.gallery_image_ids.size,
        "gallery_text": artifact.gallery_text_ids.size,
        "query": artifact.query_ids.size,
    }
    for name, ids in (
        ("gallery_image", artifact.gallery_image_ids),
        ("gallery_text", artifact.gallery_text_ids),
        ("query", artifact.query_ids),
    ):
        if len(np.unique(ids)) != ids.size:
            raise ValueError(f"Duplicate {name} IDs in {artifact.method}")
    for name, gallery_ids in (
        ("image", artifact.gallery_image_ids),
        ("text", artifact.gallery_text_ids),
    ):
        if np.intersect1d(gallery_ids, artifact.query_ids).size:
            raise ValueError(f"{name} gallery/query IDs overlap in {artifact.method}")
    arrays = (
        ("gallery_image_labels", artifact.gallery_image_labels, sizes["gallery_image"]),
        ("gallery_text_labels", artifact.gallery_text_labels, sizes["gallery_text"]),
        ("query_labels", artifact.query_labels, sizes["query"]),
        ("gallery_image_codes", artifact.gallery_image_codes, sizes["gallery_image"]),
        ("gallery_text_codes", artifact.gallery_text_codes, sizes["gallery_text"]),
        ("query_image_codes", artifact.query_image_codes, sizes["query"]),
        ("query_text_codes", artifact.query_text_codes, sizes["query"]),
    )
    for name, values, expected_rows in arrays:
        if values.ndim != 2 or values.shape[0] != expected_rows:
            raise ValueError(
                f"{name} row-count mismatch in {artifact.method}: "
                f"{values.shape} vs {expected_rows}"
            )
    label_widths = {
        artifact.gallery_image_labels.shape[1],
        artifact.gallery_text_labels.shape[1],
        artifact.query_labels.shape[1],
    }
    if len(label_widths) != 1:
        raise ValueError(f"Directional label widths differ in {artifact.method}")
    for name in (
        "gallery_image_codes",
        "gallery_text_codes",
        "query_image_codes",
        "query_text_codes",
    ):
        values = getattr(artifact, name)
        if values.shape[1] != artifact.bits:
            raise ValueError(f"{name} is not {artifact.bits}-bit in {artifact.method}")
        normalize_binary_codes(values, name=name)
    for name, expected_rows in (
        ("gallery_image_embeddings", sizes["gallery_image"]),
        ("gallery_text_embeddings", sizes["gallery_text"]),
        ("query_image_embeddings", sizes["query"]),
        ("query_text_embeddings", sizes["query"]),
    ):
        values = getattr(artifact, name)
        if values is not None and (values.ndim != 2 or values.shape[0] != expected_rows):
            raise ValueError(f"{name} row-count mismatch in {artifact.method}")


def save_directional_visualization_npz(
    path: str | Path, artifact: DirectionalVisualizationArtifact
) -> Path:
    validate_directional_artifact(artifact)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "artifact_version": np.asarray(DIRECTIONAL_ARTIFACT_VERSION, dtype=np.int64),
        "method": np.asarray(artifact.method),
        "dataset": np.asarray(artifact.dataset),
        "bits": np.asarray(artifact.bits, dtype=np.int64),
        "seed": np.asarray(artifact.seed, dtype=np.int64),
        "stage": np.asarray(artifact.stage, dtype=np.int64),
    }
    for name in (
        "gallery_image_ids",
        "gallery_text_ids",
        "query_ids",
        "gallery_image_labels",
        "gallery_text_labels",
        "query_labels",
        "gallery_image_codes",
        "gallery_text_codes",
        "query_image_codes",
        "query_text_codes",
        "gallery_image_embeddings",
        "gallery_text_embeddings",
        "query_image_embeddings",
        "query_text_embeddings",
    ):
        values = getattr(artifact, name)
        if values is not None:
            payload[name] = values
    np.savez_compressed(target, **payload)
    return target


def load_directional_visualization_artifact(
    path: str | Path,
) -> DirectionalVisualizationArtifact:
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported directional visualization artifact: {source}")
    with np.load(source, allow_pickle=False) as payload:
        version = int(_scalar(payload["artifact_version"]))
        if version != DIRECTIONAL_ARTIFACT_VERSION:
            raise ValueError(f"Expected directional artifact v2 in {source}, got v{version}")
        kwargs: dict[str, Any] = {
            "method": str(_scalar(payload["method"])),
            "dataset": str(_scalar(payload["dataset"])).lower(),
            "bits": int(_scalar(payload["bits"])),
            "seed": int(_scalar(payload["seed"])),
            "stage": int(_scalar(payload["stage"])),
            "source_path": source,
        }
        for name in ("gallery_image_ids", "gallery_text_ids", "query_ids"):
            kwargs[name] = np.asarray(payload[name], dtype=np.int64).reshape(-1)
        for name in (
            "gallery_image_labels",
            "gallery_text_labels",
            "query_labels",
        ):
            kwargs[name] = (np.asarray(payload[name]) > 0).astype(np.uint8)
        for name in (
            "gallery_image_codes",
            "gallery_text_codes",
            "query_image_codes",
            "query_text_codes",
        ):
            kwargs[name] = normalize_binary_codes(payload[name], name=name)
        for name in (
            "gallery_image_embeddings",
            "gallery_text_embeddings",
            "query_image_embeddings",
            "query_text_embeddings",
        ):
            kwargs[name] = (
                np.asarray(payload[name], dtype=np.float32) if name in payload else None
            )
    artifact = DirectionalVisualizationArtifact(**kwargs)
    validate_directional_artifact(artifact)
    return artifact


def validate_shared_directional_artifacts(
    artifacts: list[DirectionalVisualizationArtifact],
) -> None:
    if not artifacts:
        raise ValueError("No directional visualization artifacts supplied")
    reference = artifacts[0]
    for artifact in artifacts[1:]:
        for name in ("dataset", "bits", "stage"):
            if getattr(artifact, name) != getattr(reference, name):
                raise ValueError(
                    f"Directional artifact {name} mismatch: "
                    f"{reference.method} vs {artifact.method}"
                )
        for name in (
            "gallery_image_ids",
            "gallery_text_ids",
            "query_ids",
            "gallery_image_labels",
            "gallery_text_labels",
            "query_labels",
        ):
            if not np.array_equal(getattr(artifact, name), getattr(reference, name)):
                raise ValueError(
                    f"Directional artifact {name} mismatch: "
                    f"{reference.method} vs {artifact.method}"
                )


def load_visualization_artifact(path: str | Path) -> VisualizationArtifact:
    source = Path(path)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            if int(_scalar(payload["artifact_version"])) != ARTIFACT_VERSION:
                raise ValueError(f"Unsupported artifact version in {source}")
            return artifact_from_mapping(payload, source)
    if source.suffix.lower() == ".mat":
        from scipy.io import loadmat

        payload = loadmat(source, squeeze_me=True, struct_as_record=False)
        container = payload.get("artifact", payload)
        return artifact_from_mapping(container, source)
    raise ValueError(f"Unsupported visualization artifact: {source}")


def validate_shared_artifacts(artifacts: list[VisualizationArtifact]) -> None:
    if not artifacts:
        raise ValueError("No visualization artifacts supplied")
    reference = artifacts[0]
    for artifact in artifacts[1:]:
        # The training seed is provenance, not an artifact-alignment constraint.
        for name in ("dataset", "bits", "stage"):
            if getattr(artifact, name) != getattr(reference, name):
                raise ValueError(f"Artifact {name} mismatch: {reference.method} vs {artifact.method}")
        for name in ("database_indices", "query_indices", "database_labels", "query_labels"):
            if not np.array_equal(getattr(artifact, name), getattr(reference, name)):
                raise ValueError(f"Artifact {name} mismatch: {reference.method} vs {artifact.method}")


def hamming_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_binary_codes(left, name="left_codes")
    right = normalize_binary_codes(right, name="right_codes")
    if left.shape != right.shape:
        raise ValueError(f"Paired code shapes differ: {left.shape} vs {right.shape}")
    return 1.0 - np.mean(left != right, axis=1)
