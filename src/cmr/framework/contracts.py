from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CrossModalBatch:
    images: np.ndarray
    texts: np.ndarray
    labels: np.ndarray
    ids: np.ndarray

    def __post_init__(self) -> None:
        sizes = {self.images.shape[0], self.texts.shape[0], self.labels.shape[0], self.ids.shape[0]}
        if len(sizes) != 1:
            raise ValueError("CrossModalBatch arrays must contain the same number of samples")
        if self.images.ndim != 2 or self.texts.ndim != 2 or self.labels.ndim != 2:
            raise ValueError("CrossModalBatch features and labels must be two-dimensional")


@dataclass(frozen=True)
class LabelObservation:
    values: np.ndarray
    known_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.known_mask.ndim != 2:
            raise ValueError("LabelObservation arrays must be two-dimensional")
        if self.values.shape != self.known_mask.shape:
            raise ValueError("LabelObservation values and known_mask shapes must match")
        if self.known_mask.dtype != np.bool_:
            raise ValueError("LabelObservation.known_mask must be boolean")


@dataclass(frozen=True)
class TrainModalityBatch:
    modality: str
    features: np.ndarray
    ids: np.ndarray
    supervision: LabelObservation | None = None

    def __post_init__(self) -> None:
        if not self.modality:
            raise ValueError("TrainModalityBatch.modality must be non-empty")
        if self.features.ndim != 2 or self.ids.ndim != 1:
            raise ValueError("Train modality features must be 2D and IDs must be 1D")
        if self.features.shape[0] != self.ids.shape[0]:
            raise ValueError("Train modality features and IDs must have equal rows")
        if np.unique(self.ids).size != self.ids.size:
            raise ValueError("Train modality IDs must be unique")
        if (
            self.supervision is not None
            and self.supervision.values.shape[0] != self.ids.size
        ):
            raise ValueError("Train modality supervision row count must match IDs")


@dataclass(frozen=True)
class MultiModalTrainBatch:
    modalities: Mapping[str, TrainModalityBatch]

    def __post_init__(self) -> None:
        if not self.modalities:
            raise ValueError("MultiModalTrainBatch.modalities must not be empty")
        for name, batch in self.modalities.items():
            if name != batch.modality:
                raise ValueError(
                    f"Modality mapping key '{name}' does not match batch modality "
                    f"'{batch.modality}'"
                )


@dataclass(frozen=True)
class EvaluationBatch:
    modality: str
    features: np.ndarray
    ids: np.ndarray
    ground_truth: np.ndarray

    def __post_init__(self) -> None:
        sizes = {
            self.features.shape[0],
            self.ids.shape[0],
            self.ground_truth.shape[0],
        }
        if len(sizes) != 1:
            raise ValueError("Evaluation batch arrays must have equal rows")
        if (
            self.features.ndim != 2
            or self.ids.ndim != 1
            or self.ground_truth.ndim != 2
        ):
            raise ValueError("Invalid EvaluationBatch array dimensions")


@dataclass(frozen=True)
class StageContext:
    stage_index: int
    total_stages: int
    active_classes: np.ndarray
    current_classes: np.ndarray
    seen_classes: np.ndarray


@dataclass(frozen=True)
class StageTrainResult:
    elapsed_time_s: float
    log: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedModalities:
    image_codes: np.ndarray
    text_codes: np.ndarray
    image_embeddings: np.ndarray | None = None
    text_embeddings: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedModality:
    codes: np.ndarray
    embeddings: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodCapabilities:
    continuous_embeddings: bool = False
    checkpoint: bool = False
    train_log: bool = True
    required_modalities: tuple[str, ...] = ("image", "text")
    supports_independent_modalities: bool = False
    supported_supervision: tuple[str, ...] = ("full",)


@dataclass(frozen=True)
class MethodBuildContext:
    image_dim: int
    text_dim: int
    num_classes: int
    bit: int
    seed: int
    class_names: tuple[str, ...]
    device: str | None = None


@runtime_checkable
class HashMethod(Protocol):
    method_id: str
    display_name: str
    variant: str
    capabilities: MethodCapabilities

    def fit_stage(
        self,
        batch: CrossModalBatch | MultiModalTrainBatch,
        context: StageContext,
    ) -> StageTrainResult:
        ...

    def encode(self, batch: CrossModalBatch, context: StageContext) -> EncodedModalities:
        ...

    def runtime_metadata(self) -> Mapping[str, Any]:
        ...

    def save_checkpoint(self, path: str) -> None:
        ...


def validate_encoded_modalities(encoded: EncodedModalities, sample_count: int, bit: int) -> None:
    for name, values in (
        ("image_codes", encoded.image_codes),
        ("text_codes", encoded.text_codes),
    ):
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if array.shape != (sample_count, bit):
            raise ValueError(
                f"{name} must have shape ({sample_count}, {bit}), got {array.shape}"
            )
        if not np.all(np.isin(array, (-1, 1))):
            raise ValueError(f"{name} must contain only -1 and +1")

    for name, values in (
        ("image_embeddings", encoded.image_embeddings),
        ("text_embeddings", encoded.text_embeddings),
    ):
        if values is None:
            continue
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[0] != sample_count:
            raise ValueError(f"{name} must be two-dimensional with {sample_count} rows")


def validate_encoded_modality(
    encoded: EncodedModality, sample_count: int, bit: int
) -> None:
    codes = np.asarray(encoded.codes)
    if codes.ndim != 2 or codes.shape != (sample_count, bit):
        raise ValueError(
            f"modality codes must have shape ({sample_count}, {bit}), got {codes.shape}"
        )
    if not np.all(np.isin(codes, (-1, 1))):
        raise ValueError("modality codes must contain only -1 and +1")
    if encoded.embeddings is not None:
        values = np.asarray(encoded.embeddings)
        if values.ndim != 2 or values.shape[0] != sample_count:
            raise ValueError(
                f"modality embeddings must be two-dimensional with {sample_count} rows"
            )


def require_continuous_embeddings(
    encoded: EncodedModalities, *, method_id: str
) -> tuple[np.ndarray, np.ndarray]:
    if encoded.image_embeddings is None or encoded.text_embeddings is None:
        raise ValueError(
            f"Method '{method_id}' does not provide continuous embeddings required by this visualization"
        )
    return encoded.image_embeddings, encoded.text_embeddings
