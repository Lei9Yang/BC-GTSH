from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset import CanonicalDataset
from .stream import StreamLabelAlignment, StreamStep


@dataclass(frozen=True)
class HoldoutPolicy:
    type: str = "none"
    validation_size: int = 0
    test_size: int = 0
    stratified: bool = True


@dataclass(frozen=True)
class ModalityPolicy:
    type: str = "paired"
    modalities: tuple[str, ...] = ("image", "text")
    ratios: tuple[float, ...] = ()


@dataclass(frozen=True)
class ArrivalPolicy:
    type: str = "iid_stratified"
    num_stages: int = 1
    initial_class_count: int = 0


@dataclass(frozen=True)
class QueryPolicy:
    type: str = "fixed"
    split: str = "test"
    test_ratio: float = 0.1


@dataclass(frozen=True)
class SupervisionPolicy:
    type: str = "full"
    labeled_fraction: float = 1.0
    mask_path: str | None = None


@dataclass(frozen=True)
class ProtocolConfig:
    source: str = "raw"
    preset: str | None = None
    split_seed: int = 6513
    output_dir: str | None = None
    manifest: str | None = None
    train_stream_mat: str | None = None
    test_stream_mat: str | None = None
    experiment: int = 0
    holdout: HoldoutPolicy = HoldoutPolicy()
    modality: ModalityPolicy = ModalityPolicy()
    arrival: ArrivalPolicy = ArrivalPolicy()
    query: QueryPolicy = QueryPolicy()
    supervision: SupervisionPolicy = SupervisionPolicy()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtocolStage:
    train_ids: Mapping[str, np.ndarray]
    query_ids: Mapping[str, np.ndarray]
    supervision_masks: Mapping[str, np.ndarray | None]
    introduced_classes: np.ndarray
    active_classes: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamProtocol:
    protocol_id: str
    split_seed: int
    modalities: tuple[str, ...]
    label_width: int
    stages: tuple[ProtocolStage, ...]
    validation_ids: np.ndarray
    test_ids: np.ndarray
    class_order: np.ndarray
    split_label_usage: str
    training_supervision: str
    query_type: str
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def paired_training(self) -> bool:
        return all(
            all(
                np.array_equal(stage.train_ids[self.modalities[0]], stage.train_ids[name])
                for name in self.modalities[1:]
            )
            for stage in self.stages
        )


def protocol_config_from_mapping(payload: Mapping[str, Any]) -> ProtocolConfig:
    allowed = {
        "source",
        "preset",
        "split_seed",
        "output_dir",
        "manifest",
        "train_stream_mat",
        "test_stream_mat",
        "experiment",
        "holdout",
        "modality",
        "arrival",
        "query",
        "supervision",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"$.experiment.protocol.{unknown[0]} is not recognized")

    merged = _apply_preset(payload)
    config = ProtocolConfig(
        source=str(merged.get("source", "raw")).casefold(),
        preset=merged.get("preset"),
        split_seed=_integer(merged.get("split_seed", 6513), "split_seed"),
        output_dir=_optional_string(merged.get("output_dir")),
        manifest=_optional_string(merged.get("manifest")),
        train_stream_mat=_optional_string(merged.get("train_stream_mat")),
        test_stream_mat=_optional_string(merged.get("test_stream_mat")),
        experiment=_integer(merged.get("experiment", 0), "experiment"),
        holdout=_load_policy(
            HoldoutPolicy, merged.get("holdout", {}), "holdout"
        ),
        modality=_load_modality_policy(merged.get("modality", {})),
        arrival=_load_policy(
            ArrivalPolicy, merged.get("arrival", {}), "arrival"
        ),
        query=_load_policy(QueryPolicy, merged.get("query", {}), "query"),
        supervision=_load_policy(
            SupervisionPolicy, merged.get("supervision", {}), "supervision"
        ),
    )
    _validate_protocol_config(config)
    return config


def compile_protocol(
    dataset: CanonicalDataset,
    config: ProtocolConfig,
    *,
    workspace_root: Path | None = None,
) -> StreamProtocol:
    if dataset.ground_truth is None:
        raise ValueError("Protocol compilation requires evaluation ground truth")
    labels = dataset.ground_truth
    rng = np.random.default_rng(config.split_seed)
    modalities = config.modality.modalities
    missing_modalities = sorted(set(modalities) - set(dataset.modalities))
    if missing_modalities:
        raise ValueError(
            f"Dataset is missing protocol modalities: {', '.join(missing_modalities)}"
        )

    train_pool, validation_ids, test_ids = _apply_holdout(
        dataset, config.holdout, rng
    )
    modality_pools = _apply_modality_policy(
        dataset, train_pool, config.modality, rng
    )
    class_order = (
        rng.permutation(labels.shape[1]).astype(np.int64)
        if config.arrival.type == "class_incremental"
        else np.arange(labels.shape[1], dtype=np.int64)
    )

    scheduled: dict[str, list[np.ndarray]] = {}
    schedule_strata: dict[str, list[np.ndarray]] = {}
    introduced_by_stage: list[np.ndarray] | None = None
    active_by_stage: list[np.ndarray] | None = None
    for modality_index, name in enumerate(modalities):
        if config.modality.type == "paired" and modality_index > 0:
            reference = modalities[0]
            scheduled[name] = [values.copy() for values in scheduled[reference]]
            schedule_strata[name] = [
                values.copy() for values in schedule_strata[reference]
            ]
            continue
        if config.arrival.type == "iid_stratified":
            sizes = _balanced_sizes(len(modality_pools[name]), config.arrival.num_stages)
            scheduled[name] = _multilabel_partition(
                modality_pools[name], dataset, sizes, rng
            )
            schedule_strata[name] = [
                np.zeros(len(values), dtype=np.int8) for values in scheduled[name]
            ]
        else:
            (
                scheduled[name],
                schedule_strata[name],
                introduced,
                active,
            ) = _class_incremental_schedule(
                modality_pools[name],
                dataset,
                class_order,
                config.arrival,
                rng,
            )
            if introduced_by_stage is None:
                introduced_by_stage, active_by_stage = introduced, active

    if introduced_by_stage is None:
        introduced_by_stage = []
        active_by_stage = []
        seen: set[int] = set()
        for stage_index in range(config.arrival.num_stages):
            visible = set()
            for name in modalities:
                rows = _rows_for_ids(dataset, scheduled[name][stage_index])
                visible.update(np.flatnonzero(labels[rows].sum(axis=0) > 0).tolist())
            new = np.asarray(sorted(visible - seen), dtype=np.int64)
            seen.update(visible)
            introduced_by_stage.append(new)
            active_by_stage.append(np.asarray(sorted(seen), dtype=np.int64))
    assert active_by_stage is not None

    train_by_stage, query_by_stage = _apply_query_policy(
        dataset=dataset,
        scheduled=scheduled,
        strata=schedule_strata,
        validation_ids=validation_ids,
        test_ids=test_ids,
        policy=config.query,
        rng=rng,
    )
    masks = _apply_supervision_policy(
        dataset,
        train_by_stage,
        config.supervision,
        rng,
        workspace_root=workspace_root,
    )
    stages = tuple(
        ProtocolStage(
            train_ids={
                name: np.asarray(train_by_stage[name][stage], dtype=np.int64)
                for name in modalities
            },
            query_ids={
                name: np.asarray(query_by_stage[name][stage], dtype=np.int64)
                for name in modalities
            },
            supervision_masks={
                name: masks[name][stage] for name in modalities
            },
            introduced_classes=introduced_by_stage[stage],
            active_classes=active_by_stage[stage],
            metadata={
                "train_counts": {
                    name: int(len(train_by_stage[name][stage])) for name in modalities
                },
                "query_counts": {
                    name: int(len(query_by_stage[name][stage])) for name in modalities
                },
            },
        )
        for stage in range(config.arrival.num_stages)
    )
    supervision_name = {
        "full": "full",
        "none": "none",
        "sample_fraction": "semi",
        "external_mask": "semi",
    }[config.supervision.type]
    source_hash = _dataset_source_hash(dataset)
    identity_payload = {
        "dataset_hash": source_hash,
        "config": config.to_dict(),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    prefix = config.preset or (
        "class-incremental"
        if config.arrival.type == "class_incremental"
        else "stream"
    )
    protocol = StreamProtocol(
        protocol_id=f"{prefix}-{identity}",
        split_seed=config.split_seed,
        modalities=modalities,
        label_width=dataset.label_width,
        stages=stages,
        validation_ids=validation_ids,
        test_ids=test_ids,
        class_order=class_order,
        split_label_usage=(
            "split_and_eval"
            if config.holdout.stratified
            or config.arrival.type in {"iid_stratified", "class_incremental"}
            else "eval_only"
        ),
        training_supervision=supervision_name,
        query_type=config.query.type,
        source="raw",
        metadata={
            "protocol_config": config.to_dict(),
            "dataset_source_sha256": source_hash,
            "dataset_sample_count": dataset.sample_count,
            "label_counts": labels.sum(axis=0).astype(int).tolist(),
        },
    )
    validate_protocol(protocol, dataset)
    return protocol


def resolve_protocol(
    dataset: CanonicalDataset,
    config: ProtocolConfig,
    *,
    workspace_root: Path,
) -> StreamProtocol:
    if config.source == "manifest":
        if not config.manifest:
            raise ValueError("Protocol source 'manifest' requires protocol.manifest")
        return load_protocol(_resolve(config.manifest, workspace_root), dataset)
    if config.source == "legacy_mat":
        raise ValueError(
            "legacy_mat protocols are resolved by the legacy stream adapter"
        )
    if not config.output_dir:
        raise ValueError("Protocol source 'raw' requires protocol.output_dir")
    output_dir = _resolve(config.output_dir, workspace_root)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        protocol = load_protocol(manifest_path, dataset)
        stored = protocol.metadata.get("protocol_config")
        if _canonical_json(stored) != _canonical_json(config.to_dict()):
            raise ValueError(
                f"Existing protocol parameters do not match the run config: {manifest_path}"
            )
        return protocol
    protocol = compile_protocol(dataset, config, workspace_root=workspace_root)
    save_protocol(protocol, dataset, output_dir)
    return load_protocol(manifest_path, dataset)


def save_protocol(
    protocol: StreamProtocol, dataset: CanonicalDataset, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "validation_ids": protocol.validation_ids.astype(np.int64),
        "test_ids": protocol.test_ids.astype(np.int64),
        "class_order": protocol.class_order.astype(np.int64),
    }
    for index, stage in enumerate(protocol.stages):
        arrays[f"stage_{index:03d}__introduced"] = stage.introduced_classes
        arrays[f"stage_{index:03d}__active"] = stage.active_classes
        for name in protocol.modalities:
            arrays[f"stage_{index:03d}__train__{name}"] = stage.train_ids[name]
            arrays[f"stage_{index:03d}__query__{name}"] = stage.query_ids[name]
            mask = stage.supervision_masks[name]
            if mask is not None and protocol.training_supervision == "semi":
                arrays[f"stage_{index:03d}__mask__{name}"] = mask.astype(np.bool_)

    indices_path = output_dir / "indices.npz"
    _write_deterministic_npz(indices_path, arrays)
    indices_hash = sha256_file(indices_path)
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "source": protocol.source,
        "split_seed": protocol.split_seed,
        "modalities": list(protocol.modalities),
        "label_width": protocol.label_width,
        "num_stages": protocol.num_stages,
        "split_label_usage": protocol.split_label_usage,
        "training_supervision": protocol.training_supervision,
        "query_type": protocol.query_type,
        "dataset": {
            "sample_count": dataset.sample_count,
            "source_path": dataset.metadata.get("source_path"),
            "source_sha256": _dataset_source_hash(dataset),
        },
        "indices": {
            "file": "indices.npz",
            "sha256": indices_hash,
            "keys": sorted(arrays),
        },
        "counts": _protocol_counts(protocol),
        "label_statistics": _protocol_label_statistics(protocol, dataset),
        "integrity": _protocol_integrity(protocol, dataset),
        "metadata": dict(protocol.metadata),
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, indices_path


def load_protocol(manifest_path: Path, dataset: CanonicalDataset) -> StreamProtocol:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported protocol schema in {manifest_path}")
    expected_dataset_hash = payload.get("dataset", {}).get("source_sha256")
    actual_dataset_hash = _dataset_source_hash(dataset)
    if expected_dataset_hash != actual_dataset_hash:
        raise ValueError(
            f"Dataset SHA-256 mismatch for protocol {manifest_path}: "
            f"expected {expected_dataset_hash}, got {actual_dataset_hash}"
        )
    indices_info = payload.get("indices", {})
    indices_path = manifest_path.parent / str(indices_info.get("file", "indices.npz"))
    actual_indices_hash = sha256_file(indices_path)
    if actual_indices_hash != indices_info.get("sha256"):
        raise ValueError(f"Protocol indices SHA-256 mismatch: {indices_path}")
    with np.load(indices_path, allow_pickle=False) as npz:
        arrays = {key: np.asarray(npz[key]) for key in npz.files}
    expected_keys = set(indices_info.get("keys", ()))
    if set(arrays) != expected_keys:
        raise ValueError("Protocol NPZ keys do not match manifest")

    modalities = tuple(str(value) for value in payload["modalities"])
    supervision = str(payload["training_supervision"])
    stages: list[ProtocolStage] = []
    for index in range(int(payload["num_stages"])):
        train_ids = {
            name: arrays[f"stage_{index:03d}__train__{name}"].astype(np.int64)
            for name in modalities
        }
        query_ids = {
            name: arrays[f"stage_{index:03d}__query__{name}"].astype(np.int64)
            for name in modalities
        }
        masks: dict[str, np.ndarray | None] = {}
        for name in modalities:
            key = f"stage_{index:03d}__mask__{name}"
            if supervision == "none":
                masks[name] = None
            elif supervision == "full":
                masks[name] = np.ones(
                    (len(train_ids[name]), int(payload["label_width"])), dtype=np.bool_
                )
            else:
                if key not in arrays:
                    raise ValueError(f"Missing semi-supervised mask '{key}'")
                masks[name] = arrays[key].astype(np.bool_)
        stages.append(
            ProtocolStage(
                train_ids=train_ids,
                query_ids=query_ids,
                supervision_masks=masks,
                introduced_classes=arrays[
                    f"stage_{index:03d}__introduced"
                ].astype(np.int64),
                active_classes=arrays[f"stage_{index:03d}__active"].astype(np.int64),
            )
        )
    protocol = StreamProtocol(
        protocol_id=str(payload["protocol_id"]),
        split_seed=int(payload["split_seed"]),
        modalities=modalities,
        label_width=int(payload["label_width"]),
        stages=tuple(stages),
        validation_ids=arrays["validation_ids"].astype(np.int64),
        test_ids=arrays["test_ids"].astype(np.int64),
        class_order=arrays["class_order"].astype(np.int64),
        split_label_usage=str(payload["split_label_usage"]),
        training_supervision=supervision,
        query_type=str(payload["query_type"]),
        source=str(payload.get("source", "manifest")),
        metadata=dict(payload.get("metadata", {})),
    )
    validate_protocol(protocol, dataset)
    return protocol


def legacy_stream_protocol(
    dataset: CanonicalDataset,
    train_steps: Sequence[StreamStep],
    test_steps: Sequence[StreamStep],
    alignment: StreamLabelAlignment,
    *,
    split_seed: int,
    metadata: Mapping[str, Any] | None = None,
) -> StreamProtocol:
    if len(train_steps) != len(test_steps) or not train_steps:
        raise ValueError("Legacy train/test streams must have the same non-zero stage count")
    modalities = tuple(
        name for name in ("image", "text") if name in dataset.modalities
    )
    if len(modalities) != 2:
        raise ValueError("Legacy streams currently require image and text modalities")
    seen: set[int] = set()
    stages: list[ProtocolStage] = []
    for train, test in zip(train_steps, test_steps):
        rows = _rows_for_ids(dataset, np.concatenate([train.idx, test.idx]))
        visible = set(
            np.flatnonzero(dataset.ground_truth[rows].sum(axis=0) > 0).tolist()
        )
        introduced = np.asarray(sorted(visible - seen), dtype=np.int64)
        seen.update(visible)
        masks = {
            name: np.ones((len(train.idx), dataset.label_width), dtype=np.bool_)
            for name in modalities
        }
        stages.append(
            ProtocolStage(
                train_ids={name: train.idx.astype(np.int64) for name in modalities},
                query_ids={name: test.idx.astype(np.int64) for name in modalities},
                supervision_masks=masks,
                introduced_classes=introduced,
                active_classes=np.asarray(sorted(seen), dtype=np.int64),
            )
        )
    source_metadata = {
        "stream_label_alignment": asdict(alignment),
        "class_schedule_source": "inferred",
        **dict(metadata or {}),
    }
    identity = hashlib.sha256(
        json.dumps(source_metadata, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    protocol = StreamProtocol(
        protocol_id=f"legacy-mat-{identity}",
        split_seed=split_seed,
        modalities=modalities,
        label_width=dataset.label_width,
        stages=tuple(stages),
        validation_ids=np.empty(0, dtype=np.int64),
        test_ids=np.concatenate([step.idx for step in test_steps]).astype(np.int64),
        class_order=np.asarray(alignment.stream_to_feature, dtype=np.int64),
        split_label_usage="split_and_eval",
        training_supervision="full",
        query_type="per_stage_split",
        source="legacy_mat",
        metadata=source_metadata,
    )
    validate_protocol(protocol, dataset)
    return protocol


def validate_protocol(protocol: StreamProtocol, dataset: CanonicalDataset) -> None:
    if dataset.ground_truth is None:
        raise ValueError("Protocol validation requires ground truth")
    if protocol.label_width != dataset.label_width:
        raise ValueError(
            f"Protocol label width {protocol.label_width} does not match dataset "
            f"label width {dataset.label_width}"
        )
    if not protocol.stages:
        raise ValueError("Protocol must contain at least one stage")
    if len(set(protocol.modalities)) != len(protocol.modalities):
        raise ValueError("Protocol modality names must be unique")
    known = set(np.asarray(dataset.ids, dtype=np.int64).tolist())
    validation = np.asarray(protocol.validation_ids, dtype=np.int64)
    test = np.asarray(protocol.test_ids, dtype=np.int64)
    _validate_ids(validation, known, "validation")
    _validate_ids(test, known, "test")
    if np.intersect1d(validation, test).size:
        raise ValueError("Validation and test IDs overlap")

    all_train_by_modality: dict[str, list[np.ndarray]] = {
        name: [] for name in protocol.modalities
    }
    for index, stage in enumerate(protocol.stages):
        if not set(stage.introduced_classes).issubset(set(stage.active_classes)):
            raise ValueError(f"Stage {index} introduced classes are not active")
        for name in protocol.modalities:
            if name not in stage.train_ids or name not in stage.query_ids:
                raise ValueError(f"Stage {index} is missing modality '{name}'")
            train_ids = np.asarray(stage.train_ids[name], dtype=np.int64)
            query_ids = np.asarray(stage.query_ids[name], dtype=np.int64)
            _validate_ids(train_ids, known, f"stage {index} {name} train")
            _validate_ids(query_ids, known, f"stage {index} {name} query")
            if np.intersect1d(train_ids, query_ids).size:
                raise ValueError(f"Stage {index} {name} train/query IDs overlap")
            if np.intersect1d(train_ids, validation).size or np.intersect1d(
                train_ids, test
            ).size:
                raise ValueError(f"Stage {index} {name} train IDs overlap holdout IDs")
            mask = stage.supervision_masks[name]
            if protocol.training_supervision == "none":
                if mask is not None:
                    raise ValueError("Unsupervised protocols must not expose label masks")
            else:
                if mask is None or mask.shape != (
                    len(train_ids),
                    dataset.label_width,
                ):
                    raise ValueError(
                        f"Stage {index} {name} supervision mask has the wrong shape"
                    )
            all_train_by_modality[name].append(train_ids)
    for name, parts in all_train_by_modality.items():
        values = np.concatenate(parts)
        if np.unique(values).size != values.size:
            raise ValueError(f"Training IDs repeat across stages for modality '{name}'")
    config = protocol.metadata.get("protocol_config", {})
    modality_type = config.get("modality", {}).get("type")
    if modality_type == "source_disjoint":
        for left_index, left in enumerate(protocol.modalities):
            left_ids = np.concatenate(all_train_by_modality[left])
            for right in protocol.modalities[left_index + 1 :]:
                right_ids = np.concatenate(all_train_by_modality[right])
                if np.intersect1d(left_ids, right_ids).size:
                    raise ValueError(
                        f"Source-disjoint modalities '{left}' and '{right}' overlap"
                    )
        covered = set(validation.tolist()) | set(test.tolist())
        for parts in all_train_by_modality.values():
            values = set(np.concatenate(parts).tolist())
            covered.update(values)
        if covered != known:
            raise ValueError(
                "Source-disjoint train/validation/test sets do not cover the dataset"
            )
    elif modality_type == "paired":
        covered = set(validation.tolist()) | set(test.tolist())
        covered.update(
            np.concatenate(all_train_by_modality[protocol.modalities[0]]).tolist()
        )
        for stage in protocol.stages:
            covered.update(stage.query_ids[protocol.modalities[0]].tolist())
        if covered != known:
            raise ValueError("Paired train/query/holdout sets do not cover the dataset")

    query_config = config.get("query", {})
    if query_config.get("type") == "fixed":
        expected = validation if query_config.get("split") == "validation" else test
        for index, stage in enumerate(protocol.stages):
            for name in protocol.modalities:
                if not np.array_equal(stage.query_ids[name], expected):
                    raise ValueError(
                        f"Stage {index} {name} does not use the fixed query set"
                    )
    if config.get("arrival", {}).get("type") == "class_incremental":
        assert dataset.ground_truth is not None
        for index, stage in enumerate(protocol.stages):
            active = set(stage.active_classes.tolist())
            for name in protocol.modalities:
                ids = stage.train_ids[name]
                if query_config.get("type") == "per_stage_split":
                    ids = np.concatenate([ids, stage.query_ids[name]])
                rows = _rows_for_ids(dataset, ids)
                visible = set(
                    np.flatnonzero(dataset.ground_truth[rows].sum(axis=0) > 0).tolist()
                )
                if not visible.issubset(active):
                    raise ValueError(
                        f"Stage {index} {name} exposes classes before introduction"
                    )
    if config.get("preset") == "flickr_fixed_unpaired":
        assert dataset.ground_truth is not None
        for split_name, ids in (("validation", validation), ("test", test)):
            rows = _rows_for_ids(dataset, ids)
            missing = np.flatnonzero(dataset.ground_truth[rows].sum(axis=0) == 0)
            if missing.size:
                raise ValueError(
                    f"{split_name} is missing classes: {missing.tolist()}"
                )
        for index, stage in enumerate(protocol.stages):
            for name in protocol.modalities:
                rows = _rows_for_ids(dataset, stage.train_ids[name])
                missing = np.flatnonzero(dataset.ground_truth[rows].sum(axis=0) == 0)
                if missing.size:
                    raise ValueError(
                        f"Stage {index} {name} is missing classes: {missing.tolist()}"
                    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_holdout(
    dataset: CanonicalDataset, policy: HoldoutPolicy, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = dataset.ids.astype(np.int64)
    if policy.type == "none":
        empty = np.empty(0, dtype=np.int64)
        return ids.copy(), empty, empty
    sizes = [
        policy.validation_size,
        policy.test_size,
        len(ids) - policy.validation_size - policy.test_size,
    ]
    if policy.stratified:
        validation, test, train = _multilabel_partition(ids, dataset, sizes, rng)
    else:
        order = rng.permutation(ids)
        validation, test, train = _slice_sizes(order, sizes)
    return train, validation, test


def _apply_modality_policy(
    dataset: CanonicalDataset,
    train_pool: np.ndarray,
    policy: ModalityPolicy,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    if policy.type == "paired":
        return {name: train_pool.copy() for name in policy.modalities}
    ratios = np.asarray(policy.ratios, dtype=np.float64)
    raw = ratios / ratios.sum() * len(train_pool)
    sizes = np.floor(raw).astype(int)
    remainder = len(train_pool) - int(sizes.sum())
    order = np.argsort(-(raw - sizes), kind="stable")
    sizes[order[:remainder]] += 1
    parts = _multilabel_partition(train_pool, dataset, sizes.tolist(), rng)
    return dict(zip(policy.modalities, parts))


def _class_incremental_schedule(
    ids: np.ndarray,
    dataset: CanonicalDataset,
    class_order: np.ndarray,
    policy: ArrivalPolicy,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    labels = dataset.ground_truth[_rows_for_ids(dataset, ids)]
    initial = class_order[: policy.initial_class_count]
    new_classes = class_order[policy.initial_class_count :]
    new_sizes = _balanced_sizes(len(new_classes), policy.num_stages)
    groups = _slice_sizes(new_classes, new_sizes)
    class_stage = np.full(dataset.label_width, -1, dtype=np.int64)
    class_stage[initial] = 0
    for stage, group in enumerate(groups):
        class_stage[group] = stage

    sample_min_stage = np.zeros(len(ids), dtype=np.int64)
    old_only = np.ones(len(ids), dtype=np.bool_)
    for row in range(len(ids)):
        positive = np.flatnonzero(labels[row] > 0)
        new_positive = positive[class_stage[positive] >= 0]
        non_initial = new_positive[~np.isin(new_positive, initial)]
        if non_initial.size:
            sample_min_stage[row] = int(class_stage[non_initial].max())
            old_only[row] = False
    stage_rows: list[list[int]] = [[] for _ in range(policy.num_stages)]
    old_rows = rng.permutation(np.flatnonzero(old_only))
    for stage, part in enumerate(_slice_sizes(old_rows, _balanced_sizes(len(old_rows), policy.num_stages))):
        stage_rows[stage].extend(int(value) for value in part)
    counts = np.asarray([len(values) for values in stage_rows], dtype=np.int64)
    for source_stage in range(policy.num_stages):
        ready = rng.permutation(
            np.flatnonzero((~old_only) & (sample_min_stage == source_stage))
        )
        if ready.size:
            stage_rows[source_stage].append(int(ready[0]))
            counts[source_stage] += 1
        for row in ready[1:]:
            eligible = np.arange(source_stage, policy.num_stages)
            target = int(eligible[np.argmin(counts[eligible])])
            stage_rows[target].append(int(row))
            counts[target] += 1
    scheduled: list[np.ndarray] = []
    strata: list[np.ndarray] = []
    for rows in stage_rows:
        row_array = np.asarray(rows, dtype=np.int64)
        old_mask = old_only[row_array]
        order = np.concatenate(
            [np.flatnonzero(old_mask), np.flatnonzero(~old_mask)]
        )
        row_array = row_array[order]
        scheduled.append(ids[row_array].astype(np.int64))
        strata.append((~old_only[row_array]).astype(np.int8))
    introduced = [
        np.concatenate([initial, groups[0]]).astype(np.int64),
        *[group.astype(np.int64) for group in groups[1:]],
    ]
    active: list[np.ndarray] = []
    seen: list[int] = []
    for values in introduced:
        seen.extend(int(value) for value in values)
        active.append(np.asarray(seen, dtype=np.int64))
    return scheduled, strata, introduced, active


def _apply_query_policy(
    *,
    dataset: CanonicalDataset,
    scheduled: Mapping[str, list[np.ndarray]],
    strata: Mapping[str, list[np.ndarray]],
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
    policy: QueryPolicy,
    rng: np.random.Generator,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    modalities = tuple(scheduled)
    if policy.type == "fixed":
        query = validation_ids if policy.split == "validation" else test_ids
        return (
            {name: [values.copy() for values in scheduled[name]] for name in modalities},
            {name: [query.copy() for _ in scheduled[name]] for name in modalities},
        )
    reference = scheduled[modalities[0]]
    if any(
        not all(np.array_equal(left, right) for left, right in zip(reference, scheduled[name]))
        for name in modalities[1:]
    ):
        raise ValueError("per_stage_split requires paired modality stage IDs")
    train_stages: list[np.ndarray] = []
    query_stages: list[np.ndarray] = []
    for values, stage_strata in zip(reference, strata[modalities[0]]):
        train_parts: list[np.ndarray] = []
        query_parts: list[np.ndarray] = []
        for stratum in np.unique(stage_strata):
            part = values[stage_strata == stratum]
            shuffled = rng.permutation(part)
            test_count = int(math.floor(len(part) * policy.test_ratio + 0.5))
            query_parts.append(shuffled[:test_count])
            train_parts.append(shuffled[test_count:])
        train_stages.append(np.concatenate(train_parts).astype(np.int64))
        query_stages.append(np.concatenate(query_parts).astype(np.int64))
    return (
        {name: [values.copy() for values in train_stages] for name in modalities},
        {name: [values.copy() for values in query_stages] for name in modalities},
    )


def _apply_supervision_policy(
    dataset: CanonicalDataset,
    train_by_stage: Mapping[str, list[np.ndarray]],
    policy: SupervisionPolicy,
    rng: np.random.Generator,
    *,
    workspace_root: Path | None,
) -> dict[str, list[np.ndarray | None]]:
    width = dataset.label_width
    if policy.type == "external_mask":
        if not policy.mask_path:
            raise ValueError("external_mask supervision requires mask_path")
        root = workspace_root or Path.cwd()
        with np.load(_resolve(policy.mask_path, root), allow_pickle=False) as payload:
            return {
                name: [
                    np.asarray(payload[f"stage_{index:03d}__mask__{name}"], dtype=np.bool_)
                    for index in range(len(stages))
                ]
                for name, stages in train_by_stage.items()
            }
    output: dict[str, list[np.ndarray | None]] = {}
    for name, stages in train_by_stage.items():
        output[name] = []
        for values in stages:
            if policy.type == "none":
                output[name].append(None)
                continue
            mask = np.ones((len(values), width), dtype=np.bool_)
            if policy.type == "sample_fraction":
                mask[:] = False
                labeled_count = int(
                    math.floor(len(values) * policy.labeled_fraction + 0.5)
                )
                if labeled_count:
                    selected, _ = _multilabel_partition(
                        values,
                        dataset,
                        [labeled_count, len(values) - labeled_count],
                        rng,
                    )
                    selected_rows = {
                        int(value): index for index, value in enumerate(values)
                    }
                    mask[
                        np.asarray([selected_rows[int(value)] for value in selected]),
                        :,
                    ] = True
            output[name].append(mask)
    return output


def _multilabel_partition(
    ids: np.ndarray,
    dataset: CanonicalDataset,
    sizes: Sequence[int],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    sizes_array = np.asarray(sizes, dtype=np.int64)
    if np.any(sizes_array < 0) or int(sizes_array.sum()) != len(ids):
        raise ValueError("Partition sizes must be non-negative and sum to the pool size")
    if not len(ids):
        return [np.empty(0, dtype=np.int64) for _ in sizes]
    labels = dataset.ground_truth[_rows_for_ids(dataset, ids)].astype(np.bool_)
    remaining_capacity = sizes_array.copy()
    desired = (
        sizes_array[:, None] / max(len(ids), 1) * labels.sum(axis=0)[None, :]
    )
    remaining_label = labels.sum(axis=0).astype(np.int64)
    unassigned = np.ones(len(ids), dtype=np.bool_)
    assignments = np.full(len(ids), -1, dtype=np.int64)
    tie_noise = rng.random((len(ids), len(sizes)))

    while np.any(unassigned):
        positive_remaining = np.where(remaining_label > 0)[0]
        if positive_remaining.size == 0:
            break
        label = int(positive_remaining[np.argmin(remaining_label[positive_remaining])])
        candidates = np.flatnonzero(unassigned & labels[:, label])
        if not candidates.size:
            remaining_label[label] = 0
            continue
        cardinality = labels[candidates].sum(axis=1)
        candidates = candidates[
            np.lexsort((tie_noise[candidates, 0], -cardinality))
        ]
        for row in candidates:
            if not unassigned[row]:
                continue
            eligible = np.flatnonzero(remaining_capacity > 0)
            label_need = desired[eligible, label]
            best_need = label_need.max()
            best = eligible[np.isclose(label_need, best_need)]
            if best.size > 1:
                capacity = remaining_capacity[best] / np.maximum(sizes_array[best], 1)
                best = best[np.isclose(capacity, capacity.max())]
            partition = int(best[np.argmin(tie_noise[row, best])])
            assignments[row] = partition
            unassigned[row] = False
            remaining_capacity[partition] -= 1
            desired[partition] -= labels[row].astype(np.float64)
            remaining_label -= labels[row].astype(np.int64)
    for row in np.flatnonzero(unassigned):
        eligible = np.flatnonzero(remaining_capacity > 0)
        partition = int(
            eligible[
                np.argmax(
                    remaining_capacity[eligible]
                    / np.maximum(sizes_array[eligible], 1)
                )
            ]
        )
        assignments[row] = partition
        remaining_capacity[partition] -= 1
    if np.any(remaining_capacity != 0) or np.any(assignments < 0):
        raise RuntimeError("Internal multilabel partitioning failure")
    return [
        ids[np.flatnonzero(assignments == index)].astype(np.int64)
        for index in range(len(sizes))
    ]


def _validate_protocol_config(config: ProtocolConfig) -> None:
    if config.source not in {"raw", "manifest", "legacy_mat"}:
        raise ValueError("protocol.source must be raw, manifest, or legacy_mat")
    if config.split_seed < 0 or config.experiment < 0:
        raise ValueError("Protocol seeds and experiment indices must be non-negative")
    if config.holdout.type not in {"none", "fixed"}:
        raise ValueError("holdout.type must be none or fixed")
    if config.holdout.validation_size < 0 or config.holdout.test_size < 0:
        raise ValueError("Holdout sizes must be non-negative")
    if config.modality.type not in {"paired", "source_disjoint"}:
        raise ValueError("modality.type must be paired or source_disjoint")
    if not config.modality.modalities:
        raise ValueError("modality.modalities must not be empty")
    if config.modality.type == "source_disjoint":
        if len(config.modality.ratios) != len(config.modality.modalities):
            raise ValueError("source_disjoint ratios must match modalities")
        if any(value <= 0 for value in config.modality.ratios):
            raise ValueError("source_disjoint ratios must be positive")
    if config.arrival.type not in {"iid_stratified", "class_incremental"}:
        raise ValueError("arrival.type must be iid_stratified or class_incremental")
    if config.arrival.num_stages <= 0:
        raise ValueError("arrival.num_stages must be positive")
    if config.arrival.type == "class_incremental" and config.arrival.initial_class_count <= 0:
        raise ValueError("class_incremental requires initial_class_count > 0")
    if config.query.type not in {"fixed", "per_stage_split"}:
        raise ValueError("query.type must be fixed or per_stage_split")
    if config.query.split not in {"validation", "test"}:
        raise ValueError("query.split must be validation or test")
    if not 0 <= config.query.test_ratio <= 1:
        raise ValueError("query.test_ratio must be in [0, 1]")
    if config.query.type == "fixed" and config.holdout.type != "fixed":
        raise ValueError("fixed queries require a fixed holdout")
    if config.query.type == "per_stage_split" and config.modality.type != "paired":
        raise ValueError("per_stage_split currently requires paired modalities")
    if config.supervision.type not in {
        "full",
        "none",
        "sample_fraction",
        "external_mask",
    }:
        raise ValueError("Unknown supervision.type")
    if not 0 <= config.supervision.labeled_fraction <= 1:
        raise ValueError("supervision.labeled_fraction must be in [0, 1]")


def _apply_preset(payload: Mapping[str, Any]) -> dict[str, Any]:
    preset = payload.get("preset")
    defaults: dict[str, Any] = {}
    if preset is None and payload.get("source") == "legacy_mat":
        defaults = {
            "holdout": {"type": "none"},
            "modality": {"type": "paired", "modalities": ["image", "text"]},
            "arrival": {"type": "iid_stratified", "num_stages": 1},
            "query": {"type": "per_stage_split", "test_ratio": 0.1},
            "supervision": {"type": "full"},
        }
    elif preset == "class_incremental_matlab":
        defaults = {
            "holdout": {"type": "none"},
            "modality": {"type": "paired", "modalities": ["image", "text"]},
            "arrival": {
                "type": "class_incremental",
                "num_stages": 7,
                "initial_class_count": 3,
            },
            "query": {"type": "per_stage_split", "test_ratio": 0.1},
            "supervision": {"type": "full"},
        }
    elif preset == "flickr_fixed_unpaired":
        defaults = {
            "holdout": {
                "type": "fixed",
                "validation_size": 1000,
                "test_size": 2000,
                "stratified": True,
            },
            "modality": {
                "type": "source_disjoint",
                "modalities": ["image", "text"],
                "ratios": [0.5, 0.5],
            },
            "arrival": {"type": "iid_stratified", "num_stages": 10},
            "query": {"type": "fixed", "split": "validation"},
            "supervision": {"type": "full"},
        }
    elif preset == "unified_fixed_unpaired":
        defaults = {
            "holdout": {
                "type": "fixed",
                "validation_size": 5000,
                "test_size": 5000,
                "stratified": True,
            },
            "modality": {
                "type": "source_disjoint",
                "modalities": ["image", "text"],
                "ratios": [0.5, 0.5],
            },
            "arrival": {"type": "iid_stratified", "num_stages": 10},
            "query": {"type": "fixed", "split": "validation"},
            "supervision": {"type": "full"},
        }
    elif preset is not None:
        raise ValueError(f"Unknown protocol preset '{preset}'")
    merged = dict(defaults)
    for key, value in payload.items():
        if (
            key in {"holdout", "modality", "arrival", "query", "supervision"}
            and isinstance(value, Mapping)
        ):
            merged[key] = {**dict(defaults.get(key, {})), **dict(value)}
        else:
            merged[key] = value
    return merged


def _load_policy(cls: type, value: Any, name: str):
    if not isinstance(value, Mapping):
        raise ValueError(f"protocol.{name} must be an object")
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"protocol.{name}.{unknown[0]} is not recognized")
    return cls(**value)


def _load_modality_policy(value: Any) -> ModalityPolicy:
    if not isinstance(value, Mapping):
        raise ValueError("protocol.modality must be an object")
    allowed = set(ModalityPolicy.__dataclass_fields__)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"protocol.modality.{unknown[0]} is not recognized")
    return ModalityPolicy(
        type=str(value.get("type", "paired")),
        modalities=tuple(str(item) for item in value.get("modalities", ("image", "text"))),
        ratios=tuple(float(item) for item in value.get("ratios", ())),
    )


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"protocol.{name} must be an integer")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Optional protocol paths must be strings")
    return value


def _balanced_sizes(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _slice_sizes(values: np.ndarray, sizes: Sequence[int]) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    start = 0
    for size in sizes:
        output.append(np.asarray(values[start : start + size]))
        start += size
    return output


def _rows_for_ids(dataset: CanonicalDataset, ids: np.ndarray) -> np.ndarray:
    order = np.argsort(dataset.ids)
    sorted_ids = dataset.ids[order]
    positions = np.searchsorted(sorted_ids, ids)
    if np.any(positions >= len(sorted_ids)) or not np.array_equal(
        sorted_ids[np.minimum(positions, len(sorted_ids) - 1)], ids
    ):
        raise ValueError("Protocol contains IDs not found in the dataset")
    return order[positions]


def _validate_ids(values: np.ndarray, known: set[int], name: str) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} IDs must be one-dimensional")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} IDs contain duplicates")
    missing = set(values.tolist()) - known
    if missing:
        raise ValueError(f"{name} contains unknown source IDs")


def _dataset_source_hash(dataset: CanonicalDataset) -> str:
    path = dataset.metadata.get("source_path")
    if path and Path(path).is_file():
        return sha256_file(Path(path))
    digest = hashlib.sha256()
    digest.update(np.asarray(dataset.ids).tobytes())
    for name in sorted(dataset.modalities):
        digest.update(name.encode())
        digest.update(np.asarray(dataset.modalities[name]).tobytes())
    if dataset.ground_truth is not None:
        digest.update(np.asarray(dataset.ground_truth).tobytes())
    return digest.hexdigest()


def _protocol_counts(protocol: StreamProtocol) -> dict[str, Any]:
    return {
        "validation": int(len(protocol.validation_ids)),
        "test": int(len(protocol.test_ids)),
        "stages": [
            {
                "train": {
                    name: int(len(stage.train_ids[name])) for name in protocol.modalities
                },
                "query": {
                    name: int(len(stage.query_ids[name])) for name in protocol.modalities
                },
                "labeled_samples": {
                    name: (
                        0
                        if stage.supervision_masks[name] is None
                        else int(np.any(stage.supervision_masks[name], axis=1).sum())
                    )
                    for name in protocol.modalities
                },
            }
            for stage in protocol.stages
        ],
    }


def _protocol_integrity(
    protocol: StreamProtocol, dataset: CanonicalDataset
) -> dict[str, Any]:
    train_sets = {
        name: set(
            np.concatenate([stage.train_ids[name] for stage in protocol.stages]).tolist()
        )
        for name in protocol.modalities
    }
    overlaps = {}
    for left_index, left in enumerate(protocol.modalities):
        for right in protocol.modalities[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = len(train_sets[left] & train_sets[right])
    return {
        "validated": True,
        "dataset_sample_count": dataset.sample_count,
        "modality_train_overlaps": overlaps,
        "validation_test_overlap": int(
            np.intersect1d(protocol.validation_ids, protocol.test_ids).size
        ),
    }


def _protocol_label_statistics(
    protocol: StreamProtocol, dataset: CanonicalDataset
) -> dict[str, Any]:
    if dataset.ground_truth is None:
        return {}

    def counts(ids: np.ndarray) -> list[int]:
        if not len(ids):
            return [0] * dataset.label_width
        rows = _rows_for_ids(dataset, ids)
        return dataset.ground_truth[rows].sum(axis=0).astype(int).tolist()

    return {
        "validation": counts(protocol.validation_ids),
        "test": counts(protocol.test_ids),
        "stages": [
            {
                "train": {
                    name: counts(stage.train_ids[name]) for name in protocol.modalities
                },
                "query": {
                    name: counts(stage.query_ids[name]) for name in protocol.modalities
                },
            }
            for stage in protocol.stages
        ],
    }


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(value: str | Path, workspace_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
