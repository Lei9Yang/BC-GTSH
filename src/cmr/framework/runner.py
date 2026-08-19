from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cmr.data import (
    CanonicalDataset,
    CrossModalData,
    DatasetSpec,
    ProtocolConfig,
    StreamProtocol,
    StreamStep,
    align_stream_labels_to_feature_order,
    dataset_adapters,
    default_image_text_spec,
    indices_to_rows,
    legacy_stream_protocol,
    load_cross_modal_mat,
    load_dataset_spec,
    load_stream_steps,
    materialize_steps,
    pad_label_width,
    resolve_dataset_files,
    resolve_protocol,
    sha256_file,
)
from cmr.evaluation import bounded_ks, compute_retrieval_metrics, save_json, set_seed
from cmr.visualization import (
    DirectionalVisualizationArtifact,
    VisualizationArtifact,
    save_directional_visualization_npz,
    save_visualization_npz,
)

from .config import RunConfig
from .contracts import (
    CrossModalBatch,
    EncodedModalities,
    EncodedModality,
    EvaluationBatch,
    LabelObservation,
    MethodBuildContext,
    MultiModalTrainBatch,
    StageContext,
    TrainModalityBatch,
    validate_encoded_modalities,
    validate_encoded_modality,
)
from .registry import MethodRegistry, registry


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    manifest_path: Path
    metrics_i2t: dict[str, Any]
    metrics_t2i: dict[str, Any]
    train_log: list[dict[str, Any]]


@dataclass
class _EncodingCache:
    image_codes: np.ndarray | None = None
    text_codes: np.ndarray | None = None
    labels: np.ndarray | None = None
    ids: np.ndarray | None = None
    image_embeddings: np.ndarray | None = None
    text_embeddings: np.ndarray | None = None


@dataclass
class _ModalityCache:
    codes: np.ndarray | None = None
    labels: np.ndarray | None = None
    ids: np.ndarray | None = None
    embeddings: np.ndarray | None = None


class ExperimentRunner:
    def __init__(
        self,
        config: RunConfig,
        *,
        workspace_root: str | Path | None = None,
        method_registry: MethodRegistry | None = None,
    ) -> None:
        self.config = config
        self.workspace_root = Path(workspace_root or Path.cwd()).expanduser().resolve()
        self.registry = method_registry or registry

    def run(self) -> RunResult:
        if self.config.experiment.protocol is None:
            experiment = self.config.experiment
            legacy = ProtocolConfig(
                source="legacy_mat",
                split_seed=experiment.seed,
                train_stream_mat=experiment.train_stream_mat,
                test_stream_mat=experiment.test_stream_mat,
                experiment=experiment.experiment,
            )
            self.config = replace(
                self.config,
                experiment=replace(experiment, protocol=legacy),
            )
            self._implicit_legacy_protocol = True
        return self._run_protocol()

    def _run_legacy(self) -> RunResult:
        """Retained implementation reference; all public runs use StreamProtocol."""
        config = self.config
        experiment = config.experiment
        if self.registry is registry:
            from cmr.methods import load_builtin_methods

            load_builtin_methods()
        set_seed(experiment.seed)
        data_root = self._resolve(experiment.data_root)
        files = resolve_dataset_files(
            experiment.dataset,
            data_root,
            experiment.feature_mat,
            experiment.train_stream_mat,
            experiment.test_stream_mat,
        )
        data = load_cross_modal_mat(files.feature_mat)
        train_steps = load_stream_steps(
            files.train_stream_mat, experiment=experiment.experiment
        )
        test_steps = load_stream_steps(
            files.test_stream_mat, experiment=experiment.experiment
        )
        data, train_steps, test_steps = pad_label_width(data, train_steps, test_steps)
        train_steps, test_steps, alignment = align_stream_labels_to_feature_order(
            data, train_steps, test_steps
        )
        if experiment.normalize_input_features:
            data = _normalize_data(data)

        total_stages = min(len(train_steps), len(test_steps))
        if experiment.max_stages is not None:
            total_stages = min(total_stages, experiment.max_stages)
        if total_stages <= 0:
            raise ValueError("No stream stages available")

        class_names = self._load_class_names(data.labels.shape[1])
        build_context = MethodBuildContext(
            image_dim=int(data.images.shape[1]),
            text_dim=int(data.texts.shape[1]),
            num_classes=int(data.labels.shape[1]),
            bit=experiment.bit,
            seed=experiment.seed,
            class_names=class_names,
            device=experiment.device,
        )
        method_parameters = dict(config.method_parameters)
        method_parameters["variant"] = config.method_variant
        method = self.registry.create(config.method_id, method_parameters, build_context)
        run_dir = self._run_dir(method.variant)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        manifest_path = run_dir / "manifest.json"
        metrics_i2t_path = run_dir / "metrics_i2t.json"
        metrics_t2i_path = run_dir / "metrics_t2i.json"
        train_log_path = run_dir / "train_log.json"
        save_json(config_path, config.to_dict(), save_ap=True)
        manifest = self._manifest(
            method,
            files={
                "config": config_path,
                "metrics_i2t": metrics_i2t_path,
                "metrics_t2i": metrics_t2i_path,
                "train_log": train_log_path,
            },
            total_stages=total_stages,
            alignment=asdict(alignment),
            status="running",
        )
        save_json(manifest_path, manifest, save_ap=True)

        seen_classes: set[int] = set()
        cache = _EncodingCache()
        metrics_i2t: dict[str, Any] = {}
        metrics_t2i: dict[str, Any] = {}
        train_log: list[dict[str, Any]] = []

        try:
            for stage_index in range(total_stages):
                train_batch = _batch_from_steps(data, [train_steps[stage_index]])
                old_cache = cache
                current_classes = np.flatnonzero(
                    train_batch.labels.sum(axis=0) > 0
                ).astype(np.int64)
                seen_classes.update(int(item) for item in current_classes)
                active_classes = np.asarray(sorted(seen_classes), dtype=np.int64)
                stage_context = StageContext(
                    stage_index=stage_index,
                    total_stages=total_stages,
                    active_classes=active_classes,
                    current_classes=current_classes,
                    seen_classes=active_classes,
                )
                stage_result = method.fit_stage(train_batch, stage_context)
                log = dict(stage_result.log)
                log.setdefault("stage", stage_index)
                log.setdefault("train_time", float(stage_result.elapsed_time_s))
                if stage_result.metadata:
                    log["method_metadata"] = dict(stage_result.metadata)
                train_log.append(log)
                if config.output.save_train_log:
                    save_json(train_log_path, train_log, save_ap=True)

                if config.evaluation.retrieval_database_mode == "append":
                    encoded_current, _ = self._encode(method, train_batch, stage_context)
                    cache = _append_cache(cache, encoded_current, train_batch)

                should_evaluate = (
                    config.evaluation.retrieval_eval_mode == "per_stage"
                    or stage_index == total_stages - 1
                )
                if not should_evaluate:
                    continue
                query_steps = (
                    [test_steps[stage_index]]
                    if config.evaluation.query_mode == "current"
                    else test_steps[: stage_index + 1]
                )
                query_batch = _batch_from_steps(data, query_steps)
                query_encoded, query_encode_time = self._encode(
                    method, query_batch, stage_context
                )
                if config.evaluation.retrieval_database_mode == "recompute":
                    database_batch = _batch_from_steps(
                        data, train_steps[: stage_index + 1]
                    )
                    database_encoded, database_encode_time = self._encode(
                        method, database_batch, stage_context
                    )
                    cache = _cache_from(database_encoded, database_batch)
                else:
                    database_encode_time = 0.0
                if cache.labels is None or cache.image_codes is None or cache.text_codes is None:
                    raise RuntimeError("Retrieval database is empty")

                common_efficiency = {
                    **dict(method.runtime_metadata()),
                    "train_time_s": float(stage_result.elapsed_time_s),
                    "query_encode_time_s": float(query_encode_time),
                    "database_encode_time_s": float(database_encode_time),
                    "query_count": int(query_batch.labels.shape[0]),
                    "database_size": int(cache.labels.shape[0]),
                }
                top_k = bounded_ks(list(config.evaluation.top_k), cache.labels.shape[0])
                precision_ks = bounded_ks(
                    list(config.evaluation.precision_curve_ks), cache.labels.shape[0]
                )
                ndcg_ks = [
                    min(value, cache.labels.shape[0])
                    for value in config.evaluation.ndcg_ks
                ]
                i2t = compute_retrieval_metrics(
                    query_labels=query_batch.labels,
                    retrieval_labels=cache.labels,
                    query_codes=query_encoded.image_codes,
                    retrieval_codes=cache.text_codes,
                    train_elapsed_time=stage_result.elapsed_time_s,
                    hashcode_generate_time=query_encode_time + database_encode_time,
                    top_k=top_k,
                    ratio_n=list(config.evaluation.ratio_n),
                    ndcg_ks=ndcg_ks,
                    precision_curve_ks=precision_ks,
                    efficiency=dict(common_efficiency),
                )
                t2i = compute_retrieval_metrics(
                    query_labels=query_batch.labels,
                    retrieval_labels=cache.labels,
                    query_codes=query_encoded.text_codes,
                    retrieval_codes=cache.image_codes,
                    train_elapsed_time=stage_result.elapsed_time_s,
                    hashcode_generate_time=query_encode_time + database_encode_time,
                    top_k=top_k,
                    ratio_n=list(config.evaluation.ratio_n),
                    ndcg_ks=ndcg_ks,
                    precision_curve_ks=precision_ks,
                    efficiency=dict(common_efficiency),
                )
                if (
                    config.evaluation.retrieval_database_mode == "append"
                    and old_cache.labels is not None
                    and old_cache.image_codes is not None
                    and old_cache.text_codes is not None
                ):
                    old_database_size = int(old_cache.labels.shape[0])
                    old_top_k = bounded_ks(
                        list(config.evaluation.top_k), old_database_size
                    )
                    old_precision_ks = bounded_ks(
                        list(config.evaluation.precision_curve_ks),
                        old_database_size,
                    )
                    old_ndcg_ks = [
                        min(value, old_database_size)
                        for value in config.evaluation.ndcg_ks
                    ]
                    old_efficiency = {
                        **dict(common_efficiency),
                        "evaluation_scope": "frozen_old_gallery",
                        "database_size": old_database_size,
                    }
                    i2t["old_gallery"] = compute_retrieval_metrics(
                        query_labels=query_batch.labels,
                        retrieval_labels=old_cache.labels,
                        query_codes=query_encoded.image_codes,
                        retrieval_codes=old_cache.text_codes,
                        train_elapsed_time=stage_result.elapsed_time_s,
                        hashcode_generate_time=query_encode_time,
                        top_k=old_top_k,
                        ratio_n=list(config.evaluation.ratio_n),
                        ndcg_ks=old_ndcg_ks,
                        precision_curve_ks=old_precision_ks,
                        efficiency=dict(old_efficiency),
                    )
                    t2i["old_gallery"] = compute_retrieval_metrics(
                        query_labels=query_batch.labels,
                        retrieval_labels=old_cache.labels,
                        query_codes=query_encoded.text_codes,
                        retrieval_codes=old_cache.image_codes,
                        train_elapsed_time=stage_result.elapsed_time_s,
                        hashcode_generate_time=query_encode_time,
                        top_k=old_top_k,
                        ratio_n=list(config.evaluation.ratio_n),
                        ndcg_ks=old_ndcg_ks,
                        precision_curve_ks=old_precision_ks,
                        efficiency=dict(old_efficiency),
                    )
                directional_times = {
                    "search_time_i2t_s": float(i2t["efficiency"]["search_time_s"]),
                    "search_time_t2i_s": float(t2i["efficiency"]["search_time_s"]),
                    "metric_time_i2t_s": float(i2t["efficiency"]["metric_time_s"]),
                    "metric_time_t2i_s": float(t2i["efficiency"]["metric_time_s"]),
                }
                i2t["efficiency"].update(directional_times)
                t2i["efficiency"].update(directional_times)
                key = str(stage_index)
                metrics_i2t[key] = i2t
                metrics_t2i[key] = t2i
                save_json(metrics_i2t_path, metrics_i2t, save_ap=config.output.save_ap)
                save_json(metrics_t2i_path, metrics_t2i, save_ap=config.output.save_ap)
                if self._should_export_visualization(stage_index, total_stages):
                    self._save_visualization(
                        method=method,
                        stage_index=stage_index,
                        query_batch=query_batch,
                        query_encoded=query_encoded,
                        cache=cache,
                    )

            checkpoint_path = run_dir / "checkpoint.pt"
            if config.output.save_model:
                if not method.capabilities.checkpoint:
                    raise ValueError(
                        f"Method '{method.method_id}' does not support checkpoints"
                    )
                method.save_checkpoint(str(checkpoint_path))
                manifest["files"]["checkpoint"] = str(checkpoint_path)
            manifest["status"] = "completed"
            manifest["completed_stages"] = total_stages
            save_json(manifest_path, manifest, save_ap=True)
        except Exception:
            manifest["status"] = "failed"
            manifest["completed_stages"] = len(train_log)
            save_json(manifest_path, manifest, save_ap=True)
            raise

        return RunResult(
            run_dir=run_dir,
            manifest_path=manifest_path,
            metrics_i2t=metrics_i2t,
            metrics_t2i=metrics_t2i,
            train_log=train_log,
        )

    def _run_protocol(self) -> RunResult:
        config = self.config
        experiment = config.experiment
        protocol_config = experiment.protocol
        assert protocol_config is not None
        if self.registry is registry:
            from cmr.methods import load_builtin_methods

            load_builtin_methods()
        set_seed(experiment.seed)
        dataset, protocol = self._load_protocol_dataset()
        self._active_protocol_label_width = protocol.label_width
        if experiment.normalize_input_features:
            dataset = _normalize_canonical_dataset(dataset)
        total_stages = protocol.num_stages
        if experiment.max_stages is not None:
            total_stages = min(total_stages, experiment.max_stages)
        if total_stages <= 0:
            raise ValueError("No protocol stages available")

        class_names = (
            dataset.class_names
            if dataset.class_names
            else self._load_class_names(dataset.label_width)
        )
        for name in ("image", "text"):
            if name not in dataset.modalities:
                raise ValueError(
                    f"The current cross-modal hashing runner requires modality '{name}'"
                )
        build_context = MethodBuildContext(
            image_dim=int(dataset.modalities["image"].shape[1]),
            text_dim=int(dataset.modalities["text"].shape[1]),
            num_classes=dataset.label_width,
            bit=experiment.bit,
            seed=experiment.seed,
            class_names=class_names,
            device=experiment.device,
        )
        method_parameters = dict(config.method_parameters)
        method_parameters["variant"] = config.method_variant
        method = self.registry.create(config.method_id, method_parameters, build_context)
        self._validate_method_protocol(method, protocol)
        run_dir = self._run_dir(method.variant, protocol_id=protocol.protocol_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        manifest_path = run_dir / "manifest.json"
        metrics_i2t_path = run_dir / "metrics_i2t.json"
        metrics_t2i_path = run_dir / "metrics_t2i.json"
        train_log_path = run_dir / "train_log.json"
        save_json(config_path, config.to_dict(), save_ap=True)
        manifest = {
            "schema_version": 1,
            "status": "running",
            "method": {
                "id": method.method_id,
                "display_name": method.display_name,
                "variant": method.variant,
                "capabilities": asdict(method.capabilities),
            },
            "experiment": asdict(experiment),
            "protocol": {
                "id": protocol.protocol_id,
                "source": protocol.source,
                "split_seed": protocol.split_seed,
                "query_type": protocol.query_type,
                "training_supervision": protocol.training_supervision,
                "split_label_usage": protocol.split_label_usage,
                "metadata": dict(protocol.metadata),
            },
            "model_seed": experiment.seed,
            "total_stages": total_stages,
            "completed_stages": 0,
            "files": {
                "config": str(config_path),
                "metrics_i2t": str(metrics_i2t_path),
                "metrics_t2i": str(metrics_t2i_path),
                "train_log": str(train_log_path),
            },
        }
        save_json(manifest_path, manifest, save_ap=True)

        caches = {"image": _ModalityCache(), "text": _ModalityCache()}
        metrics_i2t: dict[str, Any] = {}
        metrics_t2i: dict[str, Any] = {}
        train_log: list[dict[str, Any]] = []
        try:
            for stage_index in range(total_stages):
                stage = protocol.stages[stage_index]
                train_modalities = _materialize_train_modalities(dataset, stage)
                train_batch = _adapt_train_batch(train_modalities, protocol)
                stage_context = StageContext(
                    stage_index=stage_index,
                    total_stages=total_stages,
                    active_classes=stage.active_classes,
                    current_classes=stage.introduced_classes,
                    seen_classes=stage.active_classes,
                )
                stage_result = method.fit_stage(train_batch, stage_context)
                old_caches = dict(caches)
                log = dict(stage_result.log)
                log.setdefault("stage", stage_index)
                log.setdefault("train_time", float(stage_result.elapsed_time_s))
                log["introduced_classes"] = stage.introduced_classes
                log["active_classes"] = stage.active_classes
                if stage_result.metadata:
                    log["method_metadata"] = dict(stage_result.metadata)

                database_encode_time = 0.0
                if config.evaluation.retrieval_database_mode == "append":
                    encoded_current, database_encode_time = self._encode_directional(
                        method, train_modalities, stage_context
                    )
                    caches = {
                        name: _append_modality_cache(
                            caches[name],
                            encoded_current[name],
                            train_modalities[name],
                            dataset,
                        )
                        for name in ("image", "text")
                    }
                elif config.evaluation.retrieval_database_mode == "recompute":
                    cumulative = _materialize_protocol_train_range(
                        dataset, protocol, stage_index + 1
                    )
                    encoded_database, database_encode_time = self._encode_directional(
                        method, cumulative, stage_context
                    )
                    caches = {
                        name: _modality_cache(
                            encoded_database[name], cumulative[name], dataset
                        )
                        for name in ("image", "text")
                    }
                else:
                    raise ValueError("Unknown retrieval database mode")
                log["cumulative_image_database_size"] = _cache_size(caches["image"])
                log["cumulative_text_database_size"] = _cache_size(caches["text"])
                train_log.append(log)
                if config.output.save_train_log:
                    save_json(train_log_path, train_log, save_ap=True)

                should_evaluate = (
                    config.evaluation.retrieval_eval_mode == "per_stage"
                    or stage_index == total_stages - 1
                )
                if not should_evaluate:
                    continue
                query_modalities = _materialize_protocol_queries(
                    dataset,
                    protocol,
                    stage_index,
                    config.evaluation.query_mode,
                )
                query_encoded, query_encode_time = self._encode_directional(
                    method, query_modalities, stage_context
                )
                image_cache, text_cache = caches["image"], caches["text"]
                if (
                    image_cache.codes is None
                    or image_cache.labels is None
                    or text_cache.codes is None
                    or text_cache.labels is None
                ):
                    raise RuntimeError("Retrieval database is empty")
                sizes = {
                    "image_database_size": _cache_size(image_cache),
                    "text_database_size": _cache_size(text_cache),
                }
                common_efficiency = {
                    **dict(method.runtime_metadata()),
                    "train_time_s": float(stage_result.elapsed_time_s),
                    "query_encode_time_s": float(query_encode_time),
                    "database_encode_time_s": float(database_encode_time),
                    **sizes,
                }
                i2t = self._direction_metrics(
                    query=query_modalities["image"],
                    query_encoded=query_encoded["image"],
                    cache=text_cache,
                    train_elapsed_time=stage_result.elapsed_time_s,
                    hashcode_generate_time=query_encode_time + database_encode_time,
                    efficiency={**common_efficiency, "database_size": sizes["text_database_size"]},
                )
                t2i = self._direction_metrics(
                    query=query_modalities["text"],
                    query_encoded=query_encoded["text"],
                    cache=image_cache,
                    train_elapsed_time=stage_result.elapsed_time_s,
                    hashcode_generate_time=query_encode_time + database_encode_time,
                    efficiency={**common_efficiency, "database_size": sizes["image_database_size"]},
                )
                if (
                    config.evaluation.retrieval_database_mode == "append"
                    and _cache_size(old_caches["image"]) > 0
                    and _cache_size(old_caches["text"]) > 0
                ):
                    i2t["old_gallery"] = self._direction_metrics(
                        query=query_modalities["image"],
                        query_encoded=query_encoded["image"],
                        cache=old_caches["text"],
                        train_elapsed_time=stage_result.elapsed_time_s,
                        hashcode_generate_time=query_encode_time,
                        efficiency={
                            **common_efficiency,
                            "evaluation_scope": "frozen_old_gallery",
                            "database_size": _cache_size(old_caches["text"]),
                        },
                    )
                    t2i["old_gallery"] = self._direction_metrics(
                        query=query_modalities["text"],
                        query_encoded=query_encoded["text"],
                        cache=old_caches["image"],
                        train_elapsed_time=stage_result.elapsed_time_s,
                        hashcode_generate_time=query_encode_time,
                        efficiency={
                            **common_efficiency,
                            "evaluation_scope": "frozen_old_gallery",
                            "database_size": _cache_size(old_caches["image"]),
                        },
                    )
                directional_times = {
                    "search_time_i2t_s": float(i2t["efficiency"]["search_time_s"]),
                    "search_time_t2i_s": float(t2i["efficiency"]["search_time_s"]),
                    "metric_time_i2t_s": float(i2t["efficiency"]["metric_time_s"]),
                    "metric_time_t2i_s": float(t2i["efficiency"]["metric_time_s"]),
                }
                i2t["efficiency"].update(directional_times)
                t2i["efficiency"].update(directional_times)
                key = str(stage_index)
                metrics_i2t[key] = i2t
                metrics_t2i[key] = t2i
                save_json(metrics_i2t_path, metrics_i2t, save_ap=config.output.save_ap)
                save_json(metrics_t2i_path, metrics_t2i, save_ap=config.output.save_ap)
                if self._should_export_visualization(stage_index, total_stages):
                    self._save_protocol_visualization(
                        method=method,
                        stage_index=stage_index,
                        queries=query_modalities,
                        query_encoded=query_encoded,
                        caches=caches,
                    )

            checkpoint_path = run_dir / "checkpoint.pt"
            if config.output.save_model:
                if not method.capabilities.checkpoint:
                    raise ValueError(
                        f"Method '{method.method_id}' does not support checkpoints"
                    )
                method.save_checkpoint(str(checkpoint_path))
                manifest["files"]["checkpoint"] = str(checkpoint_path)
            manifest["status"] = "completed"
            manifest["completed_stages"] = total_stages
            save_json(manifest_path, manifest, save_ap=True)
        except Exception:
            manifest["status"] = "failed"
            manifest["completed_stages"] = len(train_log)
            save_json(manifest_path, manifest, save_ap=True)
            raise
        return RunResult(
            run_dir=run_dir,
            manifest_path=manifest_path,
            metrics_i2t=metrics_i2t,
            metrics_t2i=metrics_t2i,
            train_log=train_log,
        )

    def _load_protocol_dataset(self) -> tuple[CanonicalDataset, StreamProtocol]:
        experiment = self.config.experiment
        protocol_config = experiment.protocol
        assert protocol_config is not None
        if protocol_config.source == "legacy_mat":
            data_root = self._resolve(experiment.data_root)
            files = resolve_dataset_files(
                experiment.dataset,
                data_root,
                experiment.feature_mat,
                protocol_config.train_stream_mat or experiment.train_stream_mat,
                protocol_config.test_stream_mat or experiment.test_stream_mat,
            )
            data = load_cross_modal_mat(files.feature_mat)
            train_steps = load_stream_steps(
                files.train_stream_mat, experiment=protocol_config.experiment
            )
            test_steps = load_stream_steps(
                files.test_stream_mat, experiment=protocol_config.experiment
            )
            data, train_steps, test_steps = pad_label_width(
                data, train_steps, test_steps
            )
            train_steps, test_steps, alignment = align_stream_labels_to_feature_order(
                data, train_steps, test_steps
            )
            ids = (
                data.ids
                if data.ids is not None
                else np.arange(data.images.shape[0], dtype=np.int64)
            )
            dataset = CanonicalDataset(
                ids=ids,
                modalities={"image": data.images, "text": data.texts},
                ground_truth=data.labels,
                metadata={
                    "dataset_name": experiment.dataset,
                    "source_path": str(files.feature_mat),
                    "loader": "legacy_mat",
                },
            )
            protocol = legacy_stream_protocol(
                dataset,
                train_steps,
                test_steps,
                alignment,
                split_seed=protocol_config.split_seed,
                metadata={
                    "train_stream_mat": str(files.train_stream_mat),
                    "test_stream_mat": str(files.test_stream_mat),
                    "train_stream_sha256": (
                        sha256_file(files.train_stream_mat)
                        if files.train_stream_mat.is_file()
                        else None
                    ),
                    "test_stream_sha256": (
                        sha256_file(files.test_stream_mat)
                        if files.test_stream_mat.is_file()
                        else None
                    ),
                },
            )
            return dataset, protocol

        spec = self._load_dataset_spec()
        dataset = dataset_adapters.load(spec, self.workspace_root)
        protocol = resolve_protocol(
            dataset,
            protocol_config,
            workspace_root=self.workspace_root,
        )
        return dataset, protocol

    def _load_dataset_spec(self) -> DatasetSpec:
        experiment = self.config.experiment
        if experiment.dataset_spec:
            return load_dataset_spec(self._resolve(experiment.dataset_spec))
        if experiment.feature_mat:
            feature_path = self._resolve(experiment.feature_mat)
        else:
            dataset_dir = self._resolve(experiment.data_root) / experiment.dataset
            candidates = sorted(dataset_dir.glob("*.mat")) + sorted(
                dataset_dir.glob("*.npz")
            )
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one raw MAT/NPZ in {dataset_dir}; "
                    f"found {len(candidates)}"
                )
            feature_path = candidates[0]
        return default_image_text_spec(
            name=experiment.dataset,
            path=str(feature_path),
        )

    def _validate_method_protocol(
        self, method: Any, protocol: StreamProtocol
    ) -> None:
        capabilities = method.capabilities
        missing = sorted(set(capabilities.required_modalities) - set(protocol.modalities))
        if missing:
            raise ValueError(
                f"Method '{method.method_id}' requires missing modalities: "
                f"{', '.join(missing)}"
            )
        if (
            not protocol.paired_training
            and not capabilities.supports_independent_modalities
        ):
            raise ValueError(
                f"Method '{method.method_id}' does not support independent modalities"
            )
        if protocol.training_supervision not in capabilities.supported_supervision:
            raise ValueError(
                f"Method '{method.method_id}' does not support "
                f"{protocol.training_supervision} supervision"
            )

    def _encode_directional(
        self,
        method: Any,
        batches: Mapping[str, TrainModalityBatch | EvaluationBatch],
        context: StageContext,
    ) -> tuple[dict[str, EncodedModality], float]:
        start = time.perf_counter()
        if callable(getattr(method, "encode_image", None)) and callable(
            getattr(method, "encode_text", None)
        ):
            encoded = {
                "image": method.encode_image(batches["image"].features, context),
                "text": method.encode_text(batches["text"].features, context),
            }
            for name in ("image", "text"):
                validate_encoded_modality(
                    encoded[name],
                    sample_count=batches[name].features.shape[0],
                    bit=self.config.experiment.bit,
                )
        else:
            image_batch, text_batch = batches["image"], batches["text"]
            if not np.array_equal(image_batch.ids, text_batch.ids):
                raise ValueError(
                    f"Method '{method.method_id}' requires paired IDs for encoding"
                )
            labels = _batch_labels(image_batch, self._protocol_label_width())
            paired = CrossModalBatch(
                images=image_batch.features,
                texts=text_batch.features,
                labels=labels,
                ids=image_batch.ids,
            )
            paired_encoded, _ = self._encode(method, paired, context)
            encoded = {
                "image": EncodedModality(
                    paired_encoded.image_codes, paired_encoded.image_embeddings
                ),
                "text": EncodedModality(
                    paired_encoded.text_codes, paired_encoded.text_embeddings
                ),
            }
        return encoded, time.perf_counter() - start

    def _protocol_label_width(self) -> int:
        width = getattr(self, "_active_protocol_label_width", None)
        if width is None:
            raise RuntimeError("No active protocol label width")
        return int(width)

    def _direction_metrics(
        self,
        *,
        query: EvaluationBatch,
        query_encoded: EncodedModality,
        cache: _ModalityCache,
        train_elapsed_time: float,
        hashcode_generate_time: float,
        efficiency: dict[str, Any],
    ) -> dict[str, Any]:
        if cache.codes is None or cache.labels is None:
            raise RuntimeError("Retrieval cache is empty")
        query_labels = query.ground_truth
        database_size = cache.labels.shape[0]
        return compute_retrieval_metrics(
            query_labels=query_labels,
            retrieval_labels=cache.labels,
            query_codes=query_encoded.codes,
            retrieval_codes=cache.codes,
            train_elapsed_time=train_elapsed_time,
            hashcode_generate_time=hashcode_generate_time,
            top_k=bounded_ks(list(self.config.evaluation.top_k), database_size),
            ratio_n=list(self.config.evaluation.ratio_n),
            ndcg_ks=[
                min(value, database_size)
                for value in self.config.evaluation.ndcg_ks
            ],
            precision_curve_ks=bounded_ks(
                list(self.config.evaluation.precision_curve_ks), database_size
            ),
            efficiency=efficiency,
        )

    def _save_protocol_visualization(
        self,
        *,
        method: Any,
        stage_index: int,
        queries: Mapping[str, EvaluationBatch],
        query_encoded: Mapping[str, EncodedModality],
        caches: Mapping[str, _ModalityCache],
    ) -> None:
        image_query, text_query = queries["image"], queries["text"]
        image_cache, text_cache = caches["image"], caches["text"]
        if not np.array_equal(image_query.ids, text_query.ids):
            raise ValueError("Visualization requires paired query IDs")
        if not np.array_equal(image_query.ground_truth, text_query.ground_truth):
            raise ValueError("Visualization requires shared query labels")
        required = {
            "image gallery IDs": image_cache.ids,
            "text gallery IDs": text_cache.ids,
            "image gallery labels": image_cache.labels,
            "text gallery labels": text_cache.labels,
            "image gallery codes": image_cache.codes,
            "text gallery codes": text_cache.codes,
        }
        if any(value is None for value in required.values()):
            missing = [name for name, value in required.items() if value is None]
            raise RuntimeError(f"Cannot export directional visualization; missing {missing}")
        artifact = DirectionalVisualizationArtifact(
            method=method.display_name,
            dataset=self.config.experiment.dataset.casefold(),
            bits=self.config.experiment.bit,
            seed=self.config.experiment.seed,
            stage=stage_index + 1,
            gallery_image_ids=image_cache.ids,
            gallery_text_ids=text_cache.ids,
            query_ids=image_query.ids,
            gallery_image_labels=image_cache.labels,
            gallery_text_labels=text_cache.labels,
            query_labels=image_query.ground_truth,
            gallery_image_codes=image_cache.codes,
            gallery_text_codes=text_cache.codes,
            query_image_codes=query_encoded["image"].codes,
            query_text_codes=query_encoded["text"].codes,
            gallery_image_embeddings=image_cache.embeddings,
            gallery_text_embeddings=text_cache.embeddings,
            query_image_embeddings=query_encoded["image"].embeddings,
            query_text_embeddings=query_encoded["text"].embeddings,
        )
        target = (
            self._resolve(self.config.output.visualization_root)
            / self.config.method_id
            / self.config.experiment.dataset
            / f"{self.config.experiment.bit}bit"
            / f"stage{stage_index + 1}"
            / f"{method.variant}.npz"
        )
        save_directional_visualization_npz(target, artifact)

    def _encode(
        self,
        method: Any,
        batch: CrossModalBatch,
        context: StageContext,
    ) -> tuple[EncodedModalities, float]:
        start = time.perf_counter()
        encoded = method.encode(batch, context)
        elapsed = time.perf_counter() - start
        validate_encoded_modalities(
            encoded,
            sample_count=batch.labels.shape[0],
            bit=self.config.experiment.bit,
        )
        return encoded, elapsed

    def _load_class_names(self, num_classes: int) -> tuple[str, ...]:
        configured = self.config.experiment.class_names_file
        path = (
            self._resolve(configured)
            if configured
            else self._resolve(self.config.experiment.data_root)
            / self.config.experiment.dataset
            / "class_names.txt"
        )
        if not path.is_file():
            return tuple(f"class_{index:03d}" for index in range(num_classes))
        names = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
        if len(names) != num_classes:
            raise ValueError(
                f"Expected {num_classes} class names in {path}, found {len(names)}"
            )
        return names

    def _run_dir(self, variant: str, protocol_id: str | None = None) -> Path:
        experiment = self.config.experiment
        safe_variant = variant.replace("/", "_").replace("\\", "_").replace(" ", "_")
        base = (
            self._resolve(self.config.output.root)
            / self.config.method_id
            / experiment.dataset
            / f"{experiment.bit}bit"
            / f"experiment_{experiment.experiment}"
            / f"seed_{experiment.seed}"
        )
        if protocol_id is not None and not getattr(
            self, "_implicit_legacy_protocol", False
        ):
            safe_protocol = (
                protocol_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
            )
            base = base / f"protocol_{safe_protocol}"
            protocol = experiment.protocol
            if protocol is not None and protocol.query.type == "fixed":
                base = base / f"query_{protocol.query.split}"
        return base / safe_variant

    def _should_export_visualization(self, stage_index: int, total_stages: int) -> bool:
        if not self.config.output.save_visualization:
            return False
        target = self.config.output.visualization_stage or total_stages
        return stage_index + 1 == target

    def _save_visualization(
        self,
        *,
        method: Any,
        stage_index: int,
        query_batch: CrossModalBatch,
        query_encoded: EncodedModalities,
        cache: _EncodingCache,
    ) -> None:
        if cache.ids is None or cache.labels is None:
            raise RuntimeError("Cannot export an empty visualization artifact")
        artifact = VisualizationArtifact(
            method=method.display_name,
            dataset=self.config.experiment.dataset.casefold(),
            bits=self.config.experiment.bit,
            seed=self.config.experiment.seed,
            stage=stage_index + 1,
            database_indices=cache.ids,
            query_indices=query_batch.ids,
            database_labels=cache.labels,
            query_labels=query_batch.labels,
            database_image_codes=cache.image_codes,
            database_text_codes=cache.text_codes,
            query_image_codes=query_encoded.image_codes,
            query_text_codes=query_encoded.text_codes,
            database_image_embeddings=cache.image_embeddings,
            database_text_embeddings=cache.text_embeddings,
            query_image_embeddings=query_encoded.image_embeddings,
            query_text_embeddings=query_encoded.text_embeddings,
        )
        target = (
            self._resolve(self.config.output.visualization_root)
            / self.config.method_id
            / self.config.experiment.dataset
            / f"{self.config.experiment.bit}bit"
            / f"stage{stage_index + 1}"
            / f"{method.variant}.npz"
        )
        save_visualization_npz(target, artifact)

    def _manifest(
        self,
        method: Any,
        *,
        files: dict[str, Path],
        total_stages: int,
        alignment: dict[str, Any],
        status: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": status,
            "method": {
                "id": method.method_id,
                "display_name": method.display_name,
                "variant": method.variant,
                "capabilities": asdict(method.capabilities),
            },
            "experiment": asdict(self.config.experiment),
            "total_stages": total_stages,
            "completed_stages": 0,
            "stream_label_alignment": alignment,
            "files": {name: str(path) for name, path in files.items()},
        }

    def _resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.workspace_root / path).resolve()


def _batch_from_steps(data: CrossModalData, steps: list[StreamStep]) -> CrossModalBatch:
    images, texts, labels, ids = materialize_steps(data, steps)
    return CrossModalBatch(images=images, texts=texts, labels=labels, ids=ids)


def _normalize_data(data: CrossModalData) -> CrossModalData:
    def normalize(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32, copy=False)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-12)

    return CrossModalData(
        images=normalize(data.images),
        texts=normalize(data.texts),
        labels=data.labels,
        ids=data.ids,
    )


def _cache_from(
    encoded: EncodedModalities, batch: CrossModalBatch
) -> _EncodingCache:
    return _EncodingCache(
        image_codes=encoded.image_codes,
        text_codes=encoded.text_codes,
        labels=batch.labels,
        ids=batch.ids,
        image_embeddings=encoded.image_embeddings,
        text_embeddings=encoded.text_embeddings,
    )


def _append_cache(
    cache: _EncodingCache,
    encoded: EncodedModalities,
    batch: CrossModalBatch,
) -> _EncodingCache:
    if cache.image_codes is None:
        return _cache_from(encoded, batch)

    def append(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
        if left is None and right is None:
            return None
        if left is None or right is None:
            return None
        return np.concatenate([left, right], axis=0)

    return _EncodingCache(
        image_codes=append(cache.image_codes, encoded.image_codes),
        text_codes=append(cache.text_codes, encoded.text_codes),
        labels=append(cache.labels, batch.labels),
        ids=append(cache.ids, batch.ids),
        image_embeddings=append(cache.image_embeddings, encoded.image_embeddings),
        text_embeddings=append(cache.text_embeddings, encoded.text_embeddings),
    )


def _normalize_canonical_dataset(dataset: CanonicalDataset) -> CanonicalDataset:
    modalities: dict[str, np.ndarray] = {}
    for name, values in dataset.modalities.items():
        array = np.asarray(values, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        modalities[name] = array / np.maximum(norms, 1e-12)
    return CanonicalDataset(
        ids=dataset.ids,
        modalities=modalities,
        ground_truth=dataset.ground_truth,
        class_names=dataset.class_names,
        metadata=dataset.metadata,
    )


def _materialize_train_modalities(
    dataset: CanonicalDataset, stage: Any
) -> dict[str, TrainModalityBatch]:
    if dataset.ground_truth is None:
        raise ValueError("Training protocol materialization requires ground truth")
    output: dict[str, TrainModalityBatch] = {}
    for name, ids in stage.train_ids.items():
        rows = indices_to_rows(dataset.ids, ids, dataset.sample_count)
        mask = stage.supervision_masks[name]
        supervision = None
        if mask is not None:
            truth = dataset.ground_truth[rows]
            supervision = LabelObservation(
                values=np.where(mask, truth, 0.0).astype(np.float32),
                known_mask=np.asarray(mask, dtype=np.bool_),
            )
        output[name] = TrainModalityBatch(
            modality=name,
            features=np.asarray(dataset.modalities[name][rows], dtype=np.float32),
            ids=np.asarray(ids, dtype=np.int64),
            supervision=supervision,
        )
    return output


def _adapt_train_batch(
    modalities: Mapping[str, TrainModalityBatch],
    protocol: StreamProtocol,
) -> CrossModalBatch | MultiModalTrainBatch:
    image = modalities["image"]
    text = modalities["text"]
    paired = np.array_equal(image.ids, text.ids)
    fully_supervised = (
        image.supervision is not None
        and text.supervision is not None
        and bool(np.all(image.supervision.known_mask))
        and bool(np.all(text.supervision.known_mask))
    )
    if paired and fully_supervised:
        if not np.array_equal(
            image.supervision.values, text.supervision.values
        ):
            raise ValueError("Paired modalities expose inconsistent labels")
        return CrossModalBatch(
            images=image.features,
            texts=text.features,
            labels=image.supervision.values,
            ids=image.ids,
        )
    return MultiModalTrainBatch(modalities=dict(modalities))


def _materialize_protocol_train_range(
    dataset: CanonicalDataset,
    protocol: StreamProtocol,
    stop: int,
) -> dict[str, TrainModalityBatch]:
    output: dict[str, TrainModalityBatch] = {}
    for name in protocol.modalities:
        parts = [
            _materialize_train_modalities(dataset, protocol.stages[index])[name]
            for index in range(stop)
        ]
        supervision = None
        if all(part.supervision is not None for part in parts):
            supervision = LabelObservation(
                values=np.concatenate(
                    [part.supervision.values for part in parts if part.supervision is not None]
                ),
                known_mask=np.concatenate(
                    [
                        part.supervision.known_mask
                        for part in parts
                        if part.supervision is not None
                    ]
                ),
            )
        output[name] = TrainModalityBatch(
            modality=name,
            features=np.concatenate([part.features for part in parts]),
            ids=np.concatenate([part.ids for part in parts]),
            supervision=supervision,
        )
    return output


def _materialize_protocol_queries(
    dataset: CanonicalDataset,
    protocol: StreamProtocol,
    stage_index: int,
    query_mode: str,
) -> dict[str, EvaluationBatch]:
    if dataset.ground_truth is None:
        raise ValueError("Evaluation requires ground truth")
    output: dict[str, EvaluationBatch] = {}
    for name in protocol.modalities:
        if query_mode in {"current", "fixed"}:
            ids = protocol.stages[stage_index].query_ids[name]
        elif query_mode == "cumulative":
            ids = np.concatenate(
                [
                    protocol.stages[index].query_ids[name]
                    for index in range(stage_index + 1)
                ]
            )
            if np.unique(ids).size != ids.size:
                raise ValueError("Cumulative query stages contain duplicate IDs")
        else:
            raise ValueError(f"Unknown query_mode '{query_mode}'")
        rows = indices_to_rows(dataset.ids, ids, dataset.sample_count)
        output[name] = EvaluationBatch(
            modality=name,
            features=np.asarray(dataset.modalities[name][rows], dtype=np.float32),
            ids=np.asarray(ids, dtype=np.int64),
            ground_truth=np.asarray(dataset.ground_truth[rows], dtype=np.float32),
        )
    return output


def _batch_labels(
    batch: TrainModalityBatch | EvaluationBatch, fallback_width: int
) -> np.ndarray:
    if isinstance(batch, EvaluationBatch):
        return batch.ground_truth
    if batch.supervision is not None:
        return batch.supervision.values
    return np.zeros((len(batch.ids), fallback_width), dtype=np.float32)


def _ground_truth_for_ids(
    dataset: CanonicalDataset, ids: np.ndarray
) -> np.ndarray:
    if dataset.ground_truth is None:
        raise ValueError("Retrieval evaluation requires ground truth")
    rows = indices_to_rows(dataset.ids, ids, dataset.sample_count)
    return np.asarray(dataset.ground_truth[rows], dtype=np.float32)


def _modality_cache(
    encoded: EncodedModality,
    batch: TrainModalityBatch,
    dataset: CanonicalDataset,
) -> _ModalityCache:
    return _ModalityCache(
        codes=encoded.codes,
        labels=_ground_truth_for_ids(dataset, batch.ids),
        ids=batch.ids,
        embeddings=encoded.embeddings,
    )


def _append_modality_cache(
    cache: _ModalityCache,
    encoded: EncodedModality,
    batch: TrainModalityBatch,
    dataset: CanonicalDataset,
) -> _ModalityCache:
    current = _modality_cache(encoded, batch, dataset)
    if cache.codes is None:
        return current

    def append(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
        if left is None and right is None:
            return None
        if left is None or right is None:
            return None
        return np.concatenate([left, right], axis=0)

    return _ModalityCache(
        codes=append(cache.codes, current.codes),
        labels=append(cache.labels, current.labels),
        ids=append(cache.ids, current.ids),
        embeddings=append(cache.embeddings, current.embeddings),
    )


def _cache_size(cache: _ModalityCache) -> int:
    return 0 if cache.ids is None else int(cache.ids.size)
