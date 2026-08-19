from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

from cmr.data import ProtocolConfig, protocol_config_from_mapping


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    bit: int
    seed: int = 1
    experiment: int = 0
    data_root: str = "data"
    feature_mat: str | None = None
    train_stream_mat: str | None = None
    test_stream_mat: str | None = None
    class_names_file: str | None = None
    dataset_spec: str | None = None
    protocol: ProtocolConfig | None = None
    max_stages: int | None = None
    normalize_input_features: bool = True
    device: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    query_mode: str = "current"
    retrieval_database_mode: str = "recompute"
    retrieval_eval_mode: str = "per_stage"
    top_k: tuple[int, ...] = (10, 20, 50, 100)
    ratio_n: tuple[float, ...] = (0.1, 0.5)
    ndcg_ks: tuple[int, ...] = (1000,)
    precision_curve_ks: tuple[int, ...] = (10, 20, 50, 100, 200, 500, 1000)


@dataclass(frozen=True)
class OutputConfig:
    root: str = "artifacts/results"
    save_train_log: bool = True
    save_model: bool = False
    save_visualization: bool = False
    visualization_root: str = "artifacts/visualization"
    visualization_stage: int | None = None
    save_ap: bool = False


@dataclass(frozen=True)
class RunConfig:
    schema_version: int
    experiment: ExperimentConfig
    method_id: str
    method_variant: str
    method_parameters: Mapping[str, Any]
    evaluation: EvaluationConfig
    output: OutputConfig
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment": asdict(self.experiment),
            "method": {
                "id": self.method_id,
                "variant": self.method_variant,
                "parameters": dict(self.method_parameters),
            },
            "evaluation": asdict(self.evaluation),
            "output": asdict(self.output),
        }


T = TypeVar("T")


def load_run_config(path: str | Path) -> RunConfig:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON config at line {error.lineno}, column {error.colno}") from error
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be an object")
    _reject_unknown(payload, {"schema_version", "experiment", "method", "evaluation", "output"}, "$")
    version = payload.get("schema_version")
    if version != 1:
        raise ValueError("$.schema_version must equal 1")

    experiment_payload = payload.get("experiment")
    if not isinstance(experiment_payload, dict):
        raise ValueError("$.experiment must be an object")
    protocol_payload = experiment_payload.get("protocol")
    protocol = None
    if protocol_payload is not None:
        if not isinstance(protocol_payload, dict):
            raise ValueError("$.experiment.protocol must be an object")
        protocol = protocol_config_from_mapping(protocol_payload)
    experiment_values = dict(experiment_payload)
    experiment_values["protocol"] = protocol
    experiment = _load_dataclass(ExperimentConfig, experiment_values, "$.experiment")
    evaluation = _load_dataclass(
        EvaluationConfig, payload.get("evaluation", {}), "$.evaluation"
    )
    output = _load_dataclass(OutputConfig, payload.get("output", {}), "$.output")
    method = payload.get("method")
    if not isinstance(method, dict):
        raise ValueError("$.method must be an object")
    _reject_unknown(method, {"id", "variant", "parameters"}, "$.method")
    method_id = method.get("id")
    variant = method.get("variant", "default")
    parameters = method.get("parameters", {})
    if not isinstance(method_id, str) or not method_id.strip():
        raise ValueError("$.method.id must be a non-empty string")
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("$.method.variant must be a non-empty string")
    if not isinstance(parameters, dict):
        raise ValueError("$.method.parameters must be an object")

    _validate_config(experiment, evaluation, output)
    return RunConfig(
        schema_version=1,
        experiment=experiment,
        method_id=method_id.strip().casefold(),
        method_variant=variant.strip(),
        method_parameters=parameters,
        evaluation=evaluation,
        output=output,
        source_path=source,
    )


def _load_dataclass(cls: type[T], value: Any, path: str) -> T:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    allowed = {item.name for item in fields(cls)}
    _reject_unknown(value, allowed, path)
    try:
        return cls(**value)
    except TypeError as error:
        raise ValueError(f"{path}: {error}") from error


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path}.{unknown[0]} is not a recognized field")


def _validate_config(
    experiment: ExperimentConfig,
    evaluation: EvaluationConfig,
    output: OutputConfig,
) -> None:
    if not isinstance(experiment.dataset, str) or not experiment.dataset.strip():
        raise ValueError("$.experiment.dataset must be non-empty")
    if not isinstance(experiment.bit, int) or experiment.bit <= 0:
        raise ValueError("$.experiment.bit must be a positive integer")
    if not isinstance(experiment.seed, int):
        raise ValueError("$.experiment.seed must be an integer")
    if not isinstance(experiment.experiment, int) or experiment.experiment < 0:
        raise ValueError("$.experiment.experiment must be a non-negative integer")
    if not isinstance(experiment.data_root, str):
        raise ValueError("$.experiment.data_root must be a string")
    if experiment.max_stages is not None and (
        not isinstance(experiment.max_stages, int) or experiment.max_stages <= 0
    ):
        raise ValueError("$.experiment.max_stages must be a positive integer or null")
    if not isinstance(experiment.normalize_input_features, bool):
        raise ValueError("$.experiment.normalize_input_features must be a boolean")
    if experiment.device is not None and not isinstance(experiment.device, str):
        raise ValueError("$.experiment.device must be a string or null")
    if experiment.dataset_spec is not None and not isinstance(experiment.dataset_spec, str):
        raise ValueError("$.experiment.dataset_spec must be a string or null")
    if evaluation.query_mode not in {"current", "cumulative", "fixed"}:
        raise ValueError(
            "$.evaluation.query_mode must be 'current', 'cumulative', or 'fixed'"
        )
    if evaluation.retrieval_database_mode not in {"recompute", "append"}:
        raise ValueError(
            "$.evaluation.retrieval_database_mode must be 'recompute' or 'append'"
        )
    if evaluation.retrieval_eval_mode not in {"per_stage", "final"}:
        raise ValueError("$.evaluation.retrieval_eval_mode must be 'per_stage' or 'final'")
    for name in ("top_k", "ndcg_ks", "precision_curve_ks"):
        values = tuple(getattr(evaluation, name))
        if not values or any(not isinstance(item, int) or item <= 0 for item in values):
            raise ValueError(f"$.evaluation.{name} must contain positive integers")
        object.__setattr__(evaluation, name, values)
    ratios = tuple(evaluation.ratio_n)
    if not ratios or any(not isinstance(item, (int, float)) or not 0 < item <= 1 for item in ratios):
        raise ValueError("$.evaluation.ratio_n must contain values in (0, 1]")
    object.__setattr__(evaluation, "ratio_n", ratios)
    if output.visualization_stage is not None and (
        not isinstance(output.visualization_stage, int)
        or output.visualization_stage <= 0
    ):
        raise ValueError("$.output.visualization_stage must be a positive integer or null")
    for name in ("save_train_log", "save_model", "save_visualization", "save_ap"):
        if not isinstance(getattr(output, name), bool):
            raise ValueError(f"$.output.{name} must be a boolean")
    if experiment.protocol is not None:
        protocol = experiment.protocol
        if protocol.query.type == "fixed" and evaluation.query_mode != "fixed":
            raise ValueError(
                "$.evaluation.query_mode must be 'fixed' for a fixed-query protocol"
            )
        if protocol.query.type == "per_stage_split" and evaluation.query_mode == "fixed":
            raise ValueError(
                "$.evaluation.query_mode cannot be 'fixed' for per_stage_split"
            )
        if (
            protocol.modality.type == "source_disjoint"
            and evaluation.retrieval_database_mode != "append"
        ):
            raise ValueError(
                "source_disjoint protocols require retrieval_database_mode='append'"
            )
