from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import (
    dataset_adapters,
    dataset_spec_from_mapping,
    default_image_text_spec,
    load_dataset_spec,
)
from .protocol import protocol_config_from_mapping, resolve_protocol


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify a deterministic stream protocol from raw data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Standalone protocol JSON or a framework run JSON containing experiment.protocol.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Base directory for relative paths (default: current directory).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace_root.expanduser().resolve()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Protocol configuration root must be an object")
    dataset_spec, protocol_payload = _resolve_payload(payload, workspace)
    protocol_config = protocol_config_from_mapping(protocol_payload)
    if protocol_config.source == "legacy_mat":
        raise ValueError(
            "The prepare command is for raw/manifest protocols; legacy MAT is adapted at run time"
        )
    dataset = dataset_adapters.load(dataset_spec, workspace)
    protocol = resolve_protocol(
        dataset, protocol_config, workspace_root=workspace
    )
    output = (
        workspace / protocol_config.output_dir
        if protocol_config.output_dir
        else Path(protocol_config.manifest or "").parent
    )
    print(
        f"Protocol ready: {protocol.protocol_id}\n"
        f"  stages: {protocol.num_stages}\n"
        f"  supervision: {protocol.training_supervision}\n"
        f"  directory: {output.resolve()}"
    )
    return 0


def _resolve_payload(
    payload: Mapping[str, Any], workspace: Path
) -> tuple[Any, Mapping[str, Any]]:
    if "experiment" in payload:
        experiment = payload["experiment"]
        if not isinstance(experiment, Mapping):
            raise ValueError("$.experiment must be an object")
        protocol = experiment.get("protocol")
        if not isinstance(protocol, Mapping):
            raise ValueError("$.experiment.protocol must be an object")
        dataset_spec_path = experiment.get("dataset_spec")
        if dataset_spec_path:
            spec = load_dataset_spec(_resolve(dataset_spec_path, workspace))
        else:
            feature_mat = experiment.get("feature_mat")
            if not feature_mat:
                raise ValueError(
                    "Run config must provide experiment.dataset_spec or experiment.feature_mat"
                )
            spec = default_image_text_spec(
                name=str(experiment.get("dataset", "dataset")),
                path=str(feature_mat),
            )
        return spec, protocol
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("$.protocol must be an object")
    dataset = payload.get("dataset")
    if isinstance(dataset, Mapping):
        spec = dataset_spec_from_mapping(dataset)
    elif isinstance(payload.get("dataset_spec"), str):
        spec = load_dataset_spec(_resolve(payload["dataset_spec"], workspace))
    else:
        raise ValueError(
            "Standalone config must provide dataset object or dataset_spec path"
        )
    return spec, protocol


def _resolve(value: str | Path, workspace: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())

