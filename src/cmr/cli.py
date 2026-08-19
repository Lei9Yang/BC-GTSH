from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

import numpy as np

from cmr.data import (
    dataset_adapters,
    load_dataset_spec,
    protocol_config_from_mapping,
    resolve_protocol,
)
from cmr.framework import ExperimentRunner, load_run_config
from cmr.methods import load_builtin_methods
from cmr.paper.assets import generate_paper_assets

DATASETS = ("mscoco", "nuswide81", "iapr")
BITS = (16, 32, 64, 128)
SEEDS = (2096, 6513, 9231)
FILES = {
    "mscoco": ("mscoco/MSCOCO_CLIP_image_CLIP_text_sonclass.mat",),
    "nuswide81": (
        "nuswide81/NUSwide_CLIP_image_CLIP_text.mat",
        "nuswide-TC81/NUSwide_CLIP_image_CLIP_text.mat",
    ),
    "iapr": ("iapr/IAPR_CLIP_image_CLIP_text.mat",),
}
PROFILES: dict[str, dict[str, Any]] = {
    "mscoco": {
        "epochs": 30,
        "mini_batch_size": 256,
        "lr": 1e-3,
        "topology_intra_k": 5,
        "topology_cross_k": 5,
        "topology_teacher_temperature": 0.1,
    },
    "nuswide81": {
        "epochs": 30,
        "mini_batch_size": 256,
        "lr": 1e-3,
        "topology_intra_k": 5,
        "topology_cross_k": 5,
        "topology_teacher_temperature": 0.1,
    },
    "iapr": {
        "epochs": 90,
        "mini_batch_size": 128,
        "lr": 2e-3,
        "topology_intra_k": 20,
        "topology_cross_k": 20,
        "topology_teacher_temperature": 0.2,
    },
}
CONDITIONS = {
    "full": {},
    "no-topology": {"lambda_topology": 0.0},
    "bce-topology": {"topology_objective": "bce"},
    "no-cross": {"topology_cross_k": 0},
    "no-intra": {"topology_intra_k": 0},
}


def release_root() -> Path:
    current = Path.cwd().resolve()
    if (current / "protocols").is_dir() and (current / "evidence").is_dir():
        return current
    editable_root = Path(__file__).resolve().parents[2]
    return editable_root if (editable_root / "protocols").is_dir() else current


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bc-gtsh",
        description="BC-GTSH source-disjoint continual cross-modal hashing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("make-toy-data", help="Create a deterministic toy dataset.")

    data_parser = subparsers.add_parser("data", help="Dataset utilities.")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)
    audit = data_sub.add_parser("audit")
    audit.add_argument("--dataset", choices=DATASETS, required=True)
    audit.add_argument("--data-root", type=Path, default=Path("data"))

    protocol_parser = subparsers.add_parser("protocol", help="Protocol utilities.")
    protocol_sub = protocol_parser.add_subparsers(dest="protocol_command", required=True)
    verify = protocol_sub.add_parser("verify")
    verify.add_argument("--dataset", choices=DATASETS, required=True)
    verify.add_argument("--split", choices=("validation", "test"), default="test")
    verify.add_argument("--data-root", type=Path, default=Path("data"))

    smoke = subparsers.add_parser("smoke", help="Run two toy stages.")
    _runtime_arguments(smoke, matrix=False)

    run = subparsers.add_parser("run", help="Run the frozen Full matrix.")
    _runtime_arguments(run)

    ablate = subparsers.add_parser("ablate", help="Run a structural ablation.")
    _runtime_arguments(ablate)
    ablate.add_argument(
        "--condition", choices=tuple(name for name in CONDITIONS if name != "full"), required=True
    )

    control = subparsers.add_parser("control", help="Run a registered control.")
    _runtime_arguments(control)
    control.add_argument("--kind", choices=("static-s1", "pairblind"), required=True)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate completed runs.")
    aggregate.add_argument("--result-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path)

    assets = subparsers.add_parser("paper-assets", help="Render tables and diagnostic figures.")
    assets.add_argument("--evidence-root", type=Path, default=Path("evidence"))
    assets.add_argument("--output-root", type=Path, default=Path("paper-assets"))
    assets.add_argument("--media-root", type=Path)
    return parser.parse_args(argv)


def _runtime_arguments(parser: argparse.ArgumentParser, matrix: bool = True) -> None:
    if matrix:
        parser.add_argument("--dataset", choices=DATASETS, action="append")
        parser.add_argument("--bits", choices=BITS, type=int, nargs="+")
        parser.add_argument("--seeds", choices=SEEDS, type=int, nargs="+")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = release_root()
    if args.command == "make-toy-data":
        target = make_toy_data(root)
        print(f"Toy dataset: {target}")
        return 0
    if args.command == "data":
        return audit_data(args.dataset, args.data_root, root)
    if args.command == "protocol":
        return verify_protocol(args.dataset, args.split, args.data_root, root)
    if args.command == "smoke":
        return run_smoke(args, root)
    if args.command == "aggregate":
        output = args.output or args.result_root / "summary.csv"
        aggregate_results(args.result_root, output)
        print(f"Summary: {output.resolve()}")
        return 0
    if args.command == "paper-assets":
        generate_paper_assets(
            _resolve(args.evidence_root, root),
            _resolve(args.output_root, root),
            media_root=None if args.media_root is None else args.media_root.resolve(),
        )
        return 0
    if args.command in {"run", "ablate", "control"}:
        return run_matrix(args, root)
    raise AssertionError(args.command)


def make_toy_data(root: Path) -> Path:
    target = root / "data" / "toy" / "toy.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(6513)
    count, classes = 120, 8
    labels = np.zeros((count, classes), dtype=np.float32)
    for index in range(count):
        labels[index, index % classes] = 1
        if index % 3 == 0:
            labels[index, (index + 2) % classes] = 1
    semantic = labels @ rng.normal(size=(classes, 32)).astype(np.float32)
    images = semantic + 0.08 * rng.normal(size=semantic.shape)
    texts = semantic @ rng.normal(size=(32, 24)) + 0.08 * rng.normal(size=(count, 24))
    np.savez_compressed(
        target,
        Images=images.astype(np.float32),
        Texts=texts.astype(np.float32),
        Labels=labels,
        Idx=np.arange(count, dtype=np.int64),
    )
    return target


def audit_data(dataset: str, data_root: Path, root: Path) -> int:
    spec_path = _materialize_dataset_spec(dataset, data_root, root / ".release-cache")
    spec = load_dataset_spec(spec_path)
    canonical = dataset_adapters.load(spec, root)
    manifest = _read_json(root / "protocols" / dataset / "strict" / "test" / "seed-6513" / "manifest.json")
    actual_hash = _sha256(Path(spec.path))
    expected_hash = str(manifest["dataset"]["source_sha256"])
    if actual_hash != expected_hash:
        raise ValueError(f"Dataset SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")
    report = {
        "dataset": dataset,
        "samples": canonical.sample_count,
        "classes": canonical.label_width,
        "image_dim": int(canonical.modalities["image"].shape[1]),
        "text_dim": int(canonical.modalities["text"].shape[1]),
        "sha256": actual_hash,
    }
    print(json.dumps(report, indent=2))
    return 0


def verify_protocol(dataset: str, split: str, data_root: Path, root: Path) -> int:
    spec_path = _materialize_dataset_spec(dataset, data_root, root / ".release-cache")
    canonical = dataset_adapters.load(load_dataset_spec(spec_path), root)
    manifest_path = root / "protocols" / dataset / "strict" / split / "seed-6513" / "manifest.json"
    protocol = resolve_protocol(
        canonical,
        protocol_config_from_mapping(
            {
                "source": "manifest",
                "manifest": str(manifest_path),
                "holdout": {"type": "fixed"},
                "query": {"type": "fixed", "split": split},
            }
        ),
        workspace_root=root,
    )
    image_ids = np.concatenate([stage.train_ids["image"] for stage in protocol.stages])
    text_ids = np.concatenate([stage.train_ids["text"] for stage in protocol.stages])
    report = {
        "protocol_id": protocol.protocol_id,
        "stages": protocol.num_stages,
        "image_train": int(image_ids.size),
        "text_train": int(text_ids.size),
        "source_overlap": int(np.intersect1d(image_ids, text_ids).size),
        "validation_queries": int(protocol.validation_ids.size),
        "test_queries": int(protocol.test_ids.size),
    }
    print(json.dumps(report, indent=2))
    return 0


def run_smoke(args: argparse.Namespace, root: Path) -> int:
    toy = make_toy_data(root)
    cache = _resolve(args.output_root, root) / "plans"
    spec = {
        "name": "toy",
        "loader": "npz",
        "path": str(toy),
        "modalities": {"image": {"field": "Images", "sample_axis": 0}, "text": {"field": "Texts", "sample_axis": 0}},
        "labels": {"field": "Labels", "sample_axis": 0},
        "ids": {"field": "Idx", "sample_axis": 0},
    }
    spec_path = cache / "toy-spec.json"
    _write_json(spec_path, spec)
    protocol_dir = _resolve(args.output_root, root) / "toy-protocol"
    payload = _run_payload(
        dataset="toy",
        bit=16,
        seed=6513,
        device=args.device or "cpu",
        output_root=_resolve(args.output_root, root),
        dataset_spec=spec_path,
        protocol={
            "source": "raw",
            "split_seed": 6513,
            "output_dir": str(protocol_dir),
            "holdout": {"type": "fixed", "validation_size": 12, "test_size": 20, "stratified": True},
            "modality": {"type": "source_disjoint", "modalities": ["image", "text"], "ratios": [0.5, 0.5]},
            "arrival": {"type": "iid_stratified", "num_stages": 2, "initial_class_count": 0},
            "query": {"type": "fixed", "split": "test"},
            "supervision": {"type": "full", "labeled_fraction": 1.0},
        },
        parameters={
            **_base_parameters("mscoco"),
            "epochs": 1,
            "mini_batch_size": 16,
            "eval_batch_size": 64,
            "embed_dim": 32,
            "topology_intra_k": 2,
            "topology_cross_k": 2,
        },
        variant="BC-GTSH-Smoke",
        max_stages=2,
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0
    config_path = cache / "toy-smoke.json"
    _write_json(config_path, payload)
    load_builtin_methods()
    result = ExperimentRunner(load_run_config(config_path), workspace_root=root).run()
    print(f"Completed smoke run: {result.run_dir}")
    return 0


def run_matrix(args: argparse.Namespace, root: Path) -> int:
    datasets = tuple(args.dataset or DATASETS)
    bits = tuple(args.bits or BITS)
    seeds = tuple(args.seeds or SEEDS)
    condition = "full"
    control = None
    if args.command == "ablate":
        condition = args.condition
    elif args.command == "control":
        control = args.kind
    planned = [(dataset, bit, seed) for dataset in datasets for bit in bits for seed in seeds]
    if args.dry_run:
        print(json.dumps({"command": args.command, "condition": condition, "control": control, "runs": planned}, indent=2))
        return 0

    load_builtin_methods()
    failures = 0
    for dataset, bit, seed in planned:
        try:
            _run_one(args, root, dataset, bit, seed, condition, control)
        except Exception as error:  # each matrix item has an independent record boundary
            failures += 1
            print(f"FAILED {dataset}/{bit}/{seed}: {error}", file=sys.stderr)
            if args.fail_fast:
                raise
    return 1 if failures else 0


def _run_one(
    args: argparse.Namespace,
    root: Path,
    dataset: str,
    bit: int,
    seed: int,
    condition: str,
    control: str | None,
) -> None:
    output_root = _resolve(args.output_root, root)
    plan_dir = output_root / "plans"
    spec_path = _materialize_dataset_spec(dataset, args.data_root, plan_dir)
    parameters = _base_parameters(dataset)
    parameters.update(CONDITIONS[condition])
    track = "strict"
    protocol_path = root / "protocols" / dataset / "strict" / "test" / "seed-6513" / "manifest.json"
    variant = "BC-GTSH-Full" if condition == "full" else f"BC-GTSH-{condition}"
    if control == "static-s1":
        parameters["update_after_stage_zero"] = False
        variant = "BC-GTSH-Static-S1"
    elif control == "pairblind":
        parameters["allow_source_overlap"] = True
        parameters["unpaired_protocol"] = "deranged"
        protocol_path = root / "protocols" / dataset / "pairblind" / "test" / f"seed-{seed}" / "manifest.json"
        variant = "BC-GTSH-PairBlind"
        track = "paired-source-oracle"
    payload = _run_payload(
        dataset=dataset,
        bit=bit,
        seed=seed,
        device=args.device,
        output_root=output_root,
        dataset_spec=spec_path,
        protocol={
            "source": "manifest",
            "manifest": str(protocol_path),
            "holdout": {"type": "fixed"},
            "query": {"type": "fixed", "split": "test"},
        },
        parameters=parameters,
        variant=variant,
    )
    payload["release_track"] = track
    comparable = {key: value for key, value in payload.items() if key != "release_track"}
    config_path = plan_dir / f"{dataset}-{condition}-{control or 'main'}-{bit}-{seed}.json"
    _write_json(config_path, comparable)
    digest = _effective_config_sha256(comparable)
    complete = _find_status(output_root, digest, "completed")
    failed = _find_status(output_root, digest, "failed")
    if complete and args.resume:
        print(f"SKIP completed {dataset}/{bit}/{seed}: {complete}")
        return
    if complete and not args.resume:
        raise RuntimeError(f"Completed run already exists; use --resume to skip it: {complete}")
    if failed and not args.retry_failed:
        raise RuntimeError(f"Previous failed run exists; use --retry-failed: {failed}")
    run_output_root = output_root
    if failed and args.retry_failed:
        attempt = 1 + sum(
            1
            for path in output_root.rglob("manifest.json")
            if _manifest_matches(path, digest, "failed")
        )
        run_output_root = output_root / "retry_attempts" / f"{dataset}-{condition}-{control or 'main'}-{bit}-{seed}-a{attempt:03d}"
        comparable["output"]["root"] = str(run_output_root)
        comparable["output"]["visualization_root"] = str(run_output_root / "visualization")
        _write_json(config_path, comparable)
    try:
        result = ExperimentRunner(load_run_config(config_path), workspace_root=root).run()
    except Exception:
        for manifest_path in run_output_root.rglob("manifest.json"):
            if _manifest_matches(manifest_path, digest, "failed"):
                manifest = _read_json(manifest_path)
                manifest["release_track"] = track
                manifest["effective_config_sha256"] = digest
                _write_json(manifest_path, manifest)
        raise
    manifest = _read_json(result.manifest_path)
    manifest["release_track"] = track
    manifest["effective_config_sha256"] = digest
    _write_json(result.manifest_path, manifest)
    print(f"Completed {dataset}/{bit}/{seed}: {result.run_dir}")


def _base_parameters(dataset: str) -> dict[str, Any]:
    return {
        "epochs": PROFILES[dataset]["epochs"],
        "mini_batch_size": PROFILES[dataset]["mini_batch_size"],
        "eval_batch_size": 512,
        "lr": PROFILES[dataset]["lr"],
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "embed_dim": 256,
        "topology_intra_k": PROFILES[dataset]["topology_intra_k"],
        "topology_cross_k": PROFILES[dataset]["topology_cross_k"],
        "topology_objective": "row_kl",
        "topology_teacher_temperature": PROFILES[dataset]["topology_teacher_temperature"],
        "topology_negative_ratio": 3.0,
        "logit_scale": 5.0,
        "lambda_topology": 1.0,
        "lambda_quant": 0.01,
        "lambda_bit": 0.001,
        "dropout": 0.1,
        "unpaired_protocol": "disjoint",
        "allow_source_overlap": False,
        "update_after_stage_zero": True,
    }


def _run_payload(
    *,
    dataset: str,
    bit: int,
    seed: int,
    device: str | None,
    output_root: Path,
    dataset_spec: Path,
    protocol: dict[str, Any],
    parameters: dict[str, Any],
    variant: str,
    max_stages: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": {
            "dataset": dataset,
            "bit": bit,
            "seed": seed,
            "experiment": 0,
            "data_root": ".",
            "dataset_spec": str(dataset_spec),
            "protocol": protocol,
            "max_stages": max_stages,
            "normalize_input_features": True,
            "device": device,
        },
        "method": {"id": "bc-gtsh", "variant": variant, "parameters": parameters},
        "evaluation": {
            "query_mode": "fixed",
            "retrieval_database_mode": "append",
            "retrieval_eval_mode": "per_stage",
            "top_k": [10, 20, 50, 100],
            "ratio_n": [0.1, 0.5],
            "ndcg_ks": [1000],
            "precision_curve_ks": [10, 20, 50, 100, 200, 500, 1000],
        },
        "output": {
            "root": str(output_root),
            "save_train_log": True,
            "save_model": False,
            "save_visualization": False,
            "visualization_root": str(output_root / "visualization"),
            "visualization_stage": 10,
            "save_ap": False,
        },
    }


def aggregate_results(result_root: Path, output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for manifest_path in result_root.rglob("manifest.json"):
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "completed" or "method" not in manifest:
            continue
        directory = manifest_path.parent
        i2t_path, t2i_path = directory / "metrics_i2t.json", directory / "metrics_t2i.json"
        if not i2t_path.is_file() or not t2i_path.is_file():
            continue
        i2t, t2i = _read_json(i2t_path), _read_json(t2i_path)
        final_key = str(max(int(key) for key in i2t))
        rows.append(
            {
                "dataset": manifest["experiment"]["dataset"],
                "method": manifest["method"]["variant"],
                "bit": manifest["experiment"]["bit"],
                "seed": manifest["experiment"]["seed"],
                "track": manifest.get("release_track", "strict"),
                "i2t_final_map": i2t[final_key]["map_all_queries"],
                "t2i_final_map": t2i[final_key]["map_all_queries"],
                "i2t_stage_map": mean(float(item["map_all_queries"]) for item in i2t.values()),
                "t2i_stage_map": mean(float(item["map_all_queries"]) for item in t2i.values()),
            }
        )
    if not rows:
        raise ValueError(f"No completed runs under {result_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["method"], row["bit"], row["track"]), []).append(row)
    summary_path = output.with_name(output.stem + "_mean_std.csv")
    summary_rows = []
    for key, values in sorted(grouped.items()):
        item: dict[str, Any] = {"dataset": key[0], "method": key[1], "bit": key[2], "track": key[3], "n": len(values)}
        for metric in ("i2t_final_map", "t2i_final_map", "i2t_stage_map", "t2i_stage_map"):
            numbers = [float(row[metric]) for row in values]
            item[metric + "_mean"] = mean(numbers)
            item[metric + "_std"] = stdev(numbers) if len(numbers) > 1 else 0.0
        summary_rows.append(item)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def _materialize_dataset_spec(dataset: str, data_root: Path, output_root: Path) -> Path:
    data_root = data_root.expanduser().resolve()
    candidates = [data_root / relative for relative in FILES[dataset]]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"{dataset}-dataset.json"
    _write_json(
        target,
        {
            "name": dataset,
            "loader": "hdf5",
            "path": str(path),
            "modalities": {"image": {"field": "Images", "sample_axis": 1}, "text": {"field": "Texts", "sample_axis": 1}},
            "labels": {"field": "Labels", "sample_axis": 1},
            "ids": {"field": "Idx", "sample_axis": 1},
            "metadata": {"anonymous_class_names": True},
        },
    )
    return target


def _find_status(root: Path, digest: str, status: str) -> Path | None:
    for path in root.rglob("manifest.json") if root.exists() else ():
        if _manifest_matches(path, digest, status):
            return path
    return None


def _manifest_matches(path: Path, digest: str, status: str) -> bool:
    payload = _read_json(path)
    if payload.get("status") != status:
        return False
    if payload.get("effective_config_sha256") == digest:
        return True
    config_path = path.parent / "config.json"
    return config_path.is_file() and _effective_config_sha256(_read_json(config_path)) == digest


def _effective_config_sha256(payload: dict[str, Any]) -> str:
    canonical = copy.deepcopy(payload)
    output = canonical.get("output", {})
    output["root"] = "<RESULT_ROOT>"
    output["visualization_root"] = "<VISUALIZATION_ROOT>"
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def _resolve(path: Path, root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
