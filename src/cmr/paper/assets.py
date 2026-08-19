from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

DATASETS = ("mscoco", "nuswide81", "iapr")
NAMES = {"mscoco": "MSCOCO", "nuswide81": "NUS-WIDE-81", "iapr": "IAPR"}
BITS = (16, 32, 64, 128)


def generate_paper_assets(
    evidence_root: Path,
    output_root: Path,
    *,
    media_root: Path | None = None,
) -> None:
    evidence_root = evidence_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    table_root = output_root / "tables"
    figure_root = output_root / "figures"
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    table_specs = {
        "strict_online": ("UIH", "BC-GTSH"),
        "offline_reference": ("AMSH", "GSPH", "EDMH", "BC-GTSH"),
        "paired_oracle": (
            "DOCH",
            "LCDH",
            "OCMFH",
            "OH-ELS",
            "PIC-CMH",
            "SSOCH-10\\%",
            "BC-GTSH-PairBlind",
        ),
        "ablation": (
            "No-Topology",
            "BCE-Topology",
            "No-Cross",
            "No-Intra",
            "Static-S1",
            "Full",
        ),
    }
    for name, methods in table_specs.items():
        rows = _read_csv(evidence_root / "tables" / f"{name}.csv")
        _render_table(table_root / f"{name}.tex", rows, methods)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Install the 'plot' extra to render paper figures") from error

    _plot_continual(evidence_root, figure_root, plt, np)
    _plot_retrieval(evidence_root, figure_root, plt, np)
    _plot_sensitivity(evidence_root, figure_root, plt)

    qualitative = {
        "status": "skipped",
        "reason": "Original MSCOCO images and captions were not supplied.",
    }
    if media_root is not None:
        qualitative = {
            "status": "external-media-required",
            "media_root": str(media_root),
            "reason": "Use the frozen query-selection manifest with the original media files.",
        }
    (output_root / "qualitative_status.json").write_text(
        json.dumps(qualitative, indent=2), encoding="utf-8"
    )
    generated = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            generated[str(path.relative_to(output_root)).replace("\\", "/")] = _sha256(path)
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_manifest_sha256": _sha256(evidence_root / "manifest.json"),
                "generated": generated,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Paper assets: {output_root.resolve()}")


def _render_table(path: Path, rows: list[dict[str, str]], methods: tuple[str, ...]) -> None:
    values = {
        (row["method"], row["dataset"], row["task"], int(row["bit"])): float(row["mean_map"])
        for row in rows
    }
    lines = [
        r"\begin{tabular}{ll*{12}{c}}",
        r"\toprule",
        r"Task & Method & \multicolumn{4}{c}{MSCOCO} & \multicolumn{4}{c}{NUS-WIDE-81} & \multicolumn{4}{c}{IAPR} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
        r" & & 16 & 32 & 64 & 128 & 16 & 32 & 64 & 128 & 16 & 32 & 64 & 128 \\",
        r"\midrule",
    ]
    for task_index, task in enumerate(("I2T", "T2I")):
        for method_index, method in enumerate(methods):
            task_cell = rf"\multirow{{{len(methods)}}}{{*}}{{{task}}}" if method_index == 0 else ""
            cells = [f"{values[(method, dataset, task, bit)]:.3f}" for dataset in DATASETS for bit in BITS]
            lines.append(" & ".join((task_cell, method, *cells)) + r" \\")
        if task_index == 0:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_continual(evidence_root: Path, output: Path, plt: Any, np: Any) -> None:
    rows = _read_csv(evidence_root / "continual_curves_64bit.csv")
    values: dict[tuple[str, str, int, int, str], float] = {
        (row["dataset"], row["method"], int(row["seed"]), int(row["stage"]), row["metric"]): float(row["value"])
        for row in rows
    }
    metrics = (
        ("i2t_map", "Img2Txt stage mAP"),
        ("t2i_map", "Txt2Img stage mAP"),
        ("i2t_old", "Img2Txt old-gallery mAP"),
        ("t2i_old", "Txt2Img old-gallery mAP"),
    )
    colors = {"Full": "#0072B2", "UIH": "#009E73"}
    markers = {"Full": "o", "UIH": "^"}
    fig, axes = plt.subplots(4, 3, figsize=(8.2, 8.8), sharex=True)
    for column, dataset in enumerate(DATASETS):
        for row_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index, column]
            for method in ("UIH", "Full"):
                centers, spreads = [], []
                for stage in range(1, 11):
                    selected = [
                        values[(dataset, method, seed, stage, metric)]
                        for seed in (2096, 6513, 9231)
                        if math.isfinite(values[(dataset, method, seed, stage, metric)])
                    ]
                    centers.append(mean(selected) if selected else math.nan)
                    spreads.append(stdev(selected) if len(selected) > 1 else 0.0)
                x = np.arange(1, 11)
                y, spread = np.asarray(centers), np.asarray(spreads)
                label = "BC-GTSH" if method == "Full" else method
                axis.plot(x, y, color=colors[method], marker=markers[method], markersize=3, label=label)
                axis.fill_between(x, y - spread, y + spread, color=colors[method], alpha=0.13)
            if row_index == 0:
                axis.set_title(NAMES[dataset])
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == 3:
                axis.set_xlabel("Stage")
            axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "continual_curves.pdf", bbox_inches="tight")
    fig.savefig(output / "continual_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_retrieval(evidence_root: Path, output: Path, plt: Any, np: Any) -> None:
    methods = ("Full", "UIH", "AMSH", "GSPH", "EDMH")
    colors = {"Full": "#0072B2", "UIH": "#009E73", "AMSH": "#D55E00", "GSPH": "#CC79A7", "EDMH": "#E69F00"}
    styles = {"Full": "-", "UIH": "-", "AMSH": "--", "GSPH": "--", "EDMH": "--"}
    fig, axes = plt.subplots(4, 3, figsize=(8.2, 8.8))
    for column, dataset in enumerate(DATASETS):
        payload = json.loads((evidence_root / "curves" / f"{dataset}_64bit_seed6513.json").read_text(encoding="utf-8"))
        for method in methods:
            for direction_index, direction in enumerate(("i2t", "t2i")):
                metric = payload[method][direction]
                pr = metric["prCurve"]
                axes[direction_index, column].plot(
                    pr["recall_levels"], pr["precision"], color=colors[method], linestyle=styles[method], label="BC-GTSH" if method == "Full" else method
                )
                curve = metric["topKPrecisionCurve"]
                axes[direction_index + 2, column].plot(
                    curve["Ks"], curve["precision"], color=colors[method], linestyle=styles[method], label="BC-GTSH" if method == "Full" else method
                )
        axes[0, column].set_title(NAMES[dataset])
        for row in range(4):
            axes[row, column].grid(alpha=0.22)
        axes[2, column].set_xscale("log")
        axes[3, column].set_xscale("log")
        if column == 0:
            for row, label in enumerate(("Img2Txt precision", "Txt2Img precision", "Img2Txt Precision@K", "Txt2Img Precision@K")):
                axes[row, column].set_ylabel(label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "retrieval_curves.pdf", bbox_inches="tight")
    fig.savefig(output / "retrieval_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_sensitivity(evidence_root: Path, output: Path, plt: Any) -> None:
    rows = _read_csv(evidence_root / "sensitivity.csv")
    parameters = ("lambda_topology", "topology_k", "topology_teacher_temperature")
    colors = {"mscoco": "#0072B2", "nuswide81": "#009E73", "iapr": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8))
    for column, parameter in enumerate(parameters):
        values = sorted({float(row["value"]) for row in rows if row["parameter"] == parameter})
        for dataset in DATASETS:
            selected = {
                float(row["value"]): float(row["average_stage_mean_map_all_queries"])
                for row in rows
                if row["parameter"] == parameter and row["dataset"] == dataset
            }
            axes[column].plot(range(len(values)), [selected[value] for value in values], marker="o", color=colors[dataset], label=NAMES[dataset])
        axes[column].set_xticks(range(len(values)), [f"{value:g}" for value in values])
        axes[column].set_xlabel(parameter.replace("topology_", ""))
        axes[column].grid(alpha=0.22)
    axes[0].set_ylabel("Stage mAP")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output / "parameter_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(output / "parameter_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
