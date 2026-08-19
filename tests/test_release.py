from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from cmr.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_has_expected_full_matrix(capsys) -> None:
    assert main(["run", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["runs"]) == 36


def test_bundled_strict_table_uses_tuned_iapr() -> None:
    with (ROOT / "evidence" / "tables" / "strict_online.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    row = next(
        item
        for item in rows
        if item["method"] == "BC-GTSH"
        and item["dataset"] == "iapr"
        and item["task"] == "I2T"
        and int(item["bit"]) == 64
    )
    assert abs(float(row["mean_map"]) - 0.48634128314730596) < 1e-12


def test_release_tree_has_no_private_absolute_paths_or_large_files() -> None:
    separator = chr(92)
    markers = (
        "[A-Z]:" + re.escape(separator),
        re.escape("Users" + separator),
        re.escape("Pycharm" + " project"),
        re.escape("Program" + "Data"),
    )
    private = re.compile("(?:" + "|".join(markers) + ")")
    transient = {
        "__pycache__",
        ".pytest_cache",
        ".release-cache",
        "build",
        "dist",
        "data",
        "results",
        "paper-assets",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or transient.intersection(path.relative_to(ROOT).parts):
            continue
        assert path.stat().st_size < 10 * 1024 * 1024, path
        assert path.suffix not in {".pyc", ".pyo", ".mat", ".pt", ".pth", ".ckpt"}
        if path.suffix in {".py", ".json", ".md", ".toml", ".yml", ".cff", ".csv"}:
            assert private.search(path.read_text(encoding="utf-8-sig", errors="ignore")) is None, path


def test_evidence_manifest_declares_release_transform() -> None:
    manifest = json.loads((ROOT / "evidence" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["full"] == 36
    assert manifest["counts"]["structural_ablation"] == 144
    assert manifest["run_manifest"]["row_count"] == 342
    assert manifest["release_transform"]


def test_sanitized_run_manifest_is_complete_unique_and_hashed() -> None:
    path = ROOT / "evidence" / "runs.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {
        "full": 36,
        "uih_reference": 36,
        "structural_ablation": 144,
        "static_s1": 36,
        "pairblind": 36,
        "sensitivity": 45,
        "nonclip_robustness": 9,
    }
    counts = {family: sum(row["family"] == family for row in rows) for family in expected}
    assert counts == expected
    keys = {
        (row["family"], row["dataset"], row["condition"], row["bit"], row["seed"])
        for row in rows
    }
    assert len(keys) == len(rows) == 342
    for row in rows:
        claimed = row.pop("record_sha256")
        payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        assert hashlib.sha256(payload.encode()).hexdigest() == claimed

    manifest = json.loads((ROOT / "evidence" / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["run_manifest"]["sha256"]
