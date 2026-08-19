from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", ("mscoco", "nuswide81", "iapr"))
@pytest.mark.parametrize("split", ("validation", "test"))
def test_strict_protocol_integrity_without_private_paths(dataset: str, split: str) -> None:
    directory = ROOT / "protocols" / dataset / "strict" / split / "seed-6513"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    indices_path = directory / "indices.npz"
    assert hashlib.sha256(indices_path.read_bytes()).hexdigest() == manifest["indices"]["sha256"]
    assert ":\\" not in manifest["dataset"]["source_path"]
    with np.load(indices_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    validation = set(arrays["validation_ids"].tolist())
    test = set(arrays["test_ids"].tolist())
    assert validation.isdisjoint(test)
    image_stages = [arrays[f"stage_{index:03d}__train__image"] for index in range(10)]
    text_stages = [arrays[f"stage_{index:03d}__train__text"] for index in range(10)]
    image = np.concatenate(image_stages)
    text = np.concatenate(text_stages)
    assert len(np.unique(image)) == len(image)
    assert len(np.unique(text)) == len(text)
    assert np.intersect1d(image, text).size == 0
    queries = validation if split == "validation" else test
    assert queries.isdisjoint(image.tolist())
    assert queries.isdisjoint(text.tolist())


@pytest.mark.parametrize("dataset", ("mscoco", "nuswide81", "iapr"))
@pytest.mark.parametrize("seed", (2096, 6513, 9231))
def test_pairblind_protocol_is_seeded_and_has_no_pair_index(dataset: str, seed: int) -> None:
    directory = ROOT / "protocols" / dataset / "pairblind" / "test" / f"seed-{seed}"
    manifest_text = (directory / "manifest.json").read_text(encoding="utf-8")
    assert "pair_index" not in manifest_text
    with np.load(directory / "indices.npz", allow_pickle=False) as payload:
        for stage in range(10):
            image = payload[f"stage_{stage:03d}__train__image"]
            text = payload[f"stage_{stage:03d}__train__text"]
            assert set(image.tolist()) == set(text.tolist())
            assert not np.array_equal(image, text)

