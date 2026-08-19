from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cmr.framework import (
    LabelObservation,
    MethodBuildContext,
    MultiModalTrainBatch,
    StageContext,
    TrainModalityBatch,
)
from cmr.methods.bc_gtsh.method import (
    BCGTSHMethod,
    BCGTSHMethodConfig,
    load_bc_gtsh_method_config,
)
from cmr.methods.bc_gtsh.model import BCGTSHEncoder


def context(bit: int = 16) -> MethodBuildContext:
    return MethodBuildContext(
        image_dim=12,
        text_dim=10,
        num_classes=4,
        bit=bit,
        seed=6513,
        class_names=("a", "b", "c", "d"),
        device="cpu",
    )


def stage() -> StageContext:
    classes = np.arange(4, dtype=np.int64)
    return StageContext(0, 2, classes, classes, classes)


def training_batch(overlap: bool = False) -> MultiModalTrainBatch:
    rng = np.random.default_rng(6513)
    count = 10
    labels = np.eye(4, dtype=np.float32)[np.arange(count) % 4]
    known = np.ones_like(labels, dtype=np.bool_)
    image_ids = np.arange(count, dtype=np.int64)
    text_ids = image_ids.copy() if overlap else np.arange(100, 100 + count, dtype=np.int64)
    return MultiModalTrainBatch(
        {
            "image": TrainModalityBatch(
                "image",
                rng.normal(size=(count, 12)).astype(np.float32),
                image_ids,
                LabelObservation(labels, known),
            ),
            "text": TrainModalityBatch(
                "text",
                rng.normal(size=(count, 10)).astype(np.float32),
                text_ids,
                LabelObservation(labels, known),
            ),
        }
    )


def test_removed_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="removed field"):
        load_bc_gtsh_method_config({"num_hubs": 64})


def test_source_overlap_requires_explicit_oracle_flag() -> None:
    config = BCGTSHMethodConfig(epochs=1, mini_batch_size=8, embed_dim=16)
    method = BCGTSHMethod(config, context())
    with pytest.raises(ValueError, match="overlapping source IDs"):
        method.fit_stage(training_batch(overlap=True), stage())


def test_small_source_disjoint_training_is_finite() -> None:
    config = BCGTSHMethodConfig(
        epochs=1,
        mini_batch_size=8,
        embed_dim=16,
        topology_intra_k=2,
        topology_cross_k=2,
    )
    method = BCGTSHMethod(config, context())
    result = method.fit_stage(training_batch(), stage())
    assert result.log["source_overlap_count"] == 0
    assert np.isfinite(result.log["training"]["total"])


class OldDisabledPath(nn.Module):
    """Architecture of the archived disabled-memory encoder path."""

    def __init__(self) -> None:
        super().__init__()
        self.image_projection = BCGTSHEncoder._projection(12, 16, 0.0)
        self.text_projection = BCGTSHEncoder._projection(10, 16, 0.0)
        self.memory_projection = nn.Linear(16, 16)
        self.memory_gate = nn.Linear(32, 1)
        self.hash_layer = nn.Linear(16, 8)
        nn.init.zeros_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)
        nn.init.zeros_(self.memory_gate.weight)
        nn.init.constant_(self.memory_gate.bias, -3.0)

    def encode_image(self, values: torch.Tensor) -> torch.Tensor:
        latent = torch.nn.functional.normalize(self.image_projection(values), dim=1)
        return torch.tanh(self.hash_layer(latent))


def test_clean_encoder_matches_archived_disabled_path() -> None:
    torch.manual_seed(9)
    old = OldDisabledPath().eval()
    clean = BCGTSHEncoder(
        image_dim=12, text_dim=10, embed_dim=16, bit=8, dropout=0.0
    ).eval()
    clean.load_state_dict(
        {
            name: value
            for name, value in old.state_dict().items()
            if name.startswith(("image_projection.", "text_projection.", "hash_layer."))
        }
    )
    values = torch.randn(7, 12)
    assert torch.equal(old.encode_image(values), clean.encode_image(values)["continuous"])


def test_archived_checkpoint_migration(tmp_path: Path) -> None:
    torch.manual_seed(11)
    old = OldDisabledPath().eval()
    checkpoint = tmp_path / "old.pt"
    torch.save({"model": old.state_dict(), "completed_stages": 4}, checkpoint)
    method = BCGTSHMethod(
        BCGTSHMethodConfig(embed_dim=16, dropout=0.0), context(bit=8)
    )
    method.load_checkpoint(str(checkpoint), allow_legacy_nohub=True)
    values = torch.randn(5, 12)
    with torch.no_grad():
        expected = old.encode_image(values)
        actual = method.model.encode_image(values)["continuous"]
    assert torch.equal(expected, actual)
    assert method.completed_stages == 4
