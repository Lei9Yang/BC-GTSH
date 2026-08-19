from __future__ import annotations

import torch

from cmr.methods.bc_gtsh.losses import (
    pairwise_jaccard,
    privileged_topology,
    topology_distillation_loss,
)


def test_jaccard_is_permutation_equivariant() -> None:
    labels = torch.tensor(
        [[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.float32
    )
    permutation = torch.tensor([2, 0, 3, 1])
    expected = pairwise_jaccard(labels)[permutation][:, permutation]
    assert torch.allclose(pairwise_jaccard(labels[permutation]), expected)


def test_row_kl_is_invariant_to_independent_modality_reindexing() -> None:
    generator = torch.Generator().manual_seed(6513)
    labels = torch.randint(0, 2, (8, 5), generator=generator).float()
    labels[:, 0] = 1
    continuous = torch.randn(8, 16, generator=generator).tanh()
    target = privileged_topology(labels, image_count=4, intra_k=2, cross_k=2)
    loss = topology_distillation_loss(
        continuous,
        target,
        logit_scale=5.0,
        objective="row_kl",
        teacher_temperature=0.1,
    )
    permutation = torch.tensor([2, 0, 3, 1, 7, 5, 4, 6])
    permuted_loss = topology_distillation_loss(
        continuous[permutation],
        target[permutation][:, permutation],
        logit_scale=5.0,
        objective="row_kl",
        teacher_temperature=0.1,
    )
    assert torch.allclose(loss, permuted_loss, atol=1e-6, rtol=1e-6)


def test_topk_cutoff_ties_can_depend_on_row_order() -> None:
    labels = torch.tensor([[0, 0, 1]] * 6, dtype=torch.float32)
    original = privileged_topology(labels, image_count=3, intra_k=1, cross_k=1)
    permutation = torch.tensor([1, 0, 2, 4, 3, 5])
    inverse = torch.argsort(permutation)
    reordered = privileged_topology(
        labels[permutation], image_count=3, intra_k=1, cross_k=1
    )[inverse][:, inverse]
    assert not torch.allclose(original, reordered)


def test_row_kl_implies_pinsker_total_variation_bound() -> None:
    teacher = torch.tensor([[0.55, 0.30, 0.15]], dtype=torch.float64)
    student = torch.tensor([[0.50, 0.25, 0.25]], dtype=torch.float64)
    kl = (teacher * (teacher.log() - student.log())).sum(dim=1)
    total_variation = 0.5 * (teacher - student).abs().sum(dim=1)
    assert bool(torch.all(total_variation <= torch.sqrt(kl / 2) + 1e-12))

