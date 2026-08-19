from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_jaccard(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    intersection = labels @ labels.T
    cardinality = labels.sum(dim=1, keepdim=True)
    union = cardinality + cardinality.T - intersection
    return intersection / union.clamp_min(1e-8)


def privileged_topology(
    labels: torch.Tensor,
    *,
    image_count: int,
    intra_k: int,
    cross_k: int,
) -> torch.Tensor:
    """Build the modality-balanced, first-order semantic teacher graph."""
    count = labels.shape[0]
    if count <= 1:
        return labels.new_zeros((count, count))
    if not 0 <= image_count <= count:
        raise ValueError("image_count must be within the batch")

    similarity = pairwise_jaccard(labels)
    adjacency = torch.zeros_like(similarity)
    modality = torch.zeros(count, dtype=torch.long, device=labels.device)
    modality[image_count:] = 1
    eye = torch.eye(count, dtype=torch.bool, device=labels.device)
    same_modality = modality[:, None] == modality[None, :]

    for candidates, requested in (
        (same_modality & ~eye, intra_k),
        (~same_modality, cross_k),
    ):
        if requested <= 0:
            continue
        available = candidates.sum(dim=1)
        if not bool((available > 0).any()):
            continue
        k = min(int(requested), int(available.max().item()))
        values, indices = torch.topk(
            similarity.masked_fill(~candidates, float("-inf")), k=k, dim=1
        )
        valid = torch.isfinite(values)
        previous = adjacency.gather(1, indices)
        adjacency.scatter_(1, indices, torch.where(valid, values, previous))

    adjacency = torch.maximum(adjacency, adjacency.T)
    adjacency = adjacency + torch.eye(count, device=labels.device, dtype=labels.dtype)
    degree = adjacency.sum(dim=1).clamp_min(1e-8).rsqrt()
    target = degree[:, None] * adjacency * degree[None, :]
    target = target / target.max(dim=1, keepdim=True).values.clamp_min(1e-8)
    target = target.clamp(0.0, 1.0)
    target.fill_diagonal_(0.0)
    return target


def topology_distillation_loss(
    continuous: torch.Tensor,
    target: torch.Tensor,
    *,
    logit_scale: float,
    objective: str = "row_kl",
    teacher_temperature: float = 0.1,
    negative_ratio: float = 3.0,
) -> torch.Tensor:
    count = continuous.shape[0]
    if count <= 1:
        return continuous.sum() * 0.0
    logits = logit_scale * (continuous @ continuous.T) / continuous.shape[1]
    eye = torch.eye(count, dtype=torch.bool, device=continuous.device)
    if objective == "row_kl":
        teacher_logits = (target / teacher_temperature).masked_fill(
            eye, torch.finfo(target.dtype).min
        )
        student_logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
        return F.kl_div(
            torch.log_softmax(student_logits, dim=1),
            torch.softmax(teacher_logits, dim=1).detach(),
            reduction="batchmean",
        )
    if objective != "bce":
        raise ValueError(f"Unsupported topology objective '{objective}'")

    positive = (target > 1e-8) & ~eye
    negative = ~positive & ~eye
    positive_count = int(positive.sum().item())
    negative_count = int(negative.sum().item())
    if positive_count == 0:
        mask = negative
    elif negative_count == 0 or negative_ratio <= 0:
        mask = positive
    else:
        probability = min(1.0, negative_ratio * positive_count / negative_count)
        mask = positive | (negative & (torch.rand_like(target) < probability))
    if not bool(mask.any()):
        return continuous.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask])


def quantization_loss(continuous: torch.Tensor) -> torch.Tensor:
    target = torch.where(continuous.detach() >= 0, 1.0, -1.0)
    return F.mse_loss(continuous, target)


def bit_quality_loss(continuous: torch.Tensor) -> torch.Tensor:
    if continuous.shape[0] <= 1:
        return continuous.sum() * 0.0
    balance = continuous.mean(dim=0).square().mean()
    correlation = continuous.T @ continuous / continuous.shape[0]
    off_diagonal = correlation - torch.diag(torch.diag(correlation))
    return balance + off_diagonal.square().mean()

