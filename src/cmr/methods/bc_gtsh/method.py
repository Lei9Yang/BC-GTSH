from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from cmr.evaluation.efficiency import (
    cuda_synchronize,
    module_statistics,
    peak_vram,
    reset_peak_vram,
    runtime_metadata,
)
from cmr.evaluation.metrics import sign_codes
from cmr.framework.contracts import (
    CrossModalBatch,
    EncodedModalities,
    EncodedModality,
    MethodBuildContext,
    MethodCapabilities,
    MultiModalTrainBatch,
    StageContext,
    StageTrainResult,
    TrainModalityBatch,
)

from .losses import (
    bit_quality_loss,
    privileged_topology,
    quantization_loss,
    topology_distillation_loss,
)
from .model import BCGTSHEncoder

LEGACY_FIELDS = frozenset(
    {
        "calibration_epochs",
        "num_hubs",
        "route_top_k",
        "teacher_top_k",
        "topology_diffusion",
        "route_temperature",
        "teacher_temperature",
        "teacher_feature_weight",
        "hub_eta_max",
        "hub_support_smoothing",
        "dead_hub_min_support",
        "dead_hub_patience",
        "lambda_route",
        "lambda_compat",
        "lambda_anchor",
        "lambda_xver",
        "compatibility_warmup_stages",
        "hub_update_mode",
        "use_hub_memory",
    }
)


@dataclass(frozen=True)
class BCGTSHMethodConfig:
    variant: str = "BC-GTSH-RowKL"
    epochs: int = 30
    mini_batch_size: int = 256
    eval_batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    embed_dim: int = 256
    topology_intra_k: int = 5
    topology_cross_k: int = 5
    topology_objective: str = "row_kl"
    topology_teacher_temperature: float = 0.1
    topology_negative_ratio: float = 3.0
    logit_scale: float = 5.0
    lambda_topology: float = 1.0
    lambda_quant: float = 0.01
    lambda_bit: float = 0.001
    dropout: float = 0.1
    unpaired_protocol: str = "disjoint"
    allow_source_overlap: bool = False
    update_after_stage_zero: bool = True


def load_bc_gtsh_method_config(payload: Mapping[str, Any]) -> BCGTSHMethodConfig:
    rejected = sorted(set(payload) & LEGACY_FIELDS)
    if rejected:
        raise ValueError(
            "The public Row-KL implementation does not accept removed field "
            f"'$.method.parameters.{rejected[0]}'"
        )
    allowed = {item.name for item in fields(BCGTSHMethodConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"$.method.parameters.{unknown[0]} is not a recognized BC-GTSH field"
        )
    try:
        config = BCGTSHMethodConfig(**payload)
    except TypeError as error:
        raise ValueError(f"$.method.parameters: {error}") from error

    for name in (
        "epochs",
        "mini_batch_size",
        "eval_batch_size",
        "embed_dim",
        "topology_intra_k",
        "topology_cross_k",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"$.method.parameters.{name} must be an integer")
    if config.epochs <= 0 or config.mini_batch_size < 2 or config.eval_batch_size <= 0:
        raise ValueError("epochs/eval_batch_size must be positive and mini_batch_size >= 2")
    if config.embed_dim <= 0 or config.topology_intra_k < 0 or config.topology_cross_k < 0:
        raise ValueError("embed_dim must be positive and topology k values non-negative")
    if config.topology_objective not in {"row_kl", "bce"}:
        raise ValueError("topology_objective must be 'row_kl' or 'bce'")
    if config.unpaired_protocol not in {"disjoint", "deranged"}:
        raise ValueError("unpaired_protocol must be 'disjoint' or 'deranged'")
    for name in ("allow_source_overlap", "update_after_stage_zero"):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"$.method.parameters.{name} must be boolean")
    for name in (
        "lr",
        "weight_decay",
        "grad_clip",
        "topology_negative_ratio",
        "lambda_topology",
        "lambda_quant",
        "lambda_bit",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"$.method.parameters.{name} must be non-negative")
    if config.topology_teacher_temperature <= 0 or config.logit_scale <= 0:
        raise ValueError("topology temperature and logit_scale must be positive")
    if not 0 <= config.dropout <= 1:
        raise ValueError("dropout must be in [0, 1]")
    return config


class BCGTSHMethod:
    method_id = "bc-gtsh"
    display_name = "BC-GTSH"
    capabilities = MethodCapabilities(
        continuous_embeddings=True,
        checkpoint=True,
        train_log=True,
        required_modalities=("image", "text"),
        supports_independent_modalities=True,
        supported_supervision=("full",),
    )

    def __init__(self, config: BCGTSHMethodConfig, context: MethodBuildContext) -> None:
        self.config = config
        self.context = context
        self.variant = config.variant
        torch.manual_seed(context.seed)
        torch.cuda.manual_seed_all(context.seed)
        self.device = torch.device(
            context.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = BCGTSHEncoder(
            image_dim=context.image_dim,
            text_dim=context.text_dim,
            embed_dim=config.embed_dim,
            bit=context.bit,
            dropout=config.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.completed_stages = 0
        self.has_fitted = False

    def fit_stage(
        self,
        batch: CrossModalBatch | MultiModalTrainBatch,
        context: StageContext,
    ) -> StageTrainResult:
        reset_peak_vram(self.device)
        started = time.perf_counter()
        image_values, image_labels, image_ids, text_values, text_labels, text_ids, protocol = (
            self._training_arrays(batch, context.stage_index)
        )
        overlap = int(np.intersect1d(image_ids, text_ids).size)
        if isinstance(batch, MultiModalTrainBatch) and overlap and not self.config.allow_source_overlap:
            raise ValueError("Externally source-disjoint streams contain overlapping source IDs")

        should_update = self.config.update_after_stage_zero or context.stage_index == 0
        train_log = self._train(
            torch.from_numpy(image_values).float().to(self.device),
            torch.from_numpy(image_labels).float().to(self.device),
            torch.from_numpy(text_values).float().to(self.device),
            torch.from_numpy(text_labels).float().to(self.device),
            epochs=self.config.epochs if should_update else 0,
        )
        self.completed_stages += 1
        self.has_fitted = True
        cuda_synchronize(self.device)
        elapsed = time.perf_counter() - started
        log = {
            "training": train_log,
            "unpaired_protocol": protocol,
            "image_train_count": int(len(image_ids)),
            "text_train_count": int(len(text_ids)),
            "source_overlap_count": overlap,
            "image_label_coverage": int(np.any(image_labels > 0, axis=0).sum()),
            "text_label_coverage": int(np.any(text_labels > 0, axis=0).sum()),
            "parameter_update": should_update,
            "topology_objective": self.config.topology_objective,
            "loss_weights": {
                "topology": float(self.config.lambda_topology),
                "quantization": float(self.config.lambda_quant),
                "bit_balance": float(self.config.lambda_bit),
            },
            **peak_vram(self.device),
        }
        return StageTrainResult(
            elapsed_time_s=elapsed,
            log=log,
            metadata={"strict_source_disjoint_training": overlap == 0},
        )

    def encode(self, batch: CrossModalBatch, context: StageContext) -> EncodedModalities:
        image = self.encode_image(batch.images, context)
        text = self.encode_text(batch.texts, context)
        return EncodedModalities(
            image_codes=image.codes,
            text_codes=text.codes,
            image_embeddings=image.embeddings,
            text_embeddings=text.embeddings,
        )

    def encode_image(self, features: np.ndarray, context: StageContext) -> EncodedModality:
        return self._encode_single(features, "image")

    def encode_text(self, features: np.ndarray, context: StageContext) -> EncodedModality:
        return self._encode_single(features, "text")

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {
            **runtime_metadata(self.device),
            **module_statistics(self.model),
            "bc_gtsh_completed_stages": self.completed_stages,
        }

    def save_checkpoint(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 2,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "completed_stages": self.completed_stages,
            },
            target,
        )

    def load_checkpoint(self, path: str, *, allow_legacy_nohub: bool = False) -> None:
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        state = payload["model"]
        if payload.get("schema_version") != 2:
            if not allow_legacy_nohub:
                raise ValueError("Legacy checkpoint requires allow_legacy_nohub=True")
            state = {
                name: value
                for name, value in state.items()
                if name.startswith(("image_projection.", "text_projection.", "hash_layer."))
            }
        self.model.load_state_dict(state, strict=True)
        if payload.get("schema_version") == 2 and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
            for optimizer_state in self.optimizer.state.values():
                for name, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_state[name] = value.to(self.device)
        self.completed_stages = int(payload.get("completed_stages", 0))
        self.has_fitted = True

    def _training_arrays(self, batch: CrossModalBatch | MultiModalTrainBatch, stage: int):
        if isinstance(batch, MultiModalTrainBatch):
            try:
                image_batch = batch.modalities["image"]
                text_batch = batch.modalities["text"]
            except KeyError as error:
                raise ValueError("BC-GTSH requires image and text streams") from error
            image_values, image_labels = self._fully_supervised(image_batch)
            text_values, text_labels = self._fully_supervised(text_batch)
            return (
                image_values,
                image_labels,
                image_batch.ids,
                text_values,
                text_labels,
                text_batch.ids,
                "external",
            )
        if batch.images.shape[0] < 2:
            raise ValueError("BC-GTSH requires at least two source instances per stage")
        image_indices, text_indices = self._unpaired_indices(batch, stage)
        return (
            batch.images[image_indices],
            batch.labels[image_indices],
            batch.ids[image_indices],
            batch.texts[text_indices],
            batch.labels[text_indices],
            batch.ids[text_indices],
            self.config.unpaired_protocol,
        )

    def _train(
        self,
        images: torch.Tensor,
        image_labels: torch.Tensor,
        texts: torch.Tensor,
        text_labels: torch.Tensor,
        *,
        epochs: int,
    ) -> dict[str, float]:
        sums = {"total": 0.0, "topology": 0.0, "quantization": 0.0, "bit": 0.0}
        updates = 0
        if epochs == 0:
            return {"epochs": 0.0, "updates": 0.0, **sums}
        half_batch = max(1, self.config.mini_batch_size // 2)
        steps = max(
            1,
            math.ceil(images.shape[0] / half_batch),
            math.ceil(texts.shape[0] / half_batch),
        )
        self.model.train()
        for _ in range(epochs):
            image_order = torch.randperm(images.shape[0], device=self.device)
            text_order = torch.randperm(texts.shape[0], device=self.device)
            for step in range(steps):
                image_indices = self._cyclic_slice(image_order, step * half_batch, half_batch)
                text_indices = self._cyclic_slice(text_order, step * half_batch, half_batch)
                image_output = self.model.encode_image(images[image_indices])
                text_output = self.model.encode_text(texts[text_indices])
                continuous = torch.cat(
                    [image_output["continuous"], text_output["continuous"]], dim=0
                )
                labels = torch.cat(
                    [image_labels[image_indices], text_labels[text_indices]], dim=0
                )
                zero = continuous.sum() * 0.0
                topology_loss = zero
                if self.config.lambda_topology > 0:
                    teacher = privileged_topology(
                        labels,
                        image_count=image_indices.shape[0],
                        intra_k=self.config.topology_intra_k,
                        cross_k=self.config.topology_cross_k,
                    )
                    topology_loss = topology_distillation_loss(
                        continuous,
                        teacher,
                        logit_scale=self.config.logit_scale,
                        objective=self.config.topology_objective,
                        teacher_temperature=self.config.topology_teacher_temperature,
                        negative_ratio=self.config.topology_negative_ratio,
                    )
                quant = quantization_loss(continuous)
                bit = bit_quality_loss(continuous)
                total = (
                    self.config.lambda_topology * topology_loss
                    + self.config.lambda_quant * quant
                    + self.config.lambda_bit * bit
                )
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                for name, value in {
                    "total": total,
                    "topology": topology_loss,
                    "quantization": quant,
                    "bit": bit,
                }.items():
                    sums[name] += float(value.detach().item())
                updates += 1
        return {
            "epochs": float(epochs),
            "updates": float(updates),
            **{name: value / max(1, updates) for name, value in sums.items()},
        }

    def _encode_single(self, features: np.ndarray, modality: str) -> EncodedModality:
        if not self.has_fitted:
            raise RuntimeError("BC-GTSH must be fitted or restored before encoding")
        if features.ndim != 2:
            raise ValueError(f"{modality} features must be two-dimensional")
        self.model.eval()
        values: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, features.shape[0], self.config.eval_batch_size):
                tensor = torch.from_numpy(features[start : start + self.config.eval_batch_size]).float().to(self.device)
                output = self.model.encode_image(tensor) if modality == "image" else self.model.encode_text(tensor)
                values.append(output["continuous"].cpu().numpy())
        continuous = (
            np.concatenate(values, axis=0)
            if values
            else np.empty((0, self.context.bit), dtype=np.float32)
        )
        return EncodedModality(codes=sign_codes(continuous), embeddings=continuous)

    def _unpaired_indices(self, batch: CrossModalBatch, stage: int) -> tuple[np.ndarray, np.ndarray]:
        count = batch.images.shape[0]
        ids = np.asarray(batch.ids, dtype=np.int64)
        keys = ids.astype(np.uint64) * np.uint64(11400714819323198485) + np.uint64(
            self.context.seed + stage * 1009
        )
        order = np.argsort(keys, kind="stable")
        if self.config.unpaired_protocol == "disjoint":
            split = max(1, min(count - 1, count // 2))
            return np.sort(order[:split]), np.sort(order[split:])
        image_indices = np.arange(count, dtype=np.int64)
        shift = 1 + (self.context.seed + stage) % (count - 1)
        return image_indices, np.roll(image_indices, shift)

    @staticmethod
    def _fully_supervised(batch: TrainModalityBatch) -> tuple[np.ndarray, np.ndarray]:
        if batch.supervision is None or not bool(np.all(batch.supervision.known_mask)):
            raise ValueError("BC-GTSH requires fully observed training labels")
        return batch.features, batch.supervision.values

    @staticmethod
    def _cyclic_slice(order: torch.Tensor, start: int, size: int) -> torch.Tensor:
        if order.numel() <= size:
            return order
        start %= int(order.numel())
        end = start + size
        if end <= order.numel():
            return order[start:end]
        return torch.cat([order[start:], order[: end - order.numel()]])
