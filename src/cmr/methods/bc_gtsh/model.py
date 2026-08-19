from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BCGTSHEncoder(nn.Module):
    """Modality-specific projections followed by a shared hash layer."""

    def __init__(
        self,
        *,
        image_dim: int,
        text_dim: int,
        embed_dim: int,
        bit: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.image_projection = self._projection(image_dim, embed_dim, dropout)
        self.text_projection = self._projection(text_dim, embed_dim, dropout)
        # Preserve the random-number position of the frozen experiment code so
        # a clean run starts from the same shared hash-layer initialization.
        nn.Linear(embed_dim, embed_dim)
        nn.Linear(embed_dim * 2, 1)
        self.hash_layer = nn.Linear(embed_dim, bit)

    @staticmethod
    def _projection(input_dim: int, embed_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim),
        )

    def project_image(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_projection(values), dim=1)

    def project_text(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_projection(values), dim=1)

    def encode_image(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode_latent(self.project_image(values))

    def encode_text(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode_latent(self.project_text(values))

    def encode_latent(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        continuous = torch.tanh(self.hash_layer(latent))
        return {"latent": latent, "continuous": continuous}
