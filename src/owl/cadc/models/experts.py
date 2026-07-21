"""Shared-trunk action-family residual experts."""

from __future__ import annotations

from typing import Any

from owl.cadc.models._optional import TorchModule, require_torch


class ActionFamilyExperts(TorchModule):
    """Add family residuals without changing the immutable 22-action axis."""

    def __init__(self, embedding_dim: int, output_dim: int, family_count: int = 6) -> None:
        torch, nn = require_torch()
        super().__init__()
        self._torch = torch
        self.family_count = family_count
        self.shared = nn.Linear(embedding_dim, output_dim)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.SiLU(),
                    nn.Linear(embedding_dim, output_dim),
                )
                for _ in range(family_count)
            ]
        )

    def forward(self, embedding: Any, family_index: Any) -> Any:
        """Apply the shared head plus the selected family residual head."""
        if embedding.shape[:-1] != family_index.shape:
            raise ValueError("family index must align with embedding slots")
        base = self.shared(embedding)
        stacked = self._torch.stack([expert(embedding) for expert in self.experts], dim=-2)
        index = family_index.long()[..., None, None].expand(*family_index.shape, 1, base.shape[-1])
        residual = stacked.gather(-2, index).squeeze(-2)
        return base + residual
