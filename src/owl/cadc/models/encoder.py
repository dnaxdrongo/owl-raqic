"""Structured context, action, direction-set, and history encoders."""

from __future__ import annotations

from typing import Any

from owl.cadc.models._optional import TorchModule, require_torch


class DirectionSetEncoder(TorchModule):
    """Masked DeepSets encoder for the fixed FLEE/PURSUE ``[B,2,8,F]`` tensor."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        torch, nn = require_torch()
        super().__init__()
        self._torch = torch
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim), nn.SiLU(), nn.LayerNorm(output_dim)
        )

    def forward(self, directions: Any, mask: Any) -> Any:
        """Encode masked negative and positive direction candidate sets."""
        if directions.ndim != 4 or directions.shape[1] != 2 or directions.shape[2] != 8:
            raise ValueError("direction tensor must have shape [B,2,8,F]")
        if mask.shape != directions.shape[:3]:
            raise ValueError("direction mask must have shape [B,2,8]")
        encoded = self.phi(directions)
        valid = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        count = valid.sum(dim=2).clamp_min(1.0)
        mean = (encoded * valid).sum(dim=2) / count
        negative = self._torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), negative).max(dim=2).values
        maximum = self._torch.where(mask.any(dim=2, keepdim=True), maximum, 0.0)
        family = self.rho(self._torch.cat((mean, maximum), dim=-1))
        return family.reshape(family.shape[0], -1)


class StructuredContextEncoder(TorchModule):
    """Encode context, candidate slots, directions, and optional causal history."""

    def __init__(
        self,
        context_dim: int,
        candidate_dim: int,
        direction_dim: int,
        hidden_dim: int,
        *,
        action_count: int = 22,
        horizon_count: int = 5,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        torch, nn = require_torch()
        super().__init__()
        self._torch = torch
        if depth < 1:
            raise ValueError("encoder depth must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("encoder dropout must lie in [0,1)")
        self.action_count = action_count
        context_layers: list[Any] = [
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            context_layers.extend(
                (
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )
        self.context = nn.Sequential(*context_layers)
        self.candidate = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)
        )
        self.action_embedding = nn.Embedding(action_count, hidden_dim)
        self.horizon_embedding = nn.Embedding(horizon_count, hidden_dim)
        self.direction = DirectionSetEncoder(
            direction_dim, max(16, hidden_dim // 2), hidden_dim // 2
        )
        self.direction_projection = nn.Linear(hidden_dim, hidden_dim)
        self.combine = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        context: Any,
        candidates: Any,
        directions: Any,
        direction_mask: Any,
        horizon_index: Any,
    ) -> tuple[Any, Any, Any]:
        """Encode agent context and every immutable candidate action slot."""
        if candidates.ndim != 3 or candidates.shape[1] != self.action_count:
            raise ValueError("candidate tensor must have fixed shape [B,22,F]")
        batch = context.shape[0]
        context_embedding = self.context(context)
        candidate_embedding = self.candidate(candidates)
        action_ids = self._torch.arange(
            self.action_count, device=context.device, dtype=self._torch.long
        )
        action = self.action_embedding(action_ids)[None, :, :].expand(batch, -1, -1)
        horizon = self.horizon_embedding(horizon_index.long())[:, None, :].expand(
            -1, self.action_count, -1
        )
        direction = self.direction_projection(self.direction(directions, direction_mask))
        direction = direction[:, None, :].expand(-1, self.action_count, -1)
        context_slots = context_embedding[:, None, :].expand(-1, self.action_count, -1)
        combined = self.combine(
            self._torch.cat(
                (context_slots, candidate_embedding + action, horizon, direction), dim=-1
            )
        )
        return context_embedding, candidate_embedding, combined
