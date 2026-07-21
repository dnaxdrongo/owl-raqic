"""Discrete-time competing-risk and monotone quantile heads."""

from __future__ import annotations

from typing import Any

from owl.cadc.models._optional import TorchModule, require_torch


class CompetingRiskHead(TorchModule):
    """Produce per-bin no-event and cause-specific logits."""

    def __init__(self, input_dim: int, time_bins: int, causes: int) -> None:
        _, nn = require_torch()
        super().__init__()
        if time_bins < 1 or causes < 1:
            raise ValueError("competing-risk dimensions must be positive")
        self.time_bins = time_bins
        self.causes = causes
        self.projection = nn.Linear(input_dim, time_bins * (causes + 1))

    def forward(self, embedding: Any) -> Any:
        """Project embeddings to discrete cause-specific hazard logits."""
        logits = self.projection(embedding)
        return logits.reshape(*embedding.shape[:-1], self.time_bins, self.causes + 1)


class MonotoneQuantileHead(TorchModule):
    """Produce nondecreasing quantiles through positive softplus increments."""

    def __init__(self, input_dim: int, quantile_count: int) -> None:
        _, nn = require_torch()
        super().__init__()
        if quantile_count < 2:
            raise ValueError("at least two quantiles are required")
        self.quantile_count = quantile_count
        self.base = nn.Linear(input_dim, 1)
        self.increments = nn.Linear(input_dim, quantile_count - 1)
        self.softplus = nn.Softplus()

    def forward(self, embedding: Any) -> Any:
        """Project embeddings to ordered conditional return quantiles."""
        torch, _ = require_torch()
        base = self.base(embedding)
        positive = self.softplus(self.increments(embedding))
        return torch.cat((base, base + torch.cumsum(positive, dim=-1)), dim=-1)


def competing_risk_nll(logits: Any, event_bin: Any, cause: Any, valid: Any) -> Any:
    """Stable discrete-time cause-specific negative log likelihood."""
    torch, _ = require_torch()
    if logits.ndim < 3:
        raise ValueError("competing-risk logits need [B,T,C+1]")
    log_probabilities = torch.log_softmax(logits, dim=-1)
    batch = logits.shape[0]
    bins = logits.shape[-2]
    index = torch.arange(bins, device=logits.device)[None, :]
    before = index < event_bin[:, None]
    at = index == event_bin[:, None]
    no_event = log_probabilities[..., 0]
    selected = log_probabilities.gather(
        -1, cause[:, None, None].expand(batch, bins, 1)
    ).squeeze(-1)
    likelihood = (before * no_event + at * selected) * valid[:, None]
    denominator = valid.sum().clamp_min(1)
    return -likelihood.sum() / denominator
