"""Multi-head structural action-conditioned transition ensemble."""

from __future__ import annotations

from typing import Any

from owl.cadc.models._optional import TorchModule, require_torch
from owl.cadc.models.encoder import StructuredContextEncoder
from owl.cadc.models.survival import CompetingRiskHead, MonotoneQuantileHead


class StructuralTransitionModel(TorchModule):
    """Predict raw outcomes, uncertainty, quantiles, survival, and externality."""

    def __init__(
        self,
        *,
        context_dim: int,
        candidate_dim: int,
        direction_dim: int,
        hidden_dim: int,
        outcome_dim: int,
        quantile_count: int,
        time_bins: int,
        death_causes: int,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        _, nn = require_torch()
        super().__init__()
        self.encoder = StructuredContextEncoder(
            context_dim,
            candidate_dim,
            direction_dim,
            hidden_dim,
            depth=depth,
            dropout=dropout,
            horizon_count=time_bins,
        )
        self.outcome_mean = nn.Linear(hidden_dim, outcome_dim)
        self.outcome_log_scale = nn.Linear(hidden_dim, outcome_dim)
        self.quantiles = MonotoneQuantileHead(hidden_dim, quantile_count)
        self.survival = CompetingRiskHead(hidden_dim, time_bins, death_causes)
        self.externality = nn.Linear(hidden_dim, 5)
        self.information_value = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context: Any,
        candidates: Any,
        directions: Any,
        direction_mask: Any,
        horizon_index: Any,
    ) -> dict[str, Any]:
        """Predict the complete registered outcome vector for 22 candidates."""
        _, _, combined = self.encoder(
            context, candidates, directions, direction_mask, horizon_index
        )
        return {
            "outcome_mean": self.outcome_mean(combined),
            "outcome_log_scale": self.outcome_log_scale(combined).clamp(-12.0, 8.0),
            "return_quantiles": self.quantiles(combined),
            "competing_risk_logits": self.survival(combined),
            "externality": self.externality(combined),
            "information_value": self.information_value(combined).squeeze(-1),
            "embedding": combined,
        }


class StructuralEnsemble(TorchModule):
    """Fixed-member ensemble; missing members are never silently ignored."""

    def __init__(self, members: list[StructuralTransitionModel]) -> None:
        torch, nn = require_torch()
        super().__init__()
        if not members:
            raise ValueError("structural ensemble needs at least one member")
        self._torch = torch
        self.members = nn.ModuleList(members)

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Aggregate all required ensemble members and retain member outputs."""
        outputs = [member(*args, **kwargs) for member in self.members]
        means = self._torch.stack([value["outcome_mean"] for value in outputs], dim=0)
        return {
            "member_outputs": outputs,
            "outcome_mean": means.mean(dim=0),
            "epistemic_variance": means.var(dim=0, unbiased=len(outputs) > 1),
        }
