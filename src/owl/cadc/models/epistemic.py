"""Control-relevant epistemic value and patch/collective externality heads."""

from __future__ import annotations

from typing import Any

from owl.cadc.models._optional import TorchModule, require_torch


class EpistemicValueHead(TorchModule):
    """Predict SENSE/COMMUNICATE information value from agent-visible inputs."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        _, nn = require_torch()
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, embedding: Any) -> dict[str, Any]:
        """Predict the registered control-relevant information outcomes."""
        output = self.network(embedding)
        return {
            "new_information": output[..., 0],
            "later_action_change_logit": output[..., 1],
            "later_value_improvement": output[..., 2],
            "cost_adjusted_control_value": output[..., 3],
        }


class ExternalityHead(TorchModule):
    """Predict patch, resource, population, and lineage externality components."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        _, nn = require_torch()
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 5)
        )

    def forward(self, embedding: Any) -> dict[str, Any]:
        """Predict registered patch and collective externality outcomes."""
        output = self.network(embedding)
        return {
            "population_delta_vs_anchor": output[..., 0],
            "world_food_delta_vs_anchor": output[..., 1],
            "world_toxin_delta_vs_anchor": output[..., 2],
            "world_waste_delta_vs_anchor": output[..., 3],
            "focal_lineage_persistence_delta_vs_anchor": output[..., 4],
        }
