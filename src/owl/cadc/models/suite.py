"""Integrated CADC-MORE 2 training suite with independently inspectable heads."""

from __future__ import annotations

from typing import Any

from owl.cadc.models._optional import TorchModule, require_torch
from owl.cadc.models.epistemic import EpistemicValueHead, ExternalityHead
from owl.cadc.models.experts import ActionFamilyExperts
from owl.cadc.models.ranker import PairwiseRanker
from owl.cadc.models.transition import StructuralTransitionModel
from owl.cadc.schema import ACTION_FAMILY_REGISTRY, ActionFamily


class CADCMore2Suite(TorchModule):
    """Structural ensemble member plus rank, family, information, and externality heads."""

    def __init__(self, structural: StructuralTransitionModel, hidden_dim: int) -> None:
        torch, _ = require_torch()
        super().__init__()
        self._torch = torch
        self.structural = structural
        self.ranker = PairwiseRanker(hidden_dim, hidden_dim)
        self.family_experts = ActionFamilyExperts(hidden_dim, 1)
        self.epistemic = EpistemicValueHead(hidden_dim, hidden_dim)
        self.externality = ExternalityHead(hidden_dim, hidden_dim)
        ordered_families = tuple(ActionFamily)
        family_lookup = {value: index for index, value in enumerate(ordered_families)}
        family_index = [
            family_lookup[value.primary_family] for value in ACTION_FAMILY_REGISTRY
        ]
        self.action_family_index: Any
        self.register_buffer(
            "action_family_index", torch.tensor(family_index, dtype=torch.long)
        )

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Evaluate all structural, ranking, family, and auxiliary heads."""
        output = self.structural(*args, **kwargs)
        embedding = output["embedding"]
        family = self.action_family_index[None, :].expand(embedding.shape[0], -1)
        return {
            **output,
            "rank_score": self.ranker(embedding),
            "family_value": self.family_experts(embedding, family).squeeze(-1),
            "epistemic_head": self.epistemic(embedding),
            "externality_head": self.externality(embedding),
        }
