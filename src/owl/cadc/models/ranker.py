"""Pairwise and masked listwise candidate ranking.

The loss families follow the learning-to-rank precedents in Burges et al.
(2005) and Cao et al. (2007); CADC adds immutable action slots and executable
masks as project-specific contracts. See ``docs/REFERENCES.md`` [R29–R30].
"""

from __future__ import annotations

from typing import Any

from owl.cadc.models._optional import TorchModule, require_torch


class PairwiseRanker(TorchModule):
    """Score each candidate embedding with one shared action-equivariant head."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        _, nn = require_torch()
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, candidate_embedding: Any) -> Any:
        """Score all 22 candidates with a shared equivariant scorer."""
        if candidate_embedding.ndim != 3 or candidate_embedding.shape[1] != 22:
            raise ValueError("ranker expects fixed candidate embeddings [B,22,D]")
        return self.scorer(candidate_embedding).squeeze(-1)


def pairwise_loss(
    score_a: Any,
    score_b: Any,
    win_label: Any,
    *,
    magnitude: Any | None = None,
    tie_margin: float = 0.0,
) -> Any:
    """Weighted logistic paired loss with symmetric tie handling."""
    torch, nn = require_torch()
    delta = score_a - score_b
    label = win_label.to(dtype=delta.dtype)
    decisive = label != 0.5
    signed = torch.where(label > 0.5, 1.0, -1.0)
    decisive_loss = nn.functional.softplus(-signed * delta)
    tie_loss = torch.relu(torch.abs(delta) - tie_margin)
    loss = torch.where(decisive, decisive_loss, tie_loss)
    if magnitude is not None:
        loss = loss * magnitude.to(dtype=loss.dtype).clamp_min(0.0)
    return loss.mean()


def listwise_loss(scores: Any, targets: Any, executable_mask: Any) -> Any:
    """Masked Plackett-style cross entropy over executable candidate slots."""
    torch, _ = require_torch()
    if scores.shape != targets.shape or scores.shape != executable_mask.shape:
        raise ValueError("listwise scores, targets, and mask must match")
    if scores.ndim != 2 or scores.shape[1] != 22:
        raise ValueError("listwise tensors must have shape [B,22]")
    valid_rows = executable_mask.sum(dim=-1) >= 2
    if not bool(valid_rows.any()):
        return scores.sum() * 0.0
    negative = torch.finfo(scores.dtype).min
    predicted_log = torch.log_softmax(scores.masked_fill(~executable_mask, negative), dim=-1)
    target_log = torch.log_softmax(targets.masked_fill(~executable_mask, negative), dim=-1)
    target_probability = target_log.exp().masked_fill(~executable_mask, 0.0)
    loss = -(target_probability * predicted_log).sum(dim=-1)
    return loss[valid_rows].mean()
