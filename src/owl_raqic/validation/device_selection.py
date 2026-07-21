from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ValidationSelection:
    flat_indices: Any
    reason_codes: Any
    signature_ids: Any


def select_validation_rows_device(ds: Any, *, limit: int) -> ValidationSelection:
    """Select bounded adversarial validation rows without copying full tensors."""

    xp = ds.xp
    probabilities = ds.arrays["raqic_probabilities"]
    h, w, actions = probabilities.shape
    flat = probabilities.reshape(-1, actions)
    live = ((ds.health > 0) & (~ds.obstacle)).reshape(-1)
    entropy = -xp.sum(
        xp.where(flat > 0, flat * xp.log(xp.maximum(flat, 1e-15)), 0.0),
        axis=1,
    )
    top_prob = xp.max(flat, axis=1)
    authority = ds.arrays.get("_authority_bool", ds.arrays.get("authority"))
    if authority is None:
        authority_flat = xp.ones_like(flat, dtype=bool)
    else:
        authority_flat = authority.reshape(-1, actions).astype(bool)
    legal_count = xp.sum(authority_flat, axis=1)
    health = ds.health.reshape(-1)
    resource = ds.resource.reshape(-1)

    # Score categories: uncertainty, near deterministic, one-legal action,
    # health/resource thresholds. Ineligible cells receive -inf.
    normalized_entropy = entropy / xp.log(float(max(2, actions)))
    threshold_score = 1.0 - xp.minimum(
        xp.abs(health - 0.25) + xp.abs(resource - 0.25),
        1.0,
    )
    score = 4.0 * normalized_entropy + 2.0 * top_prob + 3.0 * (legal_count == 1) + threshold_score
    score = xp.where(live, score, -xp.inf)
    k = min(max(0, int(limit)), int(flat.shape[0]))
    if k == 0:
        empty = xp.zeros((0,), dtype=xp.int64)
        return ValidationSelection(empty, empty.astype(xp.int32), empty.astype(xp.int64))
    indices = xp.argpartition(score, -k)[-k:]
    indices = indices[xp.isfinite(score[indices])]
    order = xp.argsort(score[indices])[::-1]
    indices = indices[order]
    reason = xp.zeros(indices.shape, dtype=xp.int32)
    reason = xp.where(normalized_entropy[indices] > 0.75, 1, reason)
    reason = xp.where(top_prob[indices] > 0.95, 2, reason)
    reason = xp.where(legal_count[indices] == 1, 3, reason)
    signature = legal_count[indices].astype(xp.int64) * xp.int64(1_000_003) + xp.argmax(
        flat[indices], axis=1
    ).astype(xp.int64)
    return ValidationSelection(indices.astype(xp.int64), reason, signature)
