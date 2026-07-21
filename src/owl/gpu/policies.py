from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyResult:
    probabilities: Any
    row_sum_residual: Any
    entropy: Any
    repair_count: Any


def _asarray(xp: Any, value: Any, dtype: Any | None = None) -> Any:
    return xp.asarray(value, dtype=dtype) if dtype is not None else xp.asarray(value)


def stable_masked_policy(
    logits: Any,
    mask: Any,
    *,
    xp: Any,
    temperature: float = 1.0,
    rest_index: int = 0,
    precision: str = "audit64",
    eps: float = 1e-12,
) -> PolicyResult:
    """Apply stable masked softmax to RAQIC, baseline policies, top-down context, and signals.

    Illegal actions receive zero probability. Rows with no legal action are
    repaired to REST-only and the repair count is returned.
    """

    dtype = xp.float64 if precision == "audit64" else xp.float32
    logits = _asarray(xp, logits, dtype=dtype)
    mask = _asarray(xp, mask, dtype=bool)
    if logits.ndim == 1:
        logits = logits[None, :]
        mask = mask[None, :]
    if logits.shape != mask.shape:
        raise ValueError(f"logits and mask shape mismatch: {logits.shape} vs {mask.shape}")
    if logits.shape[-1] <= int(rest_index):
        raise ValueError("rest_index outside action axis")

    temp = (
        dtype.type(max(float(temperature), eps))
        if hasattr(dtype, "type")
        else max(float(temperature), eps)
    )
    legal_count = xp.sum(mask, axis=-1)
    bad = legal_count <= 0
    if logits.ndim != 2:
        raise ValueError("stable_masked_policy expects 2D [N,A] after normalization")
    # Resolve empty legal-action rows to REST entirely on device. Avoiding
    # bool(device_array) prevents an implicit CuPy/CUDA stream synchronization.
    rest_only = xp.zeros_like(mask, dtype=bool)
    rest_only[:, int(rest_index)] = True
    repaired_mask = xp.where(bad[:, None], rest_only, mask)

    scaled = logits / temp
    neg_inf = xp.asarray(-xp.inf, dtype=dtype)
    masked = xp.where(repaired_mask, scaled, neg_inf)
    row_max = xp.max(masked, axis=-1, keepdims=True)
    # The REST fallback guarantees a finite normalization anchor for every row.
    shifted = xp.where(repaired_mask, masked - row_max, neg_inf)
    expv = xp.where(repaired_mask, xp.exp(shifted), xp.asarray(0.0, dtype=dtype))
    denom = xp.sum(expv, axis=-1, keepdims=True)
    probs = expv / xp.maximum(denom, xp.asarray(eps, dtype=dtype))
    # Force exact zero on illegal actions and renormalize after small roundoff.
    probs = xp.where(repaired_mask, probs, xp.asarray(0.0, dtype=dtype))
    sums = xp.sum(probs, axis=-1, keepdims=True)
    probs = probs / xp.maximum(sums, xp.asarray(eps, dtype=dtype))
    row_sums = xp.sum(probs, axis=-1)
    residual = xp.abs(row_sums - xp.asarray(1.0, dtype=dtype))
    entropy = -xp.sum(
        xp.where(probs > 0, probs * xp.log(xp.maximum(probs, xp.asarray(eps, dtype=dtype))), 0.0),
        axis=-1,
    )
    return PolicyResult(
        probabilities=probs,
        row_sum_residual=residual,
        entropy=entropy,
        repair_count=xp.sum(bad).astype(xp.int64),
    )


def kl_divergence(p: Any, q: Any, *, xp: Any, eps: float = 1e-12) -> Any:
    p = xp.asarray(p)
    q = xp.asarray(q)
    pp = xp.maximum(p, eps)
    qq = xp.maximum(q, eps)
    return xp.sum(pp * xp.log(pp / qq), axis=-1)
