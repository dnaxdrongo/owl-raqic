from __future__ import annotations

from typing import Any

from owl.gpu.array_write import write_array
from owl.gpu.stencil import categorical_neighbor_agreement


def _normalize(p: Any, xp: Any, eps: float) -> Any:
    q = xp.clip(p, 0, None)
    return q / xp.maximum(xp.sum(q, axis=-1, keepdims=True), eps)


def entropy_normalized_gpu(probability: float, xp: Any, epsilon: float = 1e-8) -> Any:
    p = _normalize(probability, xp, epsilon)
    k = p.shape[-1]
    if k == 1:
        return xp.zeros(p.shape[:-1], dtype=xp.float32)
    return xp.clip(-xp.sum(p * xp.log(p + epsilon), axis=-1) / xp.log(float(k)), 0, 1).astype(
        xp.float32
    )


def compute_conflict_gpu(ds: Any, cfg: Any) -> Any:
    xp = ds.xp
    eps = float(cfg.actions.epsilon)
    bias = ds.arrays.get(
        "_parent_bias_for_conflict", ds.arrays.get("pre_parent_bias", xp.zeros_like(ds.possibility))
    )
    strength = xp.sum(xp.abs(bias), axis=-1)
    shifted = bias - xp.min(bias, axis=-1, keepdims=True)
    parent = _normalize(shifted, xp, eps)
    child = _normalize(ds.possibility, xp, eps)
    parent_conf = 0.5 * xp.sum(xp.abs(parent - child), axis=-1) * (strength / (1 + strength))
    parent_conf = xp.where(strength <= eps, 0.0, parent_conf)
    same = categorical_neighbor_agreement(ds.readout, xp, "toroidal")
    disagreement = 1 - same / 8.0
    disagreement = xp.where(ds.health > 0, disagreement, 0.0)
    sig = ds.signal_reception
    n = sig.shape[-1]

    def ch(i: int) -> Any:
        return sig[..., i] if i < n else xp.zeros_like(ds.health)

    signal = xp.clip(ch(0) * ch(1) + ch(2) * ch(3) + 0.5 * ch(4) * ch(2), 0, 1)
    stress = (
        (1 - xp.clip(ds.health, 0, 1)) + (1 - xp.clip(ds.boundary, 0, 1)) + xp.clip(ds.toxin, 0, 1)
    ) / 3.0
    out = xp.clip(0.40 * parent_conf + 0.25 * disagreement + 0.20 * signal + 0.15 * stress, 0, 1)
    return xp.where(ds.health > 0, out, 0.0).astype(xp.float32)


def update_integration_gpu(ds: Any, cfg: Any) -> None:
    xp = ds.xp
    ic = cfg.integration
    entropy = entropy_normalized_gpu(ds.possibility, xp, float(cfg.actions.epsilon))
    flex = xp.exp(-((entropy - float(ic.entropy_target)) ** 2) / (2 * float(ic.entropy_sigma) ** 2))
    conflict = compute_conflict_gpu(ds, cfg)
    sync = ds.arrays.get("_synchrony_current", xp.zeros_like(ds.health))
    coh = ds.arrays.get("_coherence_current", xp.zeros_like(ds.health))
    cross = ds.arrays.get("_cross_scale_current", xp.zeros_like(ds.health))
    z = (
        float(ic.weight_memory) * xp.clip(ds.memory, 0, 1)
        + float(ic.weight_flexibility) * xp.clip(flex, 0, 1)
        + float(ic.weight_synchrony) * xp.clip(sync, 0, 1)
        + float(ic.weight_coherence) * xp.clip(coh, 0, 1)
        + float(ic.weight_cross_scale) * xp.clip(cross, 0, 1)
        + float(ic.weight_resource) * xp.clip(ds.resource, 0, 1)
        + float(ic.weight_boundary) * xp.clip(ds.boundary, 0, 1)
        - float(ic.weight_conflict) * xp.clip(conflict, 0, 1)
        - xp.clip(ds.threshold, 0, 1)
    )
    updated = 1 / (1 + xp.exp(-z))
    write_array(
        ds, "integration", xp.where(ds.health > 0, xp.clip(updated, 0, 1), 0.0).astype(xp.float32)
    )
