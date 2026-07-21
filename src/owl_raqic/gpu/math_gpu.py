from __future__ import annotations

from typing import Any, cast

import numpy as np

from owl_raqic.adelic.phases import adelic_character_phase_proxy
from owl_raqic.algorithms.feature_pipeline import FEATURE_ORDER, score_weights_for_actions
from owl_raqic.gpu.actualization_extensions import (
    ActualizationExtensionConfig,
    apply_actualization_extensions,
)
from owl_raqic.gpu.phase_kernels import build_phase_coefficients, compute_canonical_phases
from owl_raqic.gpu.random import deterministic_uniforms_to_backend
from owl_raqic.math.action_graph import action_family_edges

_WEIGHT_CACHE: dict[tuple[Any, ...], Any] = {}
_PHASE_TABLE_CACHE: dict[tuple[Any, ...], Any] = {}


def dtype_for_precision(precision: str) -> Any:
    return np.float32 if precision in ("balanced32", "fast32") else np.float64


def to_numpy(x: Any) -> np.ndarray:
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cast(np.ndarray, cp.asnumpy(x))
    except Exception:
        pass
    return np.asarray(x)


def projected_features_dense(
    features: Any,
    feature_bins: Any,
    primes: tuple[int, ...],
    prime_weights: dict[int, float],
    epsilon_adelic: float,
    xp: Any = np,
    dtype: Any = np.float64,
    max_valuation_steps: int = 32,
) -> Any:
    """Vectorized finite active-place projection without host synchronization.

    The prior implementation used ``while bool(cp.any(...))``, which forced a
    device synchronization once per p-adic valuation step.  A fixed bounded loop
    is exact for the int32 feature-code domain and graph-capturable.
    """
    x = xp.asarray(features, dtype=dtype)
    codes = xp.asarray(feature_bins, dtype=xp.int64)
    dist = xp.abs(x)
    for p in primes:
        c = xp.abs(codes).astype(xp.int64)
        zero = c == 0
        tmp = c.copy()
        v = xp.zeros_like(tmp, dtype=xp.int32)
        for _ in range(int(max_valuation_steps)):
            active = (~zero) & ((tmp % int(p)) == 0)
            v = xp.where(active, v + 1, v)
            tmp = xp.where(active, tmp // int(p), tmp)
        padic_abs = xp.where(zero, 0.0, xp.power(float(p), -v.astype(dtype)))
        bounded = xp.where(zero, 0.0, padic_abs / (1.0 + padic_abs))
        dist = dist + float(prime_weights.get(p, 0.0)) * bounded
    kernel = xp.exp(-xp.maximum(dist, 0.0))
    return x + float(epsilon_adelic) * kernel


def _score_weights(action_names: tuple[str, ...], xp: Any, dtype: Any) -> Any:
    xp_name = getattr(xp, "__name__", str(xp))
    device = -1
    if xp_name == "cupy":
        device = int(xp.cuda.Device().id)
    key = (action_names, np.dtype(dtype).str, xp_name, device)
    value = _WEIGHT_CACHE.get(key)
    if value is None:
        weights_np = score_weights_for_actions(len(action_names), action_names).astype(
            dtype, copy=False
        )
        value = xp.asarray(weights_np, dtype=dtype)
        _WEIGHT_CACHE[key] = value
    return value


def compute_scores_dense(
    features: Any,
    feature_bins: Any,
    action_names: tuple[str, ...],
    primes: tuple[int, ...],
    prime_weights: dict[int, float],
    epsilon_adelic: float = 1.0,
    xp: Any = np,
    dtype: Any = np.float64,
) -> Any:
    feats = projected_features_dense(
        features, feature_bins, primes, prime_weights, epsilon_adelic, xp=xp, dtype=dtype
    )
    return feats @ _score_weights(action_names, xp, dtype).T


def _phase_table(
    feature_names: Any, action_count: int, primes: Any, modulus_power: Any, xp: Any
) -> Any:
    xp_name = getattr(xp, "__name__", str(xp))
    device = int(xp.cuda.Device().id) if xp_name == "cupy" else -1
    key = (
        tuple(feature_names),
        int(action_count),
        tuple(primes),
        int(modulus_power),
        xp_name,
        device,
    )
    table = _PHASE_TABLE_CACHE.get(key)
    if table is None:
        table = build_phase_coefficients(
            tuple(feature_names),
            int(action_count),
            tuple(primes),
            xp=xp,
            modulus_power=max(1, min(int(modulus_power), 4)),
        )
        _PHASE_TABLE_CACHE[key] = table
    return table


def compute_phases_dense(
    feature_bins: Any,
    action_names: tuple[str, ...],
    primes: tuple[int, ...],
    modulus_power: int = 8,
    xp: Any = np,
    dtype: Any = np.float64,
    *,
    phase_mode: str = "scalar_reference",
    epsilon_adelic: float = 1.0,
    feature_names: tuple[str, ...] = tuple(FEATURE_ORDER),
) -> Any:
    """Compute RAQIC phase values for the selected mode.

    ``canonical_device`` is fully vectorized and device-native.  The compatibility
    scalar helper remains an explicit audit oracle, not a hidden production path.
    """
    if phase_mode == "canonical_device":
        table = _phase_table(feature_names, len(action_names), primes, modulus_power, xp)
        return compute_canonical_phases(
            feature_bins, table, xp=xp, epsilon_adelic=epsilon_adelic
        ).astype(dtype, copy=False)
    if phase_mode != "scalar_reference":
        raise ValueError(f"unknown phase_mode: {phase_mode}")

    bins_np = to_numpy(feature_bins).astype(np.int64, copy=False)
    n = bins_np.shape[0]
    a_count = len(action_names)
    phase_index = FEATURE_ORDER.index("phase")
    phase_code = np.maximum(bins_np[:, phase_index], 1)
    out = np.zeros((n, a_count), dtype=float)
    for a in range(a_count):
        num = ((phase_code + a + 1) % 251) + 1
        den = int((a + 2) * 257)
        for i in range(n):
            out[i, a] = adelic_character_phase_proxy(
                int(num[i]),
                den,
                primes,
                modulus_power=modulus_power,
                diagonal_test=False,
            )
    return xp.asarray(out, dtype=dtype)


def repair_authority_mask(mask: Any, rest_index: int = 0, xp: Any = np) -> Any:
    m = xp.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("authority mask must be rank-2 [N,A]")
    any_legal = xp.any(m, axis=1)
    rest_only = xp.zeros_like(m, dtype=bool)
    rest_only[:, int(rest_index)] = True
    return xp.where((~any_legal)[:, None], rest_only, m)


def normalize_intention_dense(parent_intention: Any, xp: Any = np, dtype: Any = np.float64) -> Any:
    intention = xp.asarray(parent_intention, dtype=dtype)
    intention = xp.where(xp.isfinite(intention), intention, 0.0)
    intention = xp.maximum(intention, 0.0)
    sums = xp.sum(intention, axis=1, keepdims=True)
    n_actions = intention.shape[1]
    return xp.where(
        sums > 0,
        intention / xp.maximum(sums, xp.asarray(1e-300, dtype=dtype)),
        xp.ones_like(intention) / float(n_actions),
    )


def masked_softmax(
    scores: Any, mask: Any, temperature: float = 1.0, xp: Any = np, dtype: Any = np.float64
) -> Any:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = xp.asarray(scores, dtype=dtype) / float(temperature)
    m = repair_authority_mask(mask, xp=xp)
    z = xp.where(m, z, -xp.inf)
    row_max = xp.max(z, axis=1, keepdims=True)
    shifted = xp.where(m, z - row_max, -xp.inf)
    e = xp.where(m, xp.exp(shifted), 0.0)
    denom = xp.sum(e, axis=1, keepdims=True)
    probs = e / xp.maximum(denom, xp.asarray(1e-300, dtype=dtype))
    probs = xp.where(m, probs, 0.0)
    sums = xp.sum(probs, axis=1, keepdims=True)
    return probs / xp.maximum(sums, xp.asarray(1e-300, dtype=dtype))


def apply_intention_bias(
    scores: Any, parent_intention: Any, beta_intention: float, xp: Any = np, dtype: Any = np.float64
) -> Any:
    intention = normalize_intention_dense(parent_intention, xp=xp, dtype=dtype)
    return xp.asarray(scores, dtype=dtype) + float(beta_intention) * intention


def amplitudes_from_probabilities(
    probabilities: Any, phases: Any | None = None, xp: Any = np, complex_dtype: Any | None = None
) -> Any:
    p = xp.asarray(probabilities)
    if complex_dtype is None:
        complex_dtype = xp.complex64 if p.dtype == xp.float32 else xp.complex128
    if phases is None:
        phases = xp.zeros_like(p)
    amp = xp.sqrt(xp.maximum(p, 0.0)) * xp.exp(1j * xp.asarray(phases))
    return amp.astype(complex_dtype, copy=False)


def sample_categorical(probabilities: Any, uniforms: Any, xp: Any = np) -> Any:
    p = xp.asarray(probabilities)
    u = xp.asarray(uniforms, dtype=p.dtype).reshape((-1, 1))
    cdf = xp.cumsum(p, axis=1)
    cdf[:, -1] = 1.0
    # Preserve the established baseline tie rule (first cdf >= u).
    return xp.argmax(cdf >= u, axis=1).astype(xp.int32)


def check_probability_normalization(probabilities: Any, xp: Any = np) -> Any:
    p = xp.asarray(probabilities)
    sums = xp.sum(p, axis=1)
    return {
        "max_row_sum_error": float(to_numpy(xp.max(xp.abs(sums - 1.0))) if p.size else 0.0),
        "min_probability": float(to_numpy(xp.min(p)) if p.size else 0.0),
        "max_probability": float(to_numpy(xp.max(p)) if p.size else 0.0),
    }


def decide_dense(
    batch: Any,
    *,
    seed: int,
    beta_intention: float,
    temperature: float,
    epsilon_adelic: float,
    prime_weights: dict[int, float],
    modulus_power: int = 8,
    precision: str = "audit64",
    xp: Any = np,
    phase_mode: str = "scalar_reference",
    compute_phase: bool = True,
    actualization_config: ActualizationExtensionConfig | None = None,
    return_extension_evidence: bool = False,
) -> Any:
    """Evaluate the dense RAQIC law and optional actualization extensions.

    The baseline statements intentionally retain their  order. When
    every coupling is zero, this branch is used without evaluating extension
    mathematics, preserving the certified baseline oracle.
    """
    dtype = dtype_for_precision(precision)
    features = xp.asarray(batch.features, dtype=dtype)
    bins = xp.asarray(batch.feature_bins, dtype=xp.int32)
    mask = xp.asarray(batch.authority_mask, dtype=bool)
    parent = xp.asarray(batch.parent_intention, dtype=dtype)
    scores = compute_scores_dense(
        features,
        bins,
        batch.action_names,
        batch.active_primes,
        prime_weights,
        epsilon_adelic=epsilon_adelic,
        xp=xp,
        dtype=dtype,
    )
    if compute_phase:
        phases = compute_phases_dense(
            bins,
            batch.action_names,
            batch.active_primes,
            modulus_power=modulus_power,
            xp=xp,
            dtype=dtype,
            phase_mode=phase_mode,
            epsilon_adelic=epsilon_adelic,
            feature_names=batch.feature_names,
        )
    else:
        phases = xp.zeros_like(scores)

    biased = apply_intention_bias(scores, parent, beta_intention, xp=xp, dtype=dtype)
    baseline_probs = masked_softmax(biased, mask, temperature=temperature, xp=xp, dtype=dtype)
    uniforms = deterministic_uniforms_to_backend(seed, batch.tick, batch.ow_id, xp)
    baseline_readout = sample_categorical(baseline_probs, uniforms, xp=xp)

    config = actualization_config or ActualizationExtensionConfig()
    extension_enabled = bool(config.enabled)
    evidence: dict[str, Any] = {
        "variant": config.variant,
        "extension_enabled": extension_enabled,
        "shadow_only": bool(config.shadow_only),
        "pre_mixer_probabilities": baseline_probs,
        "utility_innovation": xp.zeros_like(scores),
        "phase_alignment": xp.zeros_like(scores),
        "resonant_parent_intention": normalize_intention_dense(parent, xp=xp, dtype=dtype),
        "interference_delta_l1": xp.zeros((scores.shape[0],), dtype=dtype),
        "policy_kl": xp.zeros((scores.shape[0],), dtype=dtype),
        "utility_projection_fraction": xp.zeros((scores.shape[0],), dtype=dtype),
        "utility_score_cosine": xp.zeros((scores.shape[0],), dtype=dtype),
        "utility_orthogonality_residual": xp.zeros((scores.shape[0],), dtype=dtype),
        "utility_innovation_norm": xp.zeros((scores.shape[0],), dtype=dtype),
        "interference_norm_error": xp.zeros((scores.shape[0],), dtype=dtype),
        "interference_illegal_mass": xp.zeros((scores.shape[0],), dtype=dtype),
        "shadow_probabilities": None,
        "shadow_readout": None,
    }
    probs = baseline_probs
    readout = baseline_readout
    if extension_enabled:
        extension = apply_actualization_extensions(
            scores,
            phases,
            mask,
            parent,
            getattr(batch, "action_utilities", None),
            getattr(batch, "parent_action_phase", None),
            getattr(batch, "parent_action_coherence", None),
            beta_intention=beta_intention,
            temperature=temperature,
            config=config,
            edges=action_family_edges(tuple(batch.action_names)),
            amplitude_output=getattr(batch, "interference_amplitude_output", None),
            pair_left_scratch=getattr(batch, "interference_left_scratch", None),
            pair_right_scratch=getattr(batch, "interference_right_scratch", None),
            xp=xp,
            dtype=dtype,
        )
        extension_readout = sample_categorical(extension.probabilities, uniforms, xp=xp)
        evidence.update(
            {
                "pre_mixer_probabilities": extension.pre_mixer_probabilities,
                "utility_innovation": extension.utility_innovation,
                "phase_alignment": extension.phase_alignment,
                "resonant_parent_intention": extension.resonant_parent_intention,
                "interference_delta_l1": extension.diagnostics["interference_delta_l1"],
                "policy_kl": extension.diagnostics["policy_kl"],
                "utility_projection_fraction": extension.diagnostics["utility"].get(
                    "projection_fraction", xp.zeros((scores.shape[0],), dtype=dtype)
                ),
                "utility_score_cosine": extension.diagnostics["utility"].get(
                    "score_utility_cosine", xp.zeros((scores.shape[0],), dtype=dtype)
                ),
                "utility_orthogonality_residual": extension.diagnostics["utility"].get(
                    "orthogonality_residual", xp.zeros((scores.shape[0],), dtype=dtype)
                ),
                "utility_innovation_norm": extension.diagnostics["utility"].get(
                    "innovation_norm", xp.zeros((scores.shape[0],), dtype=dtype)
                ),
                "interference_norm_error": extension.diagnostics["interference_norm_error"],
                "interference_illegal_mass": extension.diagnostics["interference_illegal_mass"],
            }
        )
        if config.shadow_only:
            evidence["shadow_probabilities"] = extension.probabilities
            evidence["shadow_readout"] = extension_readout
        else:
            probs = extension.probabilities
            readout = extension_readout

    confidence = xp.max(probs, axis=1)
    base = (scores, phases, probs, readout, confidence)
    return (*base, evidence) if return_extension_evidence else base
