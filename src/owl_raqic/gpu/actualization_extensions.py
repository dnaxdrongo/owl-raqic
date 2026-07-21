from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActualizationExtensionConfig:
    variant: str = "stable_baseline"
    utility_coupling: float = 0.0
    utility_projection_epsilon: float = 1e-8
    utility_bound_floor: float = 1.0
    phase_resonance_coupling: float = 0.0
    interference_mixer_strength: float = 0.0
    interference_trotter_steps: int = 1
    shadow_only: bool = False

    @property
    def enabled(self) -> bool:
        return self.variant != "stable_baseline" and any(
            value != 0.0
            for value in (
                self.utility_coupling,
                self.phase_resonance_coupling,
                self.interference_mixer_strength,
            )
        )


@dataclass(frozen=True)
class ActualizationExtensionResult:
    probabilities: Any
    pre_mixer_probabilities: Any
    utility_innovation: Any
    resonant_parent_intention: Any
    phase_alignment: Any
    mixed_amplitudes: Any | None
    diagnostics: dict[str, Any]


def masked_mean(values: Any, mask: Any, *, xp: Any, dtype: Any) -> Any:
    x = xp.asarray(values, dtype=dtype)
    legal = xp.asarray(mask, dtype=bool)
    count = xp.sum(legal, axis=1, keepdims=True, dtype=xp.int64)
    total = xp.sum(xp.where(legal, x, 0.0), axis=1, keepdims=True, dtype=dtype)
    return total / xp.maximum(count.astype(dtype), xp.asarray(1.0, dtype=dtype))


def masked_standardize(
    values: Any,
    mask: Any,
    *,
    epsilon: float,
    xp: Any,
    dtype: Any,
) -> Any:
    x = xp.asarray(values, dtype=dtype)
    x = xp.where(xp.isfinite(x), x, 0.0)
    legal = xp.asarray(mask, dtype=bool)
    mean = masked_mean(x, legal, xp=xp, dtype=dtype)
    centered = xp.where(legal, x - mean, 0.0)
    count = xp.sum(legal, axis=1, keepdims=True, dtype=xp.int64).astype(dtype)
    mean_square = xp.sum(centered * centered, axis=1, keepdims=True, dtype=dtype)
    mean_square = mean_square / xp.maximum(count, xp.asarray(1.0, dtype=dtype))
    rms = xp.sqrt(xp.maximum(mean_square, xp.asarray(0.0, dtype=dtype)))
    standardized = centered / xp.maximum(rms, xp.asarray(float(epsilon), dtype=dtype))
    return xp.where(legal & (rms > float(epsilon)), standardized, 0.0)


def orthogonal_utility_innovation(
    scores: Any,
    utilities: Any,
    authority_mask: Any,
    *,
    epsilon: float,
    bound_floor: float,
    xp: Any,
    dtype: Any,
) -> tuple[Any, dict[str, Any]]:
    """Return bounded utility novelty orthogonal to the legal score direction.

    A row is invalid when fewer than two actions are legal, any legal score or
    utility is non-finite, or the standardized score direction is degenerate.
    Invalid rows receive an exact zero innovation rather than a partially
    repaired signal.
    """
    mask = xp.asarray(authority_mask, dtype=bool)
    score_input = xp.asarray(scores, dtype=dtype)
    utility_input = xp.asarray(utilities, dtype=dtype)
    legal_finite = xp.all(
        (~mask) | (xp.isfinite(score_input) & xp.isfinite(utility_input)),
        axis=1,
        keepdims=True,
    )
    s = masked_standardize(score_input, mask, epsilon=epsilon, xp=xp, dtype=dtype)
    u_z = masked_standardize(utility_input, mask, epsilon=epsilon, xp=xp, dtype=dtype)
    u = xp.where(mask, xp.tanh(u_z), 0.0)

    score_norm2 = xp.sum(s * s, axis=1, keepdims=True, dtype=dtype)
    utility_norm2 = xp.sum(u * u, axis=1, keepdims=True, dtype=dtype)
    cross = xp.sum(s * u, axis=1, keepdims=True, dtype=dtype)
    safe_score_norm2 = xp.maximum(score_norm2, xp.asarray(float(epsilon), dtype=dtype))
    coefficient = cross / safe_score_norm2
    coefficient = xp.where(score_norm2 > float(epsilon), coefficient, 0.0)
    projection = coefficient * s
    residual = xp.where(mask, u - projection, 0.0)
    max_abs = xp.max(xp.abs(residual), axis=1, keepdims=True)
    scale = xp.maximum(xp.asarray(float(bound_floor), dtype=dtype), max_abs)
    innovation = xp.where(mask, residual / scale, 0.0)
    legal_count = xp.sum(mask, axis=1, keepdims=True, dtype=xp.int64)
    valid = legal_finite & (legal_count >= 2) & (score_norm2 > float(epsilon))
    innovation = xp.where(valid, innovation, 0.0)

    projection_norm2 = xp.sum(projection * projection, axis=1, keepdims=True, dtype=dtype)
    innovation_norm = xp.sqrt(xp.maximum(xp.sum(innovation * innovation, axis=1, dtype=dtype), 0.0))
    residual_dot = xp.sum(s * innovation, axis=1, dtype=dtype)
    projection_fraction = xp.where(
        utility_norm2 > float(epsilon),
        projection_norm2 / xp.maximum(utility_norm2, float(epsilon)),
        0.0,
    )[:, 0]
    cosine = xp.where(
        (score_norm2 > float(epsilon)) & (utility_norm2 > float(epsilon)),
        cross
        / xp.maximum(
            xp.sqrt(score_norm2 * utility_norm2),
            xp.asarray(float(epsilon), dtype=dtype),
        ),
        0.0,
    )[:, 0]
    valid_row = valid[:, 0]
    return innovation, {
        "score_norm2": xp.where(valid_row, score_norm2[:, 0], 0.0),
        "utility_norm2": xp.where(valid_row, utility_norm2[:, 0], 0.0),
        "projection_coefficient": xp.where(valid_row, coefficient[:, 0], 0.0),
        "projection_fraction": xp.where(valid_row, projection_fraction, 0.0),
        "score_utility_cosine": xp.where(valid_row, cosine, 0.0),
        "innovation_norm": xp.where(valid_row, innovation_norm, 0.0),
        "orthogonality_residual": xp.where(valid_row, residual_dot, 0.0),
        "legal_count": legal_count[:, 0],
        "input_finite": legal_finite[:, 0],
    }


def _normalize_nonnegative(values: Any, *, xp: Any, dtype: Any) -> Any:
    x = xp.asarray(values, dtype=dtype)
    x = xp.where(xp.isfinite(x), x, 0.0)
    x = xp.maximum(x, 0.0)
    sums = xp.sum(x, axis=1, keepdims=True, dtype=dtype)
    fallback = xp.ones_like(x, dtype=dtype) / float(x.shape[1])
    return xp.where(sums > 0.0, x / xp.maximum(sums, 1e-300), fallback)


def phase_modulated_parent_intention(
    parent_intention: Any,
    local_phase: Any,
    parent_phase: Any,
    parent_coherence: Any,
    coupling: float,
    *,
    xp: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    base = _normalize_nonnegative(parent_intention, xp=xp, dtype=dtype)
    if float(coupling) == 0.0:
        return base, xp.zeros_like(base, dtype=dtype)
    coherence = xp.clip(xp.asarray(parent_coherence, dtype=dtype), 0.0, 1.0)
    delta = xp.asarray(local_phase, dtype=dtype) - xp.asarray(parent_phase, dtype=dtype)
    alignment = coherence * xp.cos(delta)
    alignment = xp.where(xp.isfinite(alignment), alignment, 0.0)
    log_weight = float(coupling) * alignment
    log_weight = log_weight - xp.max(log_weight, axis=1, keepdims=True)
    tilted = base * xp.exp(log_weight)
    normalizer = xp.sum(tilted, axis=1, keepdims=True, dtype=dtype)
    result = tilted / xp.maximum(normalizer, xp.asarray(1e-300, dtype=dtype))
    return result, alignment


def apply_pair_rotation(
    amplitudes: Any,
    legal_mask: Any,
    left: int,
    right: int,
    angle: float,
    *,
    xp: Any,
    left_scratch: Any | None = None,
    right_scratch: Any | None = None,
) -> None:
    if float(angle) == 0.0:
        return
    legal = xp.asarray(legal_mask, dtype=bool)
    pair_legal = legal[:, int(left)] & legal[:, int(right)]
    if left_scratch is None:
        left_old = amplitudes[:, int(left)].copy()
    else:
        left_old = left_scratch
        left_old[...] = amplitudes[:, int(left)]
    if right_scratch is None:
        right_old = amplitudes[:, int(right)].copy()
    else:
        right_old = right_scratch
        right_old[...] = amplitudes[:, int(right)]
    c = xp.asarray(math.cos(float(angle)), dtype=left_old.real.dtype)
    s = xp.asarray(math.sin(float(angle)), dtype=left_old.real.dtype)
    left_new = c * left_old - 1j * s * right_old
    right_new = -1j * s * left_old + c * right_old
    amplitudes[:, int(left)] = xp.where(pair_legal, left_new, left_old)
    amplitudes[:, int(right)] = xp.where(pair_legal, right_new, right_old)


def apply_legal_interference_mixer(
    amplitudes: Any,
    authority_mask: Any,
    edges: tuple[tuple[int, int], ...],
    *,
    strength: float,
    trotter_steps: int,
    xp: Any,
    output: Any | None = None,
    left_scratch: Any | None = None,
    right_scratch: Any | None = None,
) -> Any:
    if int(trotter_steps) < 1:
        raise ValueError("trotter_steps must be at least one")
    amplitude_array = xp.asarray(amplitudes)
    legal_array = xp.asarray(authority_mask, dtype=bool)
    if amplitude_array.ndim != 2:
        raise ValueError("amplitudes must have shape [rows, actions]")
    if tuple(legal_array.shape) != tuple(amplitude_array.shape):
        raise ValueError("authority_mask must match amplitudes")
    for left, right in edges:
        if (
            left == right
            or not (0 <= int(left) < amplitude_array.shape[1])
            or not (0 <= int(right) < amplitude_array.shape[1])
        ):
            raise ValueError("action graph contains an invalid edge")
    if xp is np and not np.all(np.isfinite(amplitude_array)):
        raise ValueError("amplitudes must contain only finite values")
    if float(strength) == 0.0:
        return amplitudes
    if output is None:
        mixed = xp.array(amplitudes, copy=True)
    else:
        mixed = output
        mixed[...] = amplitudes
    rows = int(amplitudes.shape[0])
    for name, scratch in (("left_scratch", left_scratch), ("right_scratch", right_scratch)):
        if scratch is not None and tuple(scratch.shape) != (rows,):
            raise ValueError(f"{name} must have shape {(rows,)}, got {tuple(scratch.shape)}")
    angle = float(strength) / (2.0 * float(trotter_steps))
    sequence = tuple(edges) + tuple(reversed(edges))
    for _ in range(int(trotter_steps)):
        for left, right in sequence:
            apply_pair_rotation(
                mixed,
                authority_mask,
                left,
                right,
                angle,
                xp=xp,
                left_scratch=left_scratch,
                right_scratch=right_scratch,
            )
    return mixed


def apply_actualization_extensions(
    scores: Any,
    phases: Any,
    authority_mask: Any,
    parent_intention: Any,
    utilities: Any | None,
    parent_action_phase: Any | None,
    parent_action_coherence: Any | None,
    *,
    beta_intention: float,
    temperature: float,
    config: ActualizationExtensionConfig,
    edges: tuple[tuple[int, int], ...],
    amplitude_output: Any | None = None,
    pair_left_scratch: Any | None = None,
    pair_right_scratch: Any | None = None,
    xp: Any,
    dtype: Any,
) -> ActualizationExtensionResult:
    from owl_raqic.gpu.math_gpu import masked_softmax, normalize_intention_dense

    mask = xp.asarray(authority_mask, dtype=bool)
    score = xp.asarray(scores, dtype=dtype)
    phase = xp.asarray(phases, dtype=dtype)
    parent_raw = xp.asarray(parent_intention, dtype=dtype)
    if score.ndim != 2:
        raise ValueError("scores must have shape [rows, actions]")
    expected = tuple(score.shape)
    for name, value in (
        ("phases", phase),
        ("authority_mask", mask),
        ("parent_intention", parent_raw),
    ):
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
    if utilities is not None and tuple(xp.asarray(utilities).shape) != expected:
        raise ValueError("utilities must match scores")
    if parent_action_phase is not None and tuple(xp.asarray(parent_action_phase).shape) != expected:
        raise ValueError("parent_action_phase must match scores")
    if (
        parent_action_coherence is not None
        and tuple(xp.asarray(parent_action_coherence).shape) != expected
    ):
        raise ValueError("parent_action_coherence must match scores")
    if float(config.utility_coupling) != 0.0 and utilities is None:
        raise ValueError("utility coupling requires action utilities")
    if float(config.phase_resonance_coupling) != 0.0 and (
        parent_action_phase is None or parent_action_coherence is None
    ):
        raise ValueError("phase resonance requires parent action phase and coherence")
    if xp is np:
        finite_values = [score, phase, parent_raw]
        if utilities is not None:
            finite_values.append(np.asarray(utilities))
        if parent_action_phase is not None:
            finite_values.append(np.asarray(parent_action_phase))
        if parent_action_coherence is not None:
            finite_values.append(np.asarray(parent_action_coherence))
        if not all(np.all(np.isfinite(value)) for value in finite_values):
            raise ValueError("actualization inputs must contain only finite values")
    parent = normalize_intention_dense(parent_raw, xp=xp, dtype=dtype)
    zeros = xp.zeros_like(score, dtype=dtype)
    utility_innovation = zeros
    utility_diag: dict[str, Any] = {}
    if utilities is not None and float(config.utility_coupling) != 0.0:
        utility_innovation, utility_diag = orthogonal_utility_innovation(
            score,
            utilities,
            mask,
            epsilon=float(config.utility_projection_epsilon),
            bound_floor=float(config.utility_bound_floor),
            xp=xp,
            dtype=dtype,
        )

    resonant_parent = parent
    phase_alignment = zeros
    if (
        parent_action_phase is not None
        and parent_action_coherence is not None
        and float(config.phase_resonance_coupling) != 0.0
    ):
        resonant_parent, phase_alignment = phase_modulated_parent_intention(
            parent,
            phase,
            parent_action_phase,
            parent_action_coherence,
            float(config.phase_resonance_coupling),
            xp=xp,
            dtype=dtype,
        )

    logits = (
        score
        + float(config.utility_coupling) * utility_innovation
        + float(beta_intention) * resonant_parent
    )
    pre_mixer = masked_softmax(
        logits,
        mask,
        temperature=float(temperature),
        xp=xp,
        dtype=dtype,
    )
    mixed_amplitudes = None
    probabilities = pre_mixer
    if float(config.interference_mixer_strength) != 0.0:
        complex_dtype = xp.complex64 if dtype == xp.float32 else xp.complex128
        if amplitude_output is None:
            amplitudes = xp.sqrt(xp.maximum(pre_mixer, 0.0)) * xp.exp(1j * phase)
            amplitudes = amplitudes.astype(complex_dtype, copy=False)
            mixer_output = None
        else:
            if tuple(amplitude_output.shape) != expected:
                raise ValueError(
                    f"amplitude_output must have shape {expected}, "
                    f"got {tuple(amplitude_output.shape)}"
                )
            if amplitude_output.dtype != complex_dtype:
                raise ValueError(
                    f"amplitude_output must use {complex_dtype}, got {amplitude_output.dtype}"
                )
            amplitude_output[...] = (
                xp.sqrt(xp.maximum(pre_mixer, 0.0)) * xp.exp(1j * phase)
            ).astype(complex_dtype, copy=False)
            amplitudes = amplitude_output
            # The graph-static buffer is intentionally updated in place. The
            # pair mixer reads each pair into persistent row scratch before
            # writing either endpoint, so input/output aliasing is safe.
            mixer_output = amplitude_output
        mixed_amplitudes = apply_legal_interference_mixer(
            amplitudes,
            mask,
            edges,
            strength=float(config.interference_mixer_strength),
            trotter_steps=int(config.interference_trotter_steps),
            xp=xp,
            output=mixer_output,
            left_scratch=pair_left_scratch,
            right_scratch=pair_right_scratch,
        )
        raw_probabilities = xp.real(mixed_amplitudes * xp.conjugate(mixed_amplitudes)).astype(dtype)
        illegal_mass = xp.sum(xp.where(mask, 0.0, raw_probabilities), axis=1, dtype=dtype)
        probabilities = xp.where(mask, raw_probabilities, 0.0)
        row_sum = xp.sum(probabilities, axis=1, keepdims=True, dtype=dtype)
        norm_error = xp.abs(row_sum[:, 0] - 1.0)
        probabilities = probabilities / xp.maximum(row_sum, 1e-300)
    else:
        norm_error = xp.zeros((score.shape[0],), dtype=dtype)
        illegal_mass = xp.zeros((score.shape[0],), dtype=dtype)
    delta_l1 = xp.sum(xp.abs(probabilities - pre_mixer), axis=1, dtype=dtype)
    kl = xp.sum(
        xp.where(
            probabilities > 0,
            probabilities
            * (xp.log(xp.maximum(probabilities, 1e-300)) - xp.log(xp.maximum(pre_mixer, 1e-300))),
            0.0,
        ),
        axis=1,
        dtype=dtype,
    )
    return ActualizationExtensionResult(
        probabilities=probabilities,
        pre_mixer_probabilities=pre_mixer,
        utility_innovation=utility_innovation,
        resonant_parent_intention=resonant_parent,
        phase_alignment=phase_alignment,
        mixed_amplitudes=mixed_amplitudes,
        diagnostics={
            "utility": utility_diag,
            "interference_delta_l1": delta_l1,
            "policy_kl": kl,
            "interference_norm_error": norm_error,
            "interference_illegal_mass": illegal_mass,
        },
    )


def aggregate_action_phase_context(
    probabilities: Any,
    phases: Any,
    child_weights: Any,
    *,
    patch_confidence: Any | None = None,
    patch_size: int,
    patch_weight: float,
    global_weight: float,
    support_epsilon: float,
    rest_index: int,
    xp: Any,
    dtype: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Aggregate prior-tick probability-weighted action phasors across scales."""
    p = xp.asarray(probabilities, dtype=dtype)
    phi = xp.asarray(phases, dtype=dtype)
    weights = xp.asarray(child_weights, dtype=dtype)
    if p.ndim != 3 or tuple(phi.shape) != tuple(p.shape):
        raise ValueError("probabilities and phases must have shape [height, width, actions]")
    h, w, actions = p.shape
    if tuple(weights.shape) != (h, w):
        raise ValueError(f"child_weights must have shape {(h, w)}")
    size = int(patch_size)
    if size < 1 or h % size or w % size:
        raise ValueError("world shape must be divisible by a positive patch_size")
    if abs(float(patch_weight) + float(global_weight) - 1.0) > 1e-12:
        raise ValueError("patch and global phase weights must sum to one")
    py, px = h // size, w // size
    support = weights[..., None] * xp.maximum(p, 0.0)
    real = support * xp.cos(phi)
    imag = support * xp.sin(phi)

    def patch_sum(value: Any) -> Any:
        shaped = value.reshape(py, size, px, size, actions)
        return xp.sum(shaped, axis=(1, 3), dtype=dtype)

    patch_support = patch_sum(support)
    patch_real = patch_sum(real)
    patch_imag = patch_sum(imag)
    patch_phase = xp.arctan2(patch_imag, patch_real)
    safe_patch_support = xp.maximum(patch_support, float(support_epsilon))
    patch_coherence = xp.sqrt(patch_real * patch_real + patch_imag * patch_imag)
    patch_coherence = patch_coherence / safe_patch_support
    patch_coherence = xp.where(
        patch_support > float(support_epsilon),
        xp.clip(patch_coherence, 0.0, 1.0),
        0.0,
    )

    confidence = (
        xp.ones((py, px), dtype=dtype)
        if patch_confidence is None
        else xp.clip(xp.asarray(patch_confidence, dtype=dtype), 0.0, 1.0)
    )
    if tuple(confidence.shape) != (py, px):
        raise ValueError(
            f"patch_confidence must have shape {(py, px)}, got {tuple(confidence.shape)}"
        )
    global_real = xp.sum(confidence[..., None] * patch_real, axis=(0, 1), dtype=dtype)
    global_imag = xp.sum(confidence[..., None] * patch_imag, axis=(0, 1), dtype=dtype)
    global_support = xp.sum(confidence[..., None] * patch_support, axis=(0, 1), dtype=dtype)
    global_phase = xp.arctan2(global_imag, global_real)
    safe_global_support = xp.maximum(global_support, float(support_epsilon))
    global_coherence = xp.sqrt(global_real * global_real + global_imag * global_imag)
    global_coherence = global_coherence / safe_global_support
    global_coherence = xp.where(
        global_support > float(support_epsilon),
        xp.clip(global_coherence, 0.0, 1.0),
        0.0,
    )

    patch_real_unit = patch_coherence * xp.cos(patch_phase)
    patch_imag_unit = patch_coherence * xp.sin(patch_phase)
    global_real_unit = global_coherence * xp.cos(global_phase)
    global_imag_unit = global_coherence * xp.sin(global_phase)
    mixed_real = (
        float(patch_weight) * patch_real_unit
        + float(global_weight) * global_real_unit[None, None, :]
    )
    mixed_imag = (
        float(patch_weight) * patch_imag_unit
        + float(global_weight) * global_imag_unit[None, None, :]
    )
    parent_phase_patch = xp.arctan2(mixed_imag, mixed_real)
    parent_coherence_patch = xp.clip(
        xp.sqrt(mixed_real * mixed_real + mixed_imag * mixed_imag), 0.0, 1.0
    )
    parent_phase = xp.repeat(xp.repeat(parent_phase_patch, size, axis=0), size, axis=1)
    parent_coherence = xp.repeat(xp.repeat(parent_coherence_patch, size, axis=0), size, axis=1)
    no_support = xp.repeat(
        xp.repeat(parent_coherence_patch <= float(support_epsilon), size, axis=0),
        size,
        axis=1,
    )
    parent_phase = xp.where(no_support, 0.0, parent_phase)
    parent_coherence = xp.where(no_support, 0.0, parent_coherence)
    if not 0 <= int(rest_index) < actions:
        raise ValueError("rest_index is outside the action axis")
    return (
        patch_phase,
        patch_coherence,
        global_phase,
        global_coherence,
        parent_phase,
        parent_coherence,
    )
