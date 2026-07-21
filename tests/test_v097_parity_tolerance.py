from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from owl.gpu.shadow_audit import CPUShadowAuditor


def _state(probabilities: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(raqic_probabilities=np.asarray(probabilities, dtype=np.float64))


def _auditor() -> CPUShadowAuditor:
    return CPUShadowAuditor(
        SimpleNamespace(raqic=SimpleNamespace(full_gpu_shadow_reference="scientific_cpu")),
        ticks=(1,),
        tolerance=1e-8,
        strict=False,
    )


def test_h200_probability_residual_uses_float32_lineage_relative_floor() -> None:
    left = _state(np.array([[[0.1303575898986193]]], dtype=np.float64))
    right = _state(np.array([[[0.1303576001976034]]], dtype=np.float64))

    parity = _auditor().compare(left, right, tick=1)

    assert parity.passed
    assert parity.field_residuals["raqic_probabilities"] > 1e-8
    assert parity.field_absolute_tolerances["raqic_probabilities"] == 1e-8
    assert parity.field_relative_tolerances["raqic_probabilities"] == float(
        np.finfo(np.float32).eps
    )
    assert parity.field_residual_ratios["raqic_probabilities"] < 1.0


def test_relative_floor_does_not_mask_near_zero_probability_error() -> None:
    left = _state(np.array([[[0.0]]], dtype=np.float64))
    right = _state(np.array([[[2.0e-8]]], dtype=np.float64))

    parity = _auditor().compare(left, right, tick=1)

    assert not parity.passed
    assert parity.field_residual_ratios["raqic_probabilities"] > 1.0


def test_relative_floor_rejects_material_probability_drift() -> None:
    left = _state(np.array([[[0.13]]], dtype=np.float64))
    right = _state(np.array([[[0.130001]]], dtype=np.float64))

    parity = _auditor().compare(left, right, tick=1)

    assert not parity.passed
    assert parity.field_residuals["raqic_probabilities"] > 1e-7


def test_non_probability_fields_remain_absolute_only() -> None:
    left = SimpleNamespace(memory=np.array([[0.5]], dtype=np.float64))
    right = SimpleNamespace(memory=np.array([[0.5 + 1.1e-8]], dtype=np.float64))

    parity = _auditor().compare(left, right, tick=1)

    assert not parity.passed
    assert parity.field_relative_tolerances["memory"] == 0.0


def test_utility_tick_four_evidence_residual_is_within_relative_contract() -> None:
    right_value = 0.04545455
    left_value = right_value + 1.1010495321039926e-8
    parity = _auditor().compare(
        _state(np.array([[[left_value]]], dtype=np.float64)),
        _state(np.array([[[right_value]]], dtype=np.float64)),
        tick=4,
    )

    assert parity.passed
    assert parity.field_limits_at_worst["raqic_probabilities"] > 1.1e-8
    assert parity.left_values_at_worst["raqic_probabilities"] == left_value
    assert parity.right_values_at_worst["raqic_probabilities"] == right_value


def test_probability_tolerance_never_overrides_exact_readout_contract() -> None:
    left = SimpleNamespace(
        raqic_probabilities=np.array([[[0.13]]], dtype=np.float64),
        raqic_readout=np.array([[1]], dtype=np.int16),
    )
    right = SimpleNamespace(
        raqic_probabilities=np.array([[[0.13 + 1e-8]]], dtype=np.float64),
        raqic_readout=np.array([[2]], dtype=np.int16),
    )

    parity = _auditor().compare(left, right, tick=1)

    assert not parity.passed
    assert parity.exact_event_matches["raqic_readout"] is False
