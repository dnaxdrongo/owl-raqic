from __future__ import annotations

import pytest

from owl.core.config import RAQICConfig


def test_stable_baseline_rejects_nonzero_coupling() -> None:
    with pytest.raises(ValueError, match="stable_baseline"):
        RAQICConfig(utility_coupling=0.1)


def test_phase_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        RAQICConfig(
            actualization_variant="fractal_resonance",
            utility_coupling=0.1,
            phase_resonance_coupling=0.1,
            phase_resonance_patch_weight=0.8,
            phase_resonance_global_weight=0.3,
        )


def test_phase_extension_rejects_skip_policy() -> None:
    with pytest.raises(ValueError, match="requires RAQIC phase"):
        RAQICConfig(
            actualization_variant="phase_interference",
            utility_coupling=0.1,
            interference_mixer_strength=0.1,
            full_gpu_phase_policy="skip",
        )
