from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from owl.core.config import RAQICConfig, load_config
from owl.gpu.field_registry import FIELD_REGISTRY
from owl_raqic.math.action_graph import action_family_edges


def test_all_v096_configs_load_and_baseline_is_exact_recovery_sector() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs").glob("gpu_v096_*.yaml"))
    assert len(paths) == 7
    configs = [load_config(path) for path in paths]
    baseline = next(cfg for cfg in configs if cfg.raqic.actualization_variant == "stable_baseline")
    assert baseline.raqic.utility_coupling == 0.0
    assert baseline.raqic.phase_resonance_coupling == 0.0
    assert baseline.raqic.interference_mixer_strength == 0.0


def test_misconfigured_variants_fail_closed() -> None:
    with pytest.raises(ValueError, match="stable_baseline"):
        RAQICConfig(utility_coupling=0.1)
    with pytest.raises(ValueError, match="sum to one"):
        RAQICConfig(
            actualization_variant="fractal_resonance",
            utility_coupling=0.1,
            phase_resonance_coupling=0.1,
            phase_resonance_patch_weight=0.7,
            phase_resonance_global_weight=0.4,
        )
    with pytest.raises(ValueError, match="phase computation"):
        RAQICConfig(
            actualization_variant="phase_interference",
            utility_coupling=0.1,
            interference_mixer_strength=0.1,
            full_gpu_phase_policy="skip",
        )
    with pytest.raises(ValueError, match="utility_coupling"):
        RAQICConfig(
            actualization_variant="phase_interference",
            interference_mixer_strength=0.1,
        )


def test_action_schema_missing_or_duplicate_name_is_rejected() -> None:
    names = [
        "REST",
        "SENSE",
        "MOVE_N",
        "MOVE_S",
        "MOVE_E",
        "MOVE_W",
        "MOVE_NE",
        "MOVE_NW",
        "MOVE_SE",
        "MOVE_SW",
        "FEED",
        "COMMUNICATE",
        "INHIBIT",
        "INTEGRATE",
        "REPAIR",
        "REPRODUCE",
        "INGEST",
        "EXPEL",
        "SPLIT",
        "MERGE",
        "FLEE",
        "PURSUE",
    ]
    with pytest.raises(ValueError, match="canonical"):
        action_family_edges(tuple(names[:-1]))
    duplicate = names.copy()
    duplicate[-1] = "FLEE"
    with pytest.raises(ValueError, match="canonical"):
        action_family_edges(tuple(duplicate))


def test_extension_fields_are_ephemeral_not_inherited_or_moved() -> None:
    for name in (
        "raqic_parent_action_phase",
        "raqic_parent_action_coherence",
        "raqic_utility_innovation",
        "raqic_phase_alignment",
        "raqic_shadow_probabilities",
    ):
        spec = FIELD_REGISTRY[name]
        assert spec.moves_with_cell is False
        assert spec.copy_on_reproduction is False
        assert spec.clears_on_death is True


def test_qiskit_graph_hash_mismatch_fails_before_circuit_construction() -> None:
    from owl.core.actions import Action
    from owl_raqic.qiskit_backend.circuit_families import build_circuit_family

    action_names = tuple(action.name for action in Action)
    probabilities = np.full(len(action_names), 1.0 / len(action_names))
    phases = np.zeros_like(probabilities)
    with pytest.raises(ValueError, match="graph hash"):
        build_circuit_family(
            "interference",
            probabilities,
            phases,
            action_names=action_names,
            authority_mask=np.ones(len(action_names), dtype=bool),
            mixer_strength=0.1,
            mixer_trotter_steps=1,
            action_graph_hash="deliberately-wrong",
            measure=False,
        )
