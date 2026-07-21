"""Experiment condition presets.

Conditions are pure configuration transforms. They never mutate the caller's
configuration object; each function returns a deep validated copy suitable for a
headless run.
"""

from __future__ import annotations

from owl.core.config import SimulationConfig


def _copy(cfg: SimulationConfig) -> SimulationConfig:
    """Return a deep copy of a simulation config."""
    return cfg.model_copy(deep=True)


def _scale(value: float, factor: float, lower: float = 0.0, upper: float | None = None) -> float:
    """Scale a coefficient and clamp to a configured interval."""
    out = float(value) * float(factor)
    if upper is not None:
        out = min(out, float(upper))
    return max(float(lower), out)


def make_baseline_condition(cfg: SimulationConfig) -> SimulationConfig:
    """Return a passive baseline with OW/fractal coupling disabled.

    The physical layer remains active, but the integration functional and phase
    coupling are reduced so this condition acts as a baseline/null comparison.
    """
    out = _copy(cfg)
    out.integration.weight_memory = 0.0
    out.integration.weight_synchrony = 0.0
    out.integration.weight_coherence = 0.0
    out.integration.weight_cross_scale = 0.0
    out.phase.same_scale_coupling = 0.0
    out.phase.parent_coupling = 0.0
    out.topdown.lambda_action_bias = 0.0
    out.topdown.lambda_threshold = 0.0
    out.communication.trust_lr = 0.0
    return SimulationConfig.model_validate(out.model_dump())


def make_integrated_condition(cfg: SimulationConfig) -> SimulationConfig:
    """Return a high-integration observer-window condition.

    Synchrony, coherence, cross-scale coupling, memory, and weak top-down bias
    are strengthened while remaining bounded by the config schema.
    """
    out = _copy(cfg)
    out.integration.weight_memory = _scale(out.integration.weight_memory, 1.25)
    out.integration.weight_synchrony = _scale(out.integration.weight_synchrony, 1.35)
    out.integration.weight_coherence = _scale(out.integration.weight_coherence, 1.35)
    out.integration.weight_cross_scale = _scale(out.integration.weight_cross_scale, 1.35)
    out.integration.weight_conflict = _scale(out.integration.weight_conflict, 0.85)
    out.phase.same_scale_coupling = _scale(out.phase.same_scale_coupling, 1.7)
    out.phase.parent_coupling = _scale(out.phase.parent_coupling, 1.7)
    out.phase.phase_noise_sigma = _scale(out.phase.phase_noise_sigma, 0.6)
    out.topdown.lambda_action_bias = _scale(
        out.topdown.lambda_action_bias, 1.2, upper=out.topdown.max_parent_control
    )
    out.topdown.lambda_threshold = _scale(
        out.topdown.lambda_threshold, 1.15, upper=out.topdown.max_parent_control
    )
    out.initialization.type_weights = {
        "grazer": 0.48,
        "cooperator": 0.36,
        "explorer": 0.10,
        "scavenger": 0.06,
    }
    return SimulationConfig.model_validate(out.model_dump())


def make_rivalry_condition(cfg: SimulationConfig) -> SimulationConfig:
    """Return a competing-input rivalry condition.

    Rivalry increases threat/danger signaling, predation pressure, and action
    stochasticity while preserving universal communication traits.
    """
    out = _copy(cfg)
    out.actions.stochastic = True
    out.actions.beta = _scale(out.actions.beta, 0.75, lower=0.1)
    out.communication.diffusion = [
        v * (1.25 if i in {1, 2, 4} else 0.95) for i, v in enumerate(out.communication.diffusion)
    ]
    out.communication.decay = [
        v * (0.85 if i in {1, 2, 4} else 1.05) for i, v in enumerate(out.communication.decay)
    ]
    out.integration.weight_conflict = _scale(out.integration.weight_conflict, 1.35)
    out.phase.phase_noise_sigma = _scale(out.phase.phase_noise_sigma, 1.6)
    out.initialization.type_weights = {
        "grazer": 0.42,
        "cooperator": 0.18,
        "proto_carnivore": 0.25,
        "explorer": 0.15,
    }
    out.initialization.toxin_patch_count = max(out.initialization.toxin_patch_count, 2)
    out.initialization.toxin_patch_intensity = max(out.initialization.toxin_patch_intensity, 0.35)
    return SimulationConfig.model_validate(out.model_dump())


def make_fragmented_condition(cfg: SimulationConfig) -> SimulationConfig:
    """Return a fragmented low-coupling/high-noise condition."""
    out = _copy(cfg)
    out.integration.weight_synchrony = _scale(out.integration.weight_synchrony, 0.45)
    out.integration.weight_coherence = _scale(out.integration.weight_coherence, 0.45)
    out.integration.weight_cross_scale = _scale(out.integration.weight_cross_scale, 0.35)
    out.integration.weight_conflict = _scale(out.integration.weight_conflict, 1.30)
    out.phase.same_scale_coupling = _scale(out.phase.same_scale_coupling, 0.35)
    out.phase.parent_coupling = _scale(out.phase.parent_coupling, 0.25)
    out.phase.phase_noise_sigma = _scale(out.phase.phase_noise_sigma, 3.0)
    out.topdown.lambda_action_bias = _scale(out.topdown.lambda_action_bias, 0.35)
    out.topdown.lambda_threshold = _scale(out.topdown.lambda_threshold, 0.50)
    out.initialization.trait_noise_sigma = _scale(out.initialization.trait_noise_sigma, 2.5)
    out.initialization.initial_integration_mean = min(
        out.initialization.initial_integration_mean, 0.05
    )
    return SimulationConfig.model_validate(out.model_dump())


def make_overcoupled_condition(cfg: SimulationConfig) -> SimulationConfig:
    """Return an overcoupled condition with excessive parent/apex constraint."""
    out = _copy(cfg)
    out.integration.weight_synchrony = _scale(out.integration.weight_synchrony, 1.6)
    out.integration.weight_coherence = _scale(out.integration.weight_coherence, 1.6)
    out.integration.weight_cross_scale = _scale(out.integration.weight_cross_scale, 1.8)
    out.phase.same_scale_coupling = _scale(out.phase.same_scale_coupling, 2.5)
    out.phase.parent_coupling = _scale(out.phase.parent_coupling, 2.8)
    out.phase.phase_noise_sigma = _scale(out.phase.phase_noise_sigma, 0.35)
    out.topdown.max_parent_control = max(out.topdown.max_parent_control, 0.35)
    out.topdown.lambda_action_bias = min(
        out.topdown.max_parent_control, max(out.topdown.lambda_action_bias, 0.30)
    )
    out.topdown.lambda_threshold = min(
        out.topdown.max_parent_control, max(out.topdown.lambda_threshold, 0.25)
    )
    out.actions.beta = _scale(out.actions.beta, 1.25)
    return SimulationConfig.model_validate(out.model_dump())


def make_carnivore_condition(cfg: SimulationConfig) -> SimulationConfig:
    """Return a scarcity/predation/carnivory condition."""
    out = _copy(cfg)
    out.predation.enabled = True
    out.predation.min_predation_trait = min(out.predation.min_predation_trait, 0.35)
    out.predation.resource_transfer = max(out.predation.resource_transfer, 0.75)
    out.resources.food_growth = _scale(out.resources.food_growth, 0.45)
    out.resources.food_decay = _scale(out.resources.food_decay, 1.7)
    out.initialization.background_food = min(out.initialization.background_food, 0.01)
    out.initialization.food_patch_count = max(2, out.initialization.food_patch_count // 2)
    out.initialization.food_patch_intensity = min(out.initialization.food_patch_intensity, 0.45)
    out.initialization.type_weights = {
        "grazer": 0.50,
        "proto_carnivore": 0.32,
        "scavenger": 0.12,
        "cooperator": 0.06,
    }
    return SimulationConfig.model_validate(out.model_dump())
