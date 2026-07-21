from __future__ import annotations

from typing import Literal, cast

from owl.core.actions import Action
from owl.core.config import SimulationConfig
from owl_raqic.config import ActivePlaceConfig, RAQICAlgorithmConfig, RAQICRegisterConfig
from owl_raqic.types import RAQICActionSet


def owl_action_names() -> tuple[str, ...]:
    return tuple(action.name for action in Action)


def build_raqic_action_set() -> RAQICActionSet:
    return RAQICActionSet(names=owl_action_names())


def convert_owl_cfg_to_raqic_cfg(cfg: SimulationConfig) -> RAQICAlgorithmConfig:
    mode_map = {
        "cpu_audit": "cpu_audit",
        "cpu_qiskit": "static",
        "dynamic": "dynamic",
        "deferred": "deferred",
        "hybrid": "hybrid",
        "walk": "walk",
    }
    active = ActivePlaceConfig(
        include_infinity=True,
        primes=tuple(cfg.raqic.active_primes),
        prime_weights=dict(cfg.raqic.prime_weights),
    )
    registers = RAQICRegisterConfig(
        n_scale=3,
        n_places=1 + len(cfg.raqic.active_primes),
        n_features=11,
        n_actions=len(Action),
        n_readouts=2,
        use_action_mask=True,
    )
    return RAQICAlgorithmConfig(
        mode=cast(
            Literal["static", "dynamic", "deferred", "hybrid", "walk", "cpu_audit"],
            mode_map.get(cfg.raqic.mode, "cpu_audit"),
        ),
        rounds=int(cfg.raqic.rounds_per_tick),
        shots=int(cfg.raqic.shots),
        seed=int(cfg.world.seed),
        beta_intention=float(cfg.raqic.beta_intention),
        epsilon_adelic=float(cfg.raqic.epsilon_adelic),
        backend_policy="cpu_audit" if cfg.raqic.mode == "cpu_audit" else "auto",
        action_temperature=float(cfg.raqic.action_temperature),
        phase_mode=cast(
            Literal["scalar_reference", "canonical_device"],
            str(getattr(cfg.raqic, "full_gpu_phase_mode", "scalar_reference")),
        ),
        active_places=active,
        registers=registers,
    )
