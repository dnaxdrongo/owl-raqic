"""Plan and certify a stratified multi-seed counterfactual modeling corpus."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from owl.cadc.artifacts import atomic_json, sha256_file
from owl.cadc.config import CADCPhase4Config
from owl.cadc.schema import EXPECTED_PHASE3_SOURCE_SHA256, SplitRole, stable_id
from owl.cadc.splits import seed_role_map
from owl.experiments.controller import _release_hash

CONTEXT_FAMILY_OVERRIDES: dict[str, dict[str, Any]] = {
    "balanced": {},
    "resource_scarce": {
        "initialization": {
            "initial_resource_mean": 0.30,
            "background_food": 0.005,
            "food_patch_count": 3,
            "food_patch_intensity": 0.35,
        }
    },
    "toxin_stress": {
        "initialization": {
            "toxin_patch_count": 5,
            "toxin_patch_intensity": 0.65,
            "toxin_patch_radius": 3,
        },
        "resources": {"toxin_decay": 0.004},
    },
    "social_dense": {
        "initialization": {"population_density": 0.55},
        "communication": {"base_emit_cost": 0.0025},
    },
    "boundary_stress": {
        "world": {"boundary_mode": "reflective"},
        "initialization": {
            "initial_boundary_mean": 0.45,
            "obstacle_density": 0.08,
        },
        "resources": {"movement_cost": 0.02},
    },
}


@dataclass(frozen=True)
class CorpusUnit:
    """One pre-registered seed, context, tick, repeat, and horizon run unit."""
    unit_id: str
    seed: int
    split_role: SplitRole
    context_family: str
    source_tick: int
    repeats: int
    horizons: tuple[int, ...]
    maximum_source_decisions: int
    derived_config_path: str
    output_path: str


@dataclass(frozen=True)
class CorpusPlan:
    """Store an immutable multi-seed modeling plan with sealed confirmatory seeds."""
    plan_id: str
    phase3_source_sha256: str
    base_config_sha256: str
    config_sha256: str
    units: tuple[CorpusUnit, ...]
    sealed_phase5_seeds: tuple[int, ...]
    sealed_phase6_seeds: tuple[int, ...]

    def manifest(self) -> dict[str, Any]:
        """Return the fully materialized plan and its confirmatory evaluation locks."""
        return {
            "schema_version": "owl.cadc.phase4-corpus-plan.v1",
            "plan_id": self.plan_id,
            "phase3_source_sha256": self.phase3_source_sha256,
            "base_config_sha256": self.base_config_sha256,
            "config_sha256": self.config_sha256,
            "units": [asdict(value) for value in self.units],
            "sealed_phase5_seeds": list(self.sealed_phase5_seeds),
            "sealed_phase6_seeds": list(self.sealed_phase6_seeds),
            "phase5_locked": True,
            "phase6_locked": True,
        }


def verify_immutable_phase3_engine(root: str | Path) -> str:
    """Require a separate unchanged checkout of the certified counterfactual engine."""
    engine = Path(root).resolve()
    if not (engine / "scripts/run_cadc_phase3_acceptance.py").is_file():
        raise FileNotFoundError("immutable Phase 3 engine runner is missing")
    actual = _release_hash(engine)
    if actual != EXPECTED_PHASE3_SOURCE_SHA256:
        raise RuntimeError(
            f"immutable Phase 3 engine scope mismatch: expected "
            f"{EXPECTED_PHASE3_SOURCE_SHA256}, found {actual}"
        )
    return actual


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def build_corpus_plan(
    config: CADCPhase4Config,
    *,
    output_root: str | Path,
) -> CorpusPlan:
    """Freeze stratified units without opening or materializing sealed seeds."""
    engine = Path(config.phase3_input.immutable_engine_root).resolve()
    source_sha256 = verify_immutable_phase3_engine(engine)
    base_path = Path(config.phase3_input.base_phase3_config)
    if not base_path.is_absolute():
        base_path = engine / base_path
    base_payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base_payload, dict):
        raise TypeError("Phase 3 base configuration must be a mapping")
    role_map = seed_role_map(
        development=config.corpus.development_seeds,
        validation=config.corpus.validation_seeds,
        calibration=config.corpus.calibration_seeds,
        phase5=config.corpus.reserved_phase5_seeds,
        phase6=config.corpus.reserved_phase6_seeds,
    )
    output = Path(output_root).resolve()
    configs = output / "derived_configs"
    runs = output / "runs"
    configs.mkdir(parents=True, exist_ok=True)
    units: list[CorpusUnit] = []
    visible_seeds = (
        *config.corpus.development_seeds,
        *config.corpus.validation_seeds,
        *config.corpus.calibration_seeds,
    )
    repeats = (
        config.corpus.repeat_pilot[0]
        if config.corpus.repeat_policy == "pilot_derived"
        else int(config.corpus.repeat_policy)
    )
    all_horizons = set(config.corpus.horizons)
    for family_values in config.corpus.family_horizons.values():
        all_horizons.update(family_values)
    for seed_index, seed in enumerate(visible_seeds):
        for tick_index, source_tick in enumerate(config.corpus.source_ticks):
            family = config.corpus.context_families[
                (seed_index + tick_index) % len(config.corpus.context_families)
            ]
            if family not in CONTEXT_FAMILY_OVERRIDES:
                raise KeyError(f"unknown context family: {family}")
            identity = stable_id(
                "corpus_unit", config.corpus_digest(), seed, family, source_tick, repeats
            )
            config_path = configs / f"{identity}.yaml"
            run_path = runs / identity
            derived = _deep_merge(base_payload, CONTEXT_FAMILY_OVERRIDES[family])
            world_override: dict[str, Any] = {}
            if config.corpus.world_height is not None:
                if (
                    config.corpus.world_width is None
                    or config.corpus.patch_size is None
                ):
                    raise AssertionError("validated corpus dimensions are incomplete")
                world_override = {
                    "height": int(config.corpus.world_height),
                    "width": int(config.corpus.world_width),
                    "patch_size": int(config.corpus.patch_size),
                }
            derived = _deep_merge(
                derived,
                {
                    "world": {
                        **world_override,
                        "seed": int(seed),
                        "max_steps": int(max(all_horizons) + source_tick),
                    },
                    "counterfactual": {
                        "source_selection_mode": "action_family_stratified",
                        "explicit_source_decision_ids": [],
                        "max_source_ticks": 1,
                        "max_source_decisions": config.corpus.max_source_decisions_per_seed,
                        "repeats": repeats,
                        "horizons": list(config.corpus.horizons),
                        "family_horizons": {
                            name: list(values)
                            for name, values in config.corpus.family_horizons.items()
                        },
                    },
                },
            )
            from owl.core.config import SimulationConfig

            SimulationConfig.model_validate(derived)
            config_path.write_text(
                yaml.safe_dump(derived, sort_keys=True), encoding="utf-8"
            )
            units.append(
                CorpusUnit(
                    unit_id=identity,
                    seed=int(seed),
                    split_role=role_map[int(seed)],
                    context_family=family,
                    source_tick=int(source_tick),
                    repeats=repeats,
                    horizons=config.corpus.horizons,
                    maximum_source_decisions=config.corpus.max_source_decisions_per_seed,
                    derived_config_path=str(config_path),
                    output_path=str(run_path),
                )
            )
    plan_id = stable_id(
        "corpus_plan",
        source_sha256,
        sha256_file(base_path),
        config.corpus_digest(),
        [
            {
                "unit_id": value.unit_id,
                "seed": value.seed,
                "split_role": value.split_role,
                "context_family": value.context_family,
                "source_tick": value.source_tick,
                "repeats": value.repeats,
                "horizons": value.horizons,
                "maximum_source_decisions": value.maximum_source_decisions,
            }
            for value in units
        ],
    )
    return CorpusPlan(
        plan_id,
        source_sha256,
        sha256_file(base_path),
        config.corpus_digest(),
        tuple(units),
        config.corpus.reserved_phase5_seeds,
        config.corpus.reserved_phase6_seeds,
    )


def certify_corpus_inventory(
    plan: CorpusPlan,
    inventories: Sequence[Mapping[str, Any]],
    *,
    minimum_seeds: int,
    minimum_source_decisions: int,
) -> dict[str, Any]:
    """Certify multi-seed support and exact counterfactual receipts before model fitting."""
    failures: list[str] = []
    units = {value.unit_id: value for value in plan.units}
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    seeds: set[int] = set()
    for inventory in inventories:
        unit_id = str(inventory.get("unit_id", ""))
        if unit_id not in units or unit_id in seen:
            failures.append(f"unexpected_or_duplicate_unit:{unit_id}")
            continue
        seen.add(unit_id)
        if inventory.get("phase3_source_sha256") != plan.phase3_source_sha256:
            failures.append(f"source_mismatch:{unit_id}")
        if inventory.get("passed") is not True:
            failures.append(f"unit_failed:{unit_id}")
        failures.extend(validate_corpus_qiskit_evidence(inventory, unit_id=unit_id))
        seeds.add(int(units[unit_id].seed))
        counts["source_decisions"] += int(inventory.get("source_decisions", 0))
        counts["branch_horizons"] += int(inventory.get("branch_horizons", 0))
        counts["candidate_pairs"] += int(inventory.get("candidate_pairs", 0))
    missing = sorted(set(units).difference(seen))
    failures.extend(f"missing_unit:{value}" for value in missing)
    if len(seeds) < minimum_seeds:
        failures.append("insufficient_independent_seeds")
    if counts["source_decisions"] < minimum_source_decisions:
        failures.append("insufficient_source_decisions")
    sealed = set(plan.sealed_phase5_seeds).union(plan.sealed_phase6_seeds)
    if seeds.intersection(sealed):
        failures.append("confirmatory_seed_leakage")
    return {
        "schema_version": "owl.cadc.phase4-corpus-certificate.v1",
        "plan_id": plan.plan_id,
        "corpus_contract_sha256": plan.config_sha256,
        "phase3_source_sha256": plan.phase3_source_sha256,
        "passed": not failures,
        "classification": (
            "PHASE4_DEVELOPMENT_CORPUS_CERTIFIED" if not failures else "FAILED_CLOSED"
        ),
        "failures": failures,
        "unit_count": len(inventories),
        "independent_seed_count": len(seeds),
        "row_counts": dict(counts),
        "sealed_phase5_seeds": list(plan.sealed_phase5_seeds),
        "sealed_phase6_seeds": list(plan.sealed_phase6_seeds),
        "phase5_locked": True,
    }


def validate_corpus_qiskit_evidence(
    inventory: Mapping[str, Any], *, unit_id: str
) -> list[str]:
    """Validate explicit Qiskit execution applicability without GPU conflation."""
    failures: list[str] = []
    qiskit = inventory.get("qiskit_execution")
    exercised = inventory.get("qiskit_exercised")
    runtime_required = inventory.get("qiskit_gpu_runtime_required")
    if not isinstance(qiskit, Mapping):
        return [f"qiskit_evidence_missing:{unit_id}"]
    if not isinstance(exercised, bool):
        failures.append(f"qiskit_applicability_missing:{unit_id}")
        return failures
    if not isinstance(runtime_required, bool):
        failures.append(f"qiskit_runtime_applicability_missing:{unit_id}")
        return failures
    if qiskit.get("exercised") is not exercised:
        failures.append(f"qiskit_applicability_mismatch:{unit_id}")
    if qiskit.get("automatic_execution_fallback") is not False:
        failures.append(f"qiskit_fallback_not_prohibited:{unit_id}")
    if qiskit.get("passed") is not True:
        failures.append(f"qiskit_evidence_failed:{unit_id}")
    if exercised:
        if qiskit.get("mode") in {None, "off"}:
            failures.append(f"qiskit_exercised_mode_invalid:{unit_id}")
        if qiskit.get("evidence_status") != "executed":
            failures.append(f"qiskit_execution_evidence_missing:{unit_id}")
        if runtime_required and qiskit.get("runtime_binding_required") is not True:
            failures.append(f"qiskit_runtime_requirement_mismatch:{unit_id}")
        if runtime_required and qiskit.get("runtime_binding_used") is not True:
            failures.append(f"qiskit_runtime_binding_missing:{unit_id}")
    else:
        if runtime_required:
            failures.append(f"qiskit_runtime_required_when_not_exercised:{unit_id}")
        if qiskit.get("mode") != "off":
            failures.append(f"qiskit_not_exercised_mode_invalid:{unit_id}")
        if qiskit.get("evidence_status") != "not_exercised":
            failures.append(f"qiskit_nonexecution_evidence_missing:{unit_id}")
        if qiskit.get("runtime_binding_required") is not False:
            failures.append(f"qiskit_runtime_requirement_mismatch:{unit_id}")
        if qiskit.get("runtime_binding_used") is not False:
            failures.append(f"qiskit_runtime_binding_claim_without_execution:{unit_id}")
    return failures


def write_corpus_plan(plan: CorpusPlan, path: str | Path) -> None:
    """Atomically persist one canonical corpus plan."""
    atomic_json(path, plan.manifest())
