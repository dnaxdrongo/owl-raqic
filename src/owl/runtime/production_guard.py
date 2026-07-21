from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProductionReadiness:
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]
    classification: str = "production"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failures"] = list(self.failures)
        return data


def _passed(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("passed") is True


def _graph_ok(
    plan: Any, execution_metadata: dict[str, Any], *, strict_evidence: bool = False
) -> bool:
    if getattr(plan, "graph_requirement", "") != "full_tick":
        return True
    graph = dict(execution_metadata.get("graph") or {})
    coverage = dict(graph.get("coverage") or {})
    contract = dict(graph.get("capture_contract") or {})
    required = set(coverage.get("required_segments", ()))
    replay = dict(coverage.get("replay_counts", {}))
    segments = dict(graph.get("segments") or {})
    basic = bool(
        coverage.get("full_tick")
        and required
        and all(int(replay.get(name, 0)) > 0 for name in required)
    )
    if not strict_evidence:
        return basic
    return bool(
        basic
        and all(bool((segments.get(name) or {}).get("pointer_stable")) for name in required)
        and all(int((segments.get(name) or {}).get("capture_count", 0)) > 0 for name in required)
        and int(graph.get("invalidation_count", 0)) == 0
        and bool(contract.get("captured_segments_pointer_stable"))
    )


def _qiskit_ok(plan: Any, execution_metadata: dict[str, Any]) -> bool:
    if not getattr(plan.qiskit_policy, "per_ow", False):
        return True
    meta = execution_metadata.get("per_ow_qiskit")
    if not isinstance(meta, dict):
        return False
    expected = int(meta.get("expected_count", -1))
    processed = int(meta.get("processed_count", -2))
    ids_ok = bool(meta.get("all_ow_accounted", expected == processed and expected >= 0))
    families = meta.get("families") or (meta.get("metadata") or {}).get("families") or {}
    strict = bool(getattr(plan.qiskit_policy, "strict_gpu", False))
    if strict:
        evidence = bool((meta.get("metadata") or {}).get("gpu_execution_verified"))
        if isinstance(families, dict) and families:
            evidence = evidence and all(
                bool((value or {}).get("gpu_execution_verified")) for value in families.values()
            )
        if not evidence:
            return False
    return expected >= 0 and processed == expected and ids_ok


def evaluate_production_readiness(
    *,
    plan: Any,
    execution_metadata: dict[str, Any],
    all_configs_valid: bool | None = None,
    config_usage_clean: bool | None = None,
    memory_preflight_passed: bool | None = None,
    certification_compatible: bool | None = None,
    evidence: dict[str, Any] | None = None,
) -> ProductionReadiness:
    """Evaluate exact-plan production readiness.

    When ``evidence`` is supplied, missing evidence is a hard failure.  The
    compatibility booleans remain for modeling tests and cannot produce the 
    production marker script, which always uses typed evidence.
    """
    evidence = dict(evidence or {})
    strict_evidence = bool(evidence)
    memory = dict(execution_metadata.get("memory_preflight") or {})
    graph_ok = _graph_ok(plan, execution_metadata, strict_evidence=strict_evidence)
    qiskit_ok = _qiskit_ok(plan, execution_metadata)
    distributed_ok = True
    if bool(getattr(plan, "multi_gpu", False)):
        distributed_ok = _passed(execution_metadata.get("distributed_certification"))

    base_checks = {
        "all_configs_valid": bool(evidence.get("all_configs_valid", all_configs_valid)),
        "config_behavioral_coverage": bool(
            evidence.get("config_behavioral_coverage", config_usage_clean)
        ),
        "fallback_count_zero": int(execution_metadata.get("fallback_count", 0)) == 0,
        "graph_full_tick_passed_if_required": graph_ok,
        "qiskit_all_families_and_rows_verified_if_required": qiskit_ok,
        "distributed_scientific_and_nccl_passed_if_required": distributed_ok,
        "memory_estimate_and_actual_peak_passed": bool(
            evidence.get(
                "memory_estimate_and_actual_peak_passed",
                memory.get("passed") if memory else memory_preflight_passed,
            )
        ),
        "certification_identity_match": bool(
            evidence.get("environment_identity_match", certification_compatible)
        ),
    }
    if strict_evidence:
        base_checks.update(
            {
                "source_identity_match": evidence.get("source_identity_match") is True,
                "config_identity_match": evidence.get("config_identity_match") is True,
                "plan_identity_match": evidence.get("plan_identity_match") is True,
                "scientific_contract_match": evidence.get("scientific_contract_match") is True,
                "scientific_cpu_shadow_passed": evidence.get("scientific_cpu_shadow_passed")
                is True,
                "implementation_shadow_passed_if_required": (
                    evidence.get("implementation_shadow_passed") is True
                    if bool(getattr(plan, "implementation_shadow_required", False))
                    else True
                ),
                "critical_event_drop_zero": int(evidence.get("critical_event_drops", -1)) == 0,
                "recorder_required_policy_passed": evidence.get("recorder_required_policy_passed")
                is True,
                "checkpoint_restart_passed": evidence.get("checkpoint_restart_passed") is True,
                "artifact_manifest_complete": evidence.get("artifact_manifest_complete") is True,
                "required_hardware_tests_not_skipped": evidence.get(
                    "required_hardware_tests_not_skipped"
                )
                is True,
            }
        )
    failures = tuple(name for name, passed in base_checks.items() if not passed)
    return ProductionReadiness(
        passed=not failures,
        checks=base_checks,
        failures=failures,
        classification="production" if not failures else "exploratory",
    )


def write_ready_marker(
    readiness: ProductionReadiness,
    marker: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    if not readiness.passed:
        raise RuntimeError("production readiness failed: " + ", ".join(readiness.failures))
    marker = Path(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "readiness": readiness.to_dict(),
        "metadata": metadata or {},
    }
    payload["evidence_sha256"] = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    marker.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return marker


def write_exploratory_marker(
    readiness: ProductionReadiness,
    marker: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    marker = Path(marker)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "classification": "exploratory",
                "readiness": readiness.to_dict(),
                "metadata": metadata or {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker
