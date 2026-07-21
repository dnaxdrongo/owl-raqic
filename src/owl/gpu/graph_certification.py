from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GraphCertification:
    requested_requirement: str
    required_segments: tuple[str, ...]
    captured_segments: tuple[str, ...]
    replay_counts: dict[str, int]
    unexpected_fallbacks: int
    invalidation_count: int
    uncovered_reasons: dict[str, str]
    safety_manifest_passed: bool
    allocation_guard_passed: bool
    pointer_contract_passed: bool
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = "owl.graph-certificate.v2"
        data["required_segments"] = list(self.required_segments)
        data["captured_segments"] = list(self.captured_segments)
        data["failures"] = list(self.failures)
        return data

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "v092_graph_certification.json"
        md_path = directory / "V0_9_2_GRAPH_CERTIFICATION.md"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lines = [
            "# v0.9.2 CUDA Graph Certification",
            "",
            f"**Passed:** {self.passed}",
            f"**Requirement:** {self.requested_requirement}",
            f"**Required:** {', '.join(self.required_segments)}",
            f"**Captured:** {', '.join(self.captured_segments) or 'none'}",
            f"**Invalidations:** {self.invalidation_count}",
            f"**Unexpected fallbacks:** {self.unexpected_fallbacks}",
            f"**Safety manifest:** {self.safety_manifest_passed}",
            f"**Allocation guard:** {self.allocation_guard_passed}",
            f"**Pointer contract:** {self.pointer_contract_passed}",
            "",
            "## Replay counts",
            "```json",
            json.dumps(self.replay_counts, indent=2, sort_keys=True),
            "```",
            "",
            "## Failures",
            *(f"- {item}" for item in self.failures),
        ]
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return json_path, md_path


def certify_graph_status(
    status: dict[str, Any],
    *,
    requirement: str = "full_tick",
    fallback_count: int = 0,
    strict_evidence: bool = False,
) -> GraphCertification:
    coverage = dict(status.get("coverage", {}))
    required = tuple(str(x) for x in coverage.get("required_segments", ()))
    captured = tuple(str(x) for x in coverage.get("captured_segments", ()))
    replay_counts = {
        str(name): int(value) for name, value in dict(coverage.get("replay_counts", {})).items()
    }
    uncovered = {
        str(name): str(reason)
        for name, reason in dict(coverage.get("uncovered_reasons", {})).items()
    }
    failures: list[str] = []
    segments = dict(status.get("segments") or {})
    full_tick = requirement == "full_tick"
    if full_tick:
        missing = sorted(set(required) - set(captured))
        if missing:
            failures.append(f"required graph segments were not captured: {missing}")
        no_replay = [name for name in required if replay_counts.get(name, 0) <= 0]
        if no_replay:
            failures.append(f"captured graph segments did not replay: {no_replay}")
        no_capture_count = [
            name
            for name in required
            if int((segments.get(name) or {}).get("capture_count", 0)) <= 0
        ]
        if strict_evidence and no_capture_count:
            failures.append(f"required segments lack positive capture counts: {no_capture_count}")

    contract = dict(status.get("capture_contract") or {})
    pointer_ok = bool(contract.get("pointer_snapshot_enabled")) and bool(
        contract.get("captured_segments_pointer_stable")
    )
    if full_tick and not pointer_ok:
        failures.append("persistent pointer-stability contract failed")

    manifest = dict(status.get("safety_manifest") or {})
    safety_ok = manifest.get("passed") is True
    if full_tick and strict_evidence and not safety_ok:
        failures.append("operation-level graph-safety manifest did not pass")

    allocation = dict(status.get("allocation_guard") or {})
    allocation_ok = allocation.get("passed") is True
    if full_tick and strict_evidence and not allocation_ok:
        failures.append(
            "capture allocation guard did not produce a passing record for every segment"
        )

    if int(fallback_count):
        failures.append(f"graph run recorded {int(fallback_count)} fallback(s)")
    invalidations = int(status.get("invalidation_count", 0))
    if invalidations:
        failures.append(f"graph invalidated {invalidations} time(s)")
    return GraphCertification(
        requested_requirement=str(requirement),
        required_segments=required,
        captured_segments=captured,
        replay_counts=replay_counts,
        unexpected_fallbacks=int(fallback_count),
        invalidation_count=invalidations,
        uncovered_reasons=uncovered,
        safety_manifest_passed=safety_ok,
        allocation_guard_passed=allocation_ok,
        pointer_contract_passed=pointer_ok,
        passed=not failures,
        failures=tuple(failures),
    )


def certify_run_context(run_context: Any, *, requirement: str | None = None) -> GraphCertification:
    requirement = requirement or str(run_context.graph_manager.requirement)
    return certify_graph_status(
        run_context.graph_manager.graph_status(),
        requirement=requirement,
        fallback_count=int(getattr(run_context, "fallback_count", 0)),
        strict_evidence=True,
    )
