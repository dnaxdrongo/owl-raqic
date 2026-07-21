#!/usr/bin/env python3
"""Independently certify a completed counterfactual evidence artifact tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.core.actions import Action  # noqa: E402
from owl.counterfactual.rng_registry import registry_manifest  # noqa: E402
from owl.counterfactual.schema import (  # noqa: E402
    COUNTERFACTUAL_CERTIFICATE_VERSION,
    COUNTERFACTUAL_SCHEMA_DIGEST,
    COUNTERFACTUAL_SCHEMA_VERSION,
    TABLE_CONTRACTS,
)
from owl.experiments.controller import _release_hash  # noqa: E402
from owl.record.cadc_schema import (  # noqa: E402
    CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
    CADC_ACTION_TRANSITION_SCHEMA_VERSION,
)

ACTION_ORDER = (
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
)

TARGET_COMMANDS = (
    "gpu_preflight",
    "repair_regressions",
    "acceptance_runner",
    "pytest",
    "ruff",
    "mypy",
    "hotpath_audit",
    "profile",
    "pip_check",
)

REQUIRED_ACCEPTANCE_SECTIONS = (
    "phase25",
    "counterfactual_schema_version",
    "counterfactual_schema_digest",
    "phase3_source_sha256",
    "resolved_counterfactual_config",
    "device",
    "factual_recovery",
    "factual_nonmutation",
    "source_counts",
    "anchor",
    "memory_plan",
    "transfer",
    "performance",
    "parquet",
)


@dataclass(frozen=True)
class Gate:
    passed: bool
    detail: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def gate(gates: dict[str, Gate], name: str, condition: bool, detail: str) -> None:
    gates[name] = Gate(bool(condition), detail)


def _incomplete_acceptance_certificate(
    args: argparse.Namespace,
    input_root: Path,
    manifest: dict[str, Any],
    missing: list[str],
) -> dict[str, Any]:
    gates: dict[str, Gate] = {}
    upstream_failures = [str(value) for value in manifest.get("failures", [])]
    gate(
        gates,
        "acceptance_manifest",
        False,
        json.dumps(
            {
                "passed": manifest.get("passed"),
                "failure_stage": manifest.get("failure_stage"),
                "failures": upstream_failures,
                "missing_sections": missing,
            },
            sort_keys=True,
        ),
    )
    actual_source = _release_hash(Path(args.source_root).resolve())
    declared_source = manifest.get("phase3_source_sha256")
    gate(
        gates,
        "source_scope",
        bool(declared_source) and actual_source == declared_source,
        f"expected={declared_source} actual={actual_source}",
    )
    status_path = input_root / "command_status.json"
    status = load_json(status_path) if status_path.is_file() else {}
    required_commands = TARGET_COMMANDS if args.require_target else ("acceptance_runner",)
    gate(
        gates,
        "command_status",
        all(int(status.get(name, -1)) == 0 for name in required_commands),
        json.dumps({name: status.get(name) for name in required_commands}, sort_keys=True),
    )
    phase25_path = input_root / "phase25_gate" / "phase25_target_gpu_certificate.json"
    hardening_path = input_root / "phase25_gate" / "hardening_receipt.json"
    phase25: dict[str, Any] = {}
    hardening: dict[str, Any] = {}
    if phase25_path.is_file() and hardening_path.is_file():
        phase25 = load_json(phase25_path)
        hardening = load_json(hardening_path)
    phase25_ok = (
        phase25.get("passed") is True
        and phase25.get("classification") == "PHASE2_5_TARGET_GPU_CERTIFIED"
        and phase25.get("phase3_unlocked") is True
        and hardening.get("passed") is True
        and hardening.get("base_certified_source_sha256") == phase25.get("source_sha256")
    )
    gate(gates, "phase25_artifacts", phase25_ok, "independent upstream artifact validation")
    return {
        "schema_version": COUNTERFACTUAL_CERTIFICATE_VERSION,
        "passed": False,
        "classification": "FAILED_CLOSED_UPSTREAM_ACCEPTANCE",
        "phase4_unlocked": False,
        "phase25_source_sha256": phase25.get("source_sha256"),
        "phase25_certificate_sha256": (
            sha256_file(phase25_path) if phase25_path.is_file() else None
        ),
        "phase3_source_sha256": declared_source or actual_source,
        "counterfactual_schema_digest": COUNTERFACTUAL_SCHEMA_DIGEST,
        "factual_v2_digest": CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
        "rng_registry_digest": registry_manifest()["registry_digest"],
        "target_requested": args.require_target,
        "gates": {name: asdict(value) for name, value in sorted(gates.items())},
        "row_counts": {},
        "action_coverage": {},
        "upstream_failure_stage": manifest.get("failure_stage"),
        "upstream_failures": upstream_failures,
        "failures": [name for name, value in gates.items() if not value.passed],
    }


def _column(table: Any, name: str) -> list[Any]:
    if name not in table.column_names:
        raise KeyError(f"missing column {name}")
    return cast(list[Any], table[name].combine_chunks().to_pylist())


def _read_tables(root: Path, receipts: list[dict[str, Any]]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    grouped: dict[str, list[Any]] = defaultdict(list)
    for receipt in receipts:
        table_name = str(receipt["table_name"])
        path = root / "counterfactual" / table_name / str(receipt["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(receipt["sha256"]):
            raise RuntimeError(f"Parquet checksum mismatch: {path}")
        table = pq.read_table(path)
        if table.num_rows != int(receipt["rows"]):
            raise RuntimeError(f"Parquet row mismatch: {path}")
        metadata = table.schema.metadata or {}
        expected = {
            b"owl.counterfactual.schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
            b"owl.counterfactual.schema_digest": COUNTERFACTUAL_SCHEMA_DIGEST,
            b"owl.phase3.source_sha256": str(receipt["source_sha256"]),
            b"owl.phase25.certificate_sha256": str(receipt["phase25_certificate_sha256"]),
            b"owl.cadc.factual_v2_digest": CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
            b"owl.counterfactual.rng_registry_digest": str(registry_manifest()["registry_digest"]),
        }
        for key, value in expected.items():
            if metadata.get(key, b"").decode() != value:
                raise RuntimeError(f"Parquet metadata mismatch {key!r}: {path}")
        grouped[table_name].append(table)
    return {
        name: pa.concat_tables(parts, promote_options="default") for name, parts in grouped.items()
    }


def _verify_phase25(input_root: Path, manifest: dict[str, Any]) -> tuple[bool, str]:
    certificate_path = input_root / "phase25_gate" / "phase25_target_gpu_certificate.json"
    hardening_path = input_root / "phase25_gate" / "hardening_receipt.json"
    certificate = load_json(certificate_path)
    hardening = load_json(hardening_path)
    phase25 = manifest["phase25"]
    checks = (
        certificate.get("schema_version") == "owl.phase2.5.target-gpu-certificate.v1",
        certificate.get("passed") is True,
        certificate.get("classification") == "PHASE2_5_TARGET_GPU_CERTIFIED",
        certificate.get("phase3_unlocked") is True,
        certificate.get("phase4_unlocked") is False,
        certificate.get("source_sha256") == phase25.get("certified_source_sha256"),
        certificate.get("cadc_schema_digest") == CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
        sha256_file(certificate_path) == phase25.get("certificate_sha256"),
        hardening.get("passed") is True,
        hardening.get("base_certified_source_sha256") == certificate.get("source_sha256"),
        hardening.get("hardened_source_sha256") == phase25.get("hardening_source_sha256"),
        sha256_file(hardening_path) == phase25.get("hardening_receipt_sha256"),
    )
    return all(checks), f"{sum(checks)}/{len(checks)} Phase 2.5 identity checks"


def certify(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input).resolve()
    manifest = load_json(input_root / "phase3_acceptance_manifest.json")
    missing = [name for name in REQUIRED_ACCEPTANCE_SECTIONS if name not in manifest]
    if manifest.get("passed") is not True or missing:
        return _incomplete_acceptance_certificate(args, input_root, manifest, missing)
    gates: dict[str, Gate] = {}

    gate(
        gates,
        "acceptance_manifest",
        manifest.get("schema_version") == "owl.cadc.phase3-acceptance.v1"
        and manifest.get("passed") is True
        and manifest.get("phase4_unlocked") is False,
        "acceptance passed while retaining the Phase 4 lock",
    )
    phase25_ok, phase25_detail = _verify_phase25(input_root, manifest)
    gate(gates, "phase25_gate", phase25_ok, phase25_detail)
    gate(
        gates,
        "schema_identity",
        manifest.get("counterfactual_schema_version") == COUNTERFACTUAL_SCHEMA_VERSION
        and manifest.get("counterfactual_schema_digest") == COUNTERFACTUAL_SCHEMA_DIGEST
        and manifest["phase25"].get("factual_schema") == CADC_ACTION_TRANSITION_SCHEMA_VERSION
        and manifest["phase25"].get("factual_schema_digest")
        == CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
        "counterfactual and factual-v2 schema identities",
    )
    gate(
        gates,
        "action_order",
        tuple(action.name for action in Action) == ACTION_ORDER and len(Action) == 22,
        "immutable 22-action axis",
    )
    actual_source = _release_hash(Path(args.source_root).resolve())
    gate(
        gates,
        "source_scope",
        actual_source == manifest.get("phase3_source_sha256"),
        f"expected={manifest.get('phase3_source_sha256')} actual={actual_source}",
    )

    cf_manifest = load_json(input_root / "counterfactual" / "counterfactual_manifest.json")
    receipts = list(cf_manifest.get("parts", []))
    expected_tables = {contract.name for contract in TABLE_CONTRACTS}
    receipt_tables = {str(item.get("table_name")) for item in receipts}
    gate(
        gates,
        "part_manifest",
        cf_manifest.get("schema_digest") == COUNTERFACTUAL_SCHEMA_DIGEST
        and cf_manifest.get("source_sha256") == manifest.get("phase3_source_sha256")
        and cf_manifest.get("phase25_certificate_sha256")
        == manifest["phase25"].get("certificate_sha256")
        and receipt_tables == expected_tables,
        f"tables={sorted(receipt_tables)}",
    )
    tables = _read_tables(input_root, receipts)
    gate(
        gates,
        "parquet_receipts",
        set(tables) == expected_tables,
        f"verified {len(receipts)} checksum-receipted parts",
    )

    decisions = tables["source_decisions"]
    attempts = tables["branch_attempts"]
    outcomes = tables["counterfactual_micro_rollouts"]
    events = tables["branch_events"]
    contributions = tables["branch_contributions"]
    pairs = tables["candidate_pairs"]
    nonexec = tables["nonexecutable_candidates"]
    decision_ids = set(_column(decisions, "source_decision_id"))
    gate(
        gates,
        "source_decisions",
        decisions.num_rows > 0
        and set(_column(decisions, "candidate_count")) == {22}
        and set(_column(decisions, "direction_count")) == {16}
        and set(_column(decisions, "factual_schema_digest"))
        == {CADC_ACTION_TRANSITION_SCHEMA_DIGEST}
        and len(decision_ids) == decisions.num_rows,
        f"decisions={decisions.num_rows}",
    )

    attempt_decisions = _column(attempts, "source_decision_id")
    repeat_indices = _column(attempts, "repeat_index")
    actions = _column(attempts, "forced_action")
    statuses = _column(attempts, "branch_status")
    anchors = _column(attempts, "selected_anchor")
    branch_ids = _column(attempts, "branch_id")
    repeat_values = sorted({int(value) for value in repeat_indices if int(value) >= 0})
    nonanchor_rows = sum(int(value) >= 0 for value in repeat_indices)
    expected_attempts = decisions.num_rows * 22 * len(repeat_values)
    gate(
        gates,
        "attempt_grain",
        set(attempt_decisions) <= decision_ids
        and nonanchor_rows == expected_attempts
        and len(branch_ids) == len(set(branch_ids))
        and all(0 <= int(action) < 22 for action in actions),
        f"nonanchor={nonanchor_rows} expected={expected_attempts}",
    )
    completed_indices = [
        index
        for index, status in enumerate(statuses)
        if status == "completed" and not anchors[index]
    ]
    completed_actions = Counter(Action(int(actions[index])).name for index in completed_indices)
    required_coverage = {
        "SENSE",
        "FLEE",
        "PURSUE",
        "FEED",
        "COMMUNICATE",
        "REPAIR",
        "REPRODUCE",
    }
    directional = {
        "MOVE_N",
        "MOVE_S",
        "MOVE_E",
        "MOVE_W",
        "MOVE_NE",
        "MOVE_NW",
        "MOVE_SE",
        "MOVE_SW",
    }
    gate(
        gates,
        "action_coverage",
        required_coverage <= set(completed_actions) and bool(directional & set(completed_actions)),
        json.dumps(dict(sorted(completed_actions.items())), sort_keys=True),
    )
    validation = _column(attempts, "source_validation_passed")
    changed = _column(attempts, "force_changed_leaves")
    allowed_changed = {
        "",
        "arrays.readout",
        "arrays.raqic_readout",
        "arrays.raqic_readout,arrays.readout",
    }
    gate(
        gates,
        "forcing_validation",
        all(validation[index] for index in completed_indices)
        and all(value in allowed_changed for value in changed)
        and not any(status == "failed" for status in statuses),
        "complete target validation and registered high-level force leaves",
    )
    anchor_indices = [index for index, value in enumerate(anchors) if value]
    gate(
        gates,
        "selected_anchor",
        len(anchor_indices) == decisions.num_rows
        and all(statuses[index] == "completed" for index in anchor_indices)
        and manifest["anchor"].get("passed") is True,
        f"anchors={len(anchor_indices)}",
    )

    seed_groups: dict[tuple[str, int], set[int]] = defaultdict(set)
    seeds = _column(attempts, "branch_seed")
    for decision_id, repeat, seed in zip(attempt_decisions, repeat_indices, seeds, strict=True):
        if int(repeat) >= 0:
            seed_groups[(decision_id, int(repeat))].add(int(seed))
    gate(
        gates,
        "common_random_numbers",
        bool(seed_groups) and all(len(values) == 1 for values in seed_groups.values()),
        f"paired_groups={len(seed_groups)}",
    )

    branch_id_set = set(branch_ids)
    outcome_branch_ids = _column(outcomes, "branch_id")
    pair_left = _column(pairs, "branch_a")
    pair_right = _column(pairs, "branch_b")
    expected_horizons: dict[int, set[int]] = {}
    cfg = manifest["resolved_counterfactual_config"]
    base_horizons = {int(value) for value in cfg["horizons"]}
    family_horizons = cfg.get("family_horizons", {})
    for action in Action:
        expected_horizons[int(action)] = base_horizons | {
            int(value) for value in family_horizons.get(action.name, [])
        }
    outcome_by_branch: dict[str, set[int]] = defaultdict(set)
    for branch_id_value, horizon in zip(
        outcome_branch_ids, _column(outcomes, "horizon"), strict=True
    ):
        outcome_by_branch[branch_id_value].add(int(horizon))
    completed_branch_actions = {
        branch_ids[index]: int(actions[index])
        for index, status in enumerate(statuses)
        if status == "completed"
    }
    horizon_ok = all(
        outcome_by_branch.get(branch_id_value, set()) == expected_horizons[action]
        for branch_id_value, action in completed_branch_actions.items()
    )
    gate(
        gates,
        "multi_horizon_outcomes",
        horizon_ok and set(outcome_branch_ids) <= branch_id_set,
        f"outcome_rows={outcomes.num_rows}",
    )
    gate(
        gates,
        "pair_joins",
        pairs.num_rows > 0
        and set(_column(pairs, "source_decision_id")) <= decision_ids
        and set(pair_left) <= branch_id_set
        and set(pair_right) <= branch_id_set,
        f"pairs={pairs.num_rows}",
    )
    gate(
        gates,
        "nonexecutable_joins",
        nonexec.num_rows == sum(status == "nonexecutable" for status in statuses)
        and set(_column(nonexec, "source_decision_id")) <= decision_ids
        and set(_column(nonexec, "prechoice_executable")) <= {False},
        f"nonexecutable={nonexec.num_rows}",
    )
    gate(
        gates,
        "event_contribution_joins",
        set(_column(events, "branch_id")) <= branch_id_set
        and set(_column(contributions, "branch_id")) <= branch_id_set
        and len(_column(events, "event_id")) == len(set(_column(events, "event_id")))
        and len(_column(contributions, "contribution_id"))
        == len(set(_column(contributions, "contribution_id"))),
        f"events={events.num_rows} contributions={contributions.num_rows}",
    )
    gate(
        gates,
        "factual_isolation",
        manifest["factual_recovery"].get("passed") is True
        and manifest["factual_nonmutation"].get("passed") is True
        and manifest["source_counts"].get("pointer_isolation_validated") is True,
        "observer no-op, branch nonmutation, and disjoint pointers",
    )
    memory = manifest["memory_plan"]
    gate(
        gates,
        "memory_bounds",
        memory.get("passed") is True
        and int(memory.get("max_active_branches", 0)) >= 1
        and int(manifest["performance"].get("packet_bytes", 0)) <= int(cfg["max_pending_bytes"]),
        f"Bmax={memory.get('max_active_branches')}",
    )
    transfer = manifest["transfer"]
    branch_transfer = transfer["branch"]
    gate(
        gates,
        "transfer_contract",
        int(branch_transfer.get("d2d_bytes", 0)) > 0
        and all(record.get("scheduled") is True for record in branch_transfer.get("records", [])),
        f"d2d={branch_transfer.get('d2d_bytes')} d2h={branch_transfer.get('d2h_bytes')}",
    )
    gate(
        gates,
        "overflow",
        int(manifest.get("event_overflow", -1)) == 0,
        f"event_overflow={manifest.get('event_overflow')}",
    )

    status = load_json(input_root / "command_status.json")
    required_commands = TARGET_COMMANDS if args.require_target else ("acceptance_runner",)
    gate(
        gates,
        "command_status",
        all(int(status.get(name, -1)) == 0 for name in required_commands),
        json.dumps({name: status.get(name) for name in required_commands}, sort_keys=True),
    )
    target_requested = args.require_target is not None
    device = manifest["device"]
    target_name = str(args.require_target or "")
    cuda = device.get("cuda_device", {})
    preflight: dict[str, Any] = {}
    preflight_path = input_root / "gpu_preflight.json"
    if target_requested and preflight_path.is_file():
        preflight = load_json(preflight_path)
    preflight_device = preflight.get("device", {})
    if target_requested:
        gate(
            gates,
            "gpu_preflight",
            preflight.get("passed") is True
            and preflight.get("classification")
            == f"{target_name.upper()}_PHASE3_GPU_PREFLIGHT_PASSED"
            and preflight.get("source_sha256") == manifest.get("phase3_source_sha256")
            and target_name.upper() in str(preflight_device.get("name", "")).upper()
            and int(preflight.get("device_to_host_bytes", 0))
            > int(preflight.get("array_bytes", 0)),
            json.dumps(preflight, sort_keys=True),
        )

    memory_samples: list[tuple[int, int, int]] = []
    samples_path = input_root / "gpu_memory_samples.csv"
    if target_requested and samples_path.is_file():
        import csv

        with samples_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                memory_samples.append(
                    (
                        int(row["memory_used_mib"]),
                        int(row["memory_total_mib"]),
                        int(row["process_rss_kib"]),
                    )
                )
    if target_requested:
        used = [sample[0] for sample in memory_samples]
        totals = [sample[1] for sample in memory_samples]
        rss = [sample[2] for sample in memory_samples]
        peak_delta_bytes = (max(used) - used[0]) * 1024**2 if used else -1
        gate(
            gates,
            "observed_memory_bounds",
            len(memory_samples) >= 2
            and min(used) >= 0
            and min(totals) > 0
            and max(used) <= min(totals)
            and min(rss) >= 0
            and peak_delta_bytes <= int(memory.get("allowed_bytes", -1)),
            json.dumps(
                {
                    "samples": len(memory_samples),
                    "baseline_used_mib": used[0] if used else None,
                    "peak_used_mib": max(used) if used else None,
                    "peak_process_rss_kib": max(rss) if rss else None,
                    "peak_delta_bytes": peak_delta_bytes,
                    "allowed_bytes": memory.get("allowed_bytes"),
                },
                sort_keys=True,
            ),
        )
    target_ok = (
        manifest.get("backend") == "cupy"
        and device.get("is_gpu") is True
        and target_name.upper() in str(cuda.get("name", "")).upper()
        and int(cuda.get("total_global_memory_bytes", 0)) > 0
        and int(cuda.get("runtime_version", 0)) > 0
        and int(cuda.get("driver_version", 0)) > 0
        and bool(cuda.get("cupy_version"))
        and int(branch_transfer.get("d2h_bytes", 0)) > 0
        and manifest["performance"].get("execution_strategy") == "multi_stream_whole_array"
        and int(manifest["performance"].get("worker_count", 0)) > 1
    )
    if target_requested:
        gate(gates, "target_gpu", target_ok, json.dumps(cuda, sort_keys=True))

    failures = [name for name, value in gates.items() if not value.passed]
    passed = not failures
    if target_requested and passed and target_name.upper() == "H100":
        classification = "H100_PHASE3_TARGET_CERTIFIED"
        phase4_unlocked = True
    elif passed:
        classification = "LOCAL_VALIDATED_TARGET_GPU_PENDING"
        phase4_unlocked = False
    else:
        classification = "FAILED_CLOSED"
        phase4_unlocked = False
    return {
        "schema_version": COUNTERFACTUAL_CERTIFICATE_VERSION,
        "passed": passed,
        "classification": classification,
        "phase4_unlocked": phase4_unlocked,
        "phase25_source_sha256": manifest["phase25"]["certified_source_sha256"],
        "phase25_certificate_sha256": manifest["phase25"]["certificate_sha256"],
        "phase3_source_sha256": manifest["phase3_source_sha256"],
        "counterfactual_schema_digest": COUNTERFACTUAL_SCHEMA_DIGEST,
        "factual_v2_digest": CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
        "rng_registry_digest": registry_manifest()["registry_digest"],
        "target_requested": args.require_target,
        "gates": {name: asdict(value) for name, value in sorted(gates.items())},
        "row_counts": {name: table.num_rows for name, table in sorted(tables.items())},
        "action_coverage": dict(sorted(completed_actions.items())),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--require-target", choices=("H100", "H200", "B200"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    try:
        certificate = certify(args)
    except Exception as exc:
        certificate = {
            "schema_version": COUNTERFACTUAL_CERTIFICATE_VERSION,
            "passed": False,
            "classification": "FAILED_CLOSED",
            "phase4_unlocked": False,
            "failures": [f"certifier_exception: {type(exc).__name__}: {exc}"],
        }
    atomic_json(output, certificate)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    if not certificate.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
