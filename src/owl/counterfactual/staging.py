"""Stage counterfactual tables in bounded columnar batches after transfer."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from owl.counterfactual.scheduler import CounterfactualRunResult
from owl.counterfactual.schema import contribution_id, event_id
from owl.counterfactual.source import CollectedSource


@dataclass(frozen=True)
class TablePacket:
    table_name: str
    columns: dict[str, Any]

    @property
    def rows(self) -> int:
        if not self.columns:
            return 0
        return int(len(next(iter(self.columns.values()))))

    @property
    def nbytes(self) -> int:
        total = 0
        for value in self.columns.values():
            total += int(getattr(value, "nbytes", 0))
            if isinstance(value, list):
                total += sum(len(str(item).encode("utf-8")) for item in value)
        return total


def _array(values: Iterable[Any], dtype: Any | None = None) -> np.ndarray:
    return np.asarray(tuple(values), dtype=dtype)


def _host(backend: Any, value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    return np.asarray(backend.asnumpy(value))


def _context_hash(fields: dict[str, np.ndarray], index: int) -> str:
    digest = hashlib.sha256()
    for name in sorted(fields):
        value = np.ascontiguousarray(fields[name][index])
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode())
        digest.update(len(value.dtype.str).to_bytes(8, "big"))
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def stage_counterfactual_result(
    collected: CollectedSource,
    result: CounterfactualRunResult,
) -> tuple[TablePacket, ...]:
    """Transfer bounded packets and build deterministic host columns."""
    backend = collected.state.backend
    decisions = collected.decisions
    decision_ids = decisions.materialize_ids(backend)
    decision_sequence = _host(backend, decisions.decision_sequence)
    ow_ids = _host(backend, decisions.ow_id)
    selected = _host(backend, decisions.selected_action)
    source_y = _host(backend, decisions.source_y)
    source_x = _host(backend, decisions.source_x)
    agent = {name: _host(backend, value) for name, value in decisions.agent_context_fields.items()}
    oracle = {
        name: _host(backend, value) for name, value in decisions.oracle_context_fields.items()
    }
    source_states = TablePacket(
        "source_states",
        {
            "tick": _array([decisions.tick], np.int64),
            "source_tick": _array([decisions.tick], np.int64),
            "source_state_id": [result.source_state_id],
            "source_root": [result.source_hash.root],
            "state_hash_algorithm": [result.source_hash.algorithm],
            "array_bytes": _array([result.source_hash.array_bytes], np.int64),
            "decision_count": _array([decisions.count], np.int32),
        },
    )
    source_decisions = TablePacket(
        "source_decisions",
        {
            "tick": np.full(decisions.count, decisions.tick, dtype=np.int64),
            "source_tick": np.full(decisions.count, decisions.tick, dtype=np.int64),
            "source_state_id": [result.source_state_id] * decisions.count,
            "source_decision_id": list(decision_ids),
            "decision_sequence": decision_sequence.astype(np.int64, copy=False),
            "ow_id": ow_ids.astype(np.int64, copy=False),
            "source_y": source_y.astype(np.int32, copy=False),
            "source_x": source_x.astype(np.int32, copy=False),
            "factual_selected_action": selected.astype(np.int16, copy=False),
            "candidate_count": np.full(decisions.count, 22, dtype=np.int16),
            "direction_count": np.full(decisions.count, 16, dtype=np.int16),
            "agent_context_hash": [_context_hash(agent, index) for index in range(decisions.count)],
            "oracle_context_hash": [
                _context_hash(oracle, index) for index in range(decisions.count)
            ],
            "factual_schema_digest": [decisions.factual_schema_digest] * decisions.count,
        },
    )
    branch_rows = []
    for branch in result.branches:
        branch_rows.append(
            (
                branch.branch_id,
                branch.source_decision_id,
                branch.repeat_index,
                branch.branch_seed,
                branch.factual_action,
                branch.forced_action,
                branch.anchor,
                branch.status.value,
                branch.validation_passed,
                branch.pre_force_hash.root,
                branch.post_force_hash.root,
                ",".join(branch.force_changed_leaves),
                branch.failure or "",
                branch.runtime_seconds,
            )
        )
    for attempt in result.nonexecutable:
        branch_rows.append(
            (
                attempt.branch_id,
                attempt.source_decision_id,
                attempt.repeat_index,
                attempt.branch_seed,
                attempt.factual_action,
                attempt.forced_action,
                False,
                "nonexecutable",
                False,
                result.source_hash.root,
                result.source_hash.root,
                "",
                "prechoice_nonexecutable",
                0.0,
            )
        )
    branch_rows.sort(key=lambda row: (row[1], row[2], row[5], row[6], row[0]))
    branch_attempts = TablePacket(
        "branch_attempts",
        {
            "tick": np.full(len(branch_rows), decisions.tick, dtype=np.int64),
            "source_state_id": [result.source_state_id] * len(branch_rows),
            "branch_id": [row[0] for row in branch_rows],
            "source_decision_id": [row[1] for row in branch_rows],
            "repeat_index": _array((row[2] for row in branch_rows), np.int32),
            "branch_seed": _array((row[3] for row in branch_rows), np.uint64),
            "factual_action": _array((row[4] for row in branch_rows), np.int16),
            "forced_action": _array((row[5] for row in branch_rows), np.int16),
            "selected_anchor": _array((row[6] for row in branch_rows), bool),
            "branch_status": [row[7] for row in branch_rows],
            "source_validation_passed": _array((row[8] for row in branch_rows), bool),
            "pre_force_root": [row[9] for row in branch_rows],
            "post_force_root": [row[10] for row in branch_rows],
            "force_changed_leaves": [row[11] for row in branch_rows],
            "failure": [row[12] for row in branch_rows],
            "runtime_seconds": _array((row[13] for row in branch_rows), np.float64),
        },
    )

    outcome_rows: list[tuple[Any, ...]] = []
    for branch in result.branches:
        for horizon, packet in sorted(branch.outcomes.items()):
            host = {name: _host(backend, value)[0] for name, value in packet.values.items()}
            outcome_rows.append((branch, horizon, host, branch.horizon_hashes[horizon].root))
    outcome_keys = sorted(
        {name for _, _, host, _ in outcome_rows for name in host} - {"horizon", "source_tick"}
    )
    outcome_columns: dict[str, Any] = {
        "tick": _array((row[2]["end_tick"] for row in outcome_rows), np.int64),
        "source_tick": np.full(len(outcome_rows), decisions.tick, dtype=np.int64),
        "source_state_id": [result.source_state_id] * len(outcome_rows),
        "source_decision_id": [row[0].source_decision_id for row in outcome_rows],
        "branch_id": [row[0].branch_id for row in outcome_rows],
        "repeat_index": _array((row[0].repeat_index for row in outcome_rows), np.int32),
        "forced_action": _array((row[0].forced_action for row in outcome_rows), np.int16),
        "horizon": _array((row[1] for row in outcome_rows), np.int32),
        "end_root": [row[3] for row in outcome_rows],
    }
    for key in outcome_keys:
        outcome_columns[key] = _array(row[2][key] for row in outcome_rows)
    outcomes = TablePacket("counterfactual_micro_rollouts", outcome_columns)

    event_columns: dict[str, list[Any]] = {
        name: []
        for name in (
            "tick",
            "source_decision_id",
            "branch_id",
            "event_id",
            "branch_tick",
            "event_code",
            "stage_code",
            "reason_code",
            "source_y",
            "source_x",
            "target_y",
            "target_x",
            "target_ow_id",
            "payload0",
            "payload1",
            "payload2",
            "payload3",
        )
    }
    contribution_columns: dict[str, list[Any]] = {
        name: []
        for name in (
            "tick",
            "source_decision_id",
            "branch_id",
            "contribution_id",
            "branch_tick",
            "contribution_code",
            "source_y",
            "source_x",
            "field",
            "delta",
            "start_value",
            "end_value",
        )
    }
    for branch in result.branches:
        for evidence_packet in branch.evidence:
            events = {
                name: _host(backend, value) for name, value in evidence_packet.event_arrays.items()
            }
            slots, flat = np.nonzero(events["event_active"])
            width = int(collected.state.arrays["health"].shape[1])
            sy = flat // width
            sx = flat % width
            for index in range(len(flat)):
                slot = int(slots[index])
                code = int(evidence_packet.event_codes[slot])
                identity = event_id(branch.branch_id, evidence_packet.tick, code, int(flat[index]))
                values = {
                    "tick": evidence_packet.tick,
                    "source_decision_id": branch.source_decision_id,
                    "branch_id": branch.branch_id,
                    "event_id": identity,
                    "branch_tick": evidence_packet.tick - decisions.tick + 1,
                    "event_code": code,
                    "stage_code": int(events["event_stage_code"][slot, flat[index]]),
                    "reason_code": int(events["event_reason_code"][slot, flat[index]]),
                    "source_y": int(sy[index]),
                    "source_x": int(sx[index]),
                    "target_y": int(events["event_target_y"][slot, flat[index]]),
                    "target_x": int(events["event_target_x"][slot, flat[index]]),
                    "target_ow_id": int(events["event_target_ow_id"][slot, flat[index]]),
                    "payload0": float(events["event_payload"][slot, flat[index], 0]),
                    "payload1": float(events["event_payload"][slot, flat[index], 1]),
                    "payload2": float(events["event_payload"][slot, flat[index], 2]),
                    "payload3": float(events["event_payload"][slot, flat[index], 3]),
                }
                for name, value in values.items():
                    event_columns[name].append(value)
            contributions = {
                name: _host(backend, value)
                for name, value in evidence_packet.contribution_arrays.items()
            }
            delta = contributions["contribution_delta"]
            code_slots, cy, cx, field_slots = np.nonzero(delta != 0)
            for index in range(len(code_slots)):
                code_slot = int(code_slots[index])
                field_slot = int(field_slots[index])
                code = int(evidence_packet.contribution_codes[code_slot])
                field_name = evidence_packet.contribution_fields[field_slot]
                identity = contribution_id(
                    branch.branch_id,
                    evidence_packet.tick,
                    code,
                    int(cy[index]),
                    int(cx[index]),
                    field_name,
                )
                values = {
                    "tick": evidence_packet.tick,
                    "source_decision_id": branch.source_decision_id,
                    "branch_id": branch.branch_id,
                    "contribution_id": identity,
                    "branch_tick": evidence_packet.tick - decisions.tick + 1,
                    "contribution_code": code,
                    "source_y": int(cy[index]),
                    "source_x": int(cx[index]),
                    "field": field_name,
                    "delta": float(delta[code_slot, cy[index], cx[index], field_slot]),
                    "start_value": float(
                        contributions["tick_start"][cy[index], cx[index], field_slot]
                    ),
                    "end_value": float(contributions["tick_end"][cy[index], cx[index], field_slot]),
                }
                for name, value in values.items():
                    contribution_columns[name].append(value)
    events_packet = TablePacket(
        "branch_events",
        {
            name: _array(values)
            if name not in {"source_decision_id", "branch_id", "event_id"}
            else values
            for name, values in event_columns.items()
        },
    )
    contributions_packet = TablePacket(
        "branch_contributions",
        {
            name: _array(values)
            if name not in {"source_decision_id", "branch_id", "contribution_id", "field"}
            else values
            for name, values in contribution_columns.items()
        },
    )
    summary_keys: dict[tuple[str, str, int], int] = {}
    for branch_value, decision_value, code in zip(
        event_columns["branch_id"],
        event_columns["source_decision_id"],
        event_columns["event_code"],
        strict=True,
    ):
        key = (branch_value, decision_value, int(code))
        summary_keys[key] = summary_keys.get(key, 0) + 1
    summary_order = sorted(summary_keys)
    summaries = TablePacket(
        "branch_event_summaries",
        {
            "tick": np.full(len(summary_order), decisions.tick, dtype=np.int64),
            "source_decision_id": [item[1] for item in summary_order],
            "branch_id": [item[0] for item in summary_order],
            "event_code": _array((item[2] for item in summary_order), np.int16),
            "event_count": _array((summary_keys[item] for item in summary_order), np.int64),
        },
    )
    pair_rows = sorted(
        result.pairs,
        key=lambda row: (
            row.source_decision_id,
            row.repeat_index,
            row.action_a,
            row.action_b,
            row.horizon,
        ),
    )
    pairs = TablePacket(
        "candidate_pairs",
        {
            "tick": np.full(len(pair_rows), decisions.tick, dtype=np.int64),
            "pair_id": [row.pair_id for row in pair_rows],
            "source_decision_id": [row.source_decision_id for row in pair_rows],
            "repeat_index": _array((row.repeat_index for row in pair_rows), np.int32),
            "action_a": _array((row.action_a for row in pair_rows), np.int16),
            "action_b": _array((row.action_b for row in pair_rows), np.int16),
            "branch_a": [row.branch_a for row in pair_rows],
            "branch_b": [row.branch_b for row in pair_rows],
            "horizon": _array((row.horizon for row in pair_rows), np.int32),
        },
    )
    missing = result.nonexecutable
    nonexecutable = TablePacket(
        "nonexecutable_candidates",
        {
            "tick": np.full(len(missing), decisions.tick, dtype=np.int64),
            "source_decision_id": [row.source_decision_id for row in missing],
            "repeat_index": _array((row.repeat_index for row in missing), np.int32),
            "forced_action": _array((row.forced_action for row in missing), np.int16),
            "policy_legal": _array((row.policy_legal for row in missing), bool),
            "prechoice_executable": _array((row.prechoice_executable for row in missing), bool),
            "reason_code": _array((row.reason_code for row in missing), np.int16),
        },
    )
    return (
        source_states,
        source_decisions,
        branch_attempts,
        outcomes,
        events_packet,
        summaries,
        contributions_packet,
        pairs,
        nonexecutable,
    )
