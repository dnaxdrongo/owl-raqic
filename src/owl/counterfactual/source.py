"""Collect factual source decisions at the action-transition seam."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from owl.counterfactual.schema import SourceBoundary, source_decision_id, source_state_id
from owl.counterfactual.state_clone import CounterfactualSourceState, capture_source_state
from owl.record.cadc_schema import CADC_ACTION_TRANSITION_SCHEMA_DIGEST


class DecisionBoundaryObserver(Protocol):
    def capture(self, run: Any, decision_event: Any) -> None: ...


@dataclass(frozen=True)
class SourceDecisionBatch:
    """Backend-native selected decision and candidate evidence."""

    source_state_id: str
    run_id: str
    condition: str
    seed: int
    tick: int
    source_flat: Any
    source_y: Any
    source_x: Any
    decision_sequence: Any
    ow_id: Any
    lineage_id: Any
    selected_action: Any
    selected_probability: Any
    policy_legal: Any
    prechoice_executable: Any
    candidate_reason: Any
    candidate_target_kind: Any
    candidate_proposed_y: Any
    candidate_proposed_x: Any
    candidate_resolved_y: Any
    candidate_resolved_x: Any
    candidate_target_ow_id: Any
    candidate_target_source: Any
    candidate_target_distance: Any
    candidate_target_confidence: Any
    candidate_compiled_action: Any
    direction_fields: Mapping[str, Any]
    agent_context_fields: Mapping[str, Any]
    oracle_context_fields: Mapping[str, Any]
    factual_schema_digest: str = CADC_ACTION_TRANSITION_SCHEMA_DIGEST

    @property
    def count(self) -> int:
        return int(self.source_flat.shape[0])

    def materialize_ids(self, backend: Any) -> tuple[str, ...]:
        """Transfer only compact identity vectors at the orchestration boundary."""
        sequences = backend.asnumpy(self.decision_sequence)
        ow_ids = backend.asnumpy(self.ow_id)
        return tuple(
            source_decision_id(
                self.source_state_id,
                self.run_id,
                self.condition,
                self.seed,
                self.tick,
                int(sequence),
                int(ow_id),
            )
            for sequence, ow_id in zip(sequences, ow_ids, strict=True)
        )


@dataclass
class CollectedSource:
    state: CounterfactualSourceState
    decisions: SourceDecisionBatch
    factual_evidence: Mapping[str, Any]
    factual_horizons: dict[int, CounterfactualSourceState] = field(default_factory=dict)


@dataclass
class CounterfactualSourceCollector:
    """Bounded observer attached after factual selected-intent capture."""

    cfg: Any
    phase25_source_sha256: str
    run_id: str = "unidentified"
    condition: str = "unidentified"
    sources: list[CollectedSource] = field(default_factory=list)
    source_copy_bytes: int = 0
    source_copy_count: int = 0
    _source_stream: Any = None

    def __post_init__(self) -> None:
        if not bool(self.cfg.counterfactual.enabled):
            raise ValueError("counterfactual source collector requires enabled configuration")
        if self.cfg.counterfactual.source_boundary != SourceBoundary.POST_SELECTION_PRE_ACTIONS:
            raise ValueError("unsupported counterfactual source boundary")

    def _stream(self, run: Any) -> Any:
        if self._source_stream is not None:
            return self._source_stream
        if run.ds.is_gpu:
            self._source_stream = run.ds.xp.cuda.Stream(non_blocking=True)
        else:
            from owl.gpu.streams import NullStream

            self._source_stream = NullStream()
        return self._source_stream

    def _select_flats(self, buffer: Any) -> Any:
        xp = buffer.xp
        sequence = buffer.arrays["decision_sequence"].reshape(-1)
        live = xp.nonzero(sequence >= 0)[0]
        if self.cfg.counterfactual.source_selection_mode == "explicit":
            requested = []
            for token in self.cfg.counterfactual.explicit_source_decision_ids:
                value = str(token)
                if value.startswith("sequence:"):
                    value = value.split(":", 1)[1]
                try:
                    requested.append(int(value))
                except ValueError as exc:
                    raise ValueError(
                        "live explicit collection accepts decision sequence tokens; "
                        "stable IDs are resolved by the offline source index"
                    ) from exc
            requested_array = xp.asarray(requested, dtype=xp.int64)
            mask = xp.any(sequence[:, None] == requested_array[None, :], axis=1)
            live = xp.nonzero(mask)[0]
        elif self.cfg.counterfactual.source_selection_mode == "deterministic_hash":
            # Backend-native SplitMix64 ordering: selection requires no scan of
            # candidate rows on the host and is invariant to launch ordering.
            values = sequence[live].astype(xp.uint64, copy=False)
            values ^= xp.uint64(int(self.cfg.world.seed))
            values += xp.uint64(0x9E3779B97F4A7C15)
            values = (values ^ (values >> xp.uint64(30))) * xp.uint64(0xBF58476D1CE4E5B9)
            values = (values ^ (values >> xp.uint64(27))) * xp.uint64(0x94D049BB133111EB)
            values ^= values >> xp.uint64(31)
            live = live[xp.argsort(values, kind="stable")]
        elif self.cfg.counterfactual.source_selection_mode == "action_family_stratified":
            actions = buffer.arrays["selected_action"].reshape(-1)[live]
            # Stable action-major ordering guarantees early family coverage;
            # decision sequence provides the deterministic within-family key.
            order = xp.lexsort((sequence[live], actions))
            live = live[order]
        else:  # pragma: no cover - configuration validation owns this branch
            raise RuntimeError("unknown counterfactual source selection mode")
        remaining = int(self.cfg.counterfactual.max_source_decisions) - sum(
            source.decisions.count for source in self.sources
        )
        return live[: max(remaining, 0)]

    def capture(self, run: Any, decision_event: Any) -> None:
        if len(self.sources) >= int(self.cfg.counterfactual.max_source_ticks):
            return
        if run.cadc_buffer is None:
            raise RuntimeError("Phase 3 source collection requires factual v2 device evidence")
        buffer = run.cadc_buffer
        if buffer.schema_digest != CADC_ACTION_TRANSITION_SCHEMA_DIGEST:
            raise RuntimeError("Phase 3 source collector requires owl.cadc.factual.v2")
        flat = self._select_flats(buffer)
        if int(flat.shape[0]) == 0:
            return
        xp = run.ds.xp
        width = int(run.ds.health.shape[1])
        y = (flat // width).astype(xp.int32)
        x = (flat % width).astype(xp.int32)
        tick = int(run.ds.tick)
        sid = source_state_id(
            self.phase25_source_sha256,
            self.run_id,
            self.condition,
            int(self.cfg.world.seed),
            tick,
            SourceBoundary.POST_SELECTION_PRE_ACTIONS.value,
        )
        stream = self._stream(run)
        run.streams.wait(stream, decision_event)
        ready_event = run.streams.new_event()
        source = capture_source_state(
            run.ds,
            sid,
            stream=stream,
            ready_event=ready_event,
        )
        run.streams.wait(run.streams.compute, ready_event)
        arrays = buffer.arrays
        candidate_names = {
            "policy_legal": "policy_legal",
            "prechoice_executable": "candidate_executable",
            "candidate_reason": "candidate_reason_code",
            "candidate_target_kind": "candidate_target_kind",
            "candidate_proposed_y": "candidate_proposed_y",
            "candidate_proposed_x": "candidate_proposed_x",
            "candidate_resolved_y": "candidate_resolved_y",
            "candidate_resolved_x": "candidate_resolved_x",
            "candidate_target_ow_id": "candidate_target_ow_id",
            "candidate_target_source": "candidate_target_source",
            "candidate_target_distance": "candidate_target_distance",
            "candidate_target_confidence": "candidate_target_confidence",
            "candidate_compiled_action": "candidate_compiled_action",
        }
        selected_candidate = {
            name: arrays[source_name][y, x].copy() for name, source_name in candidate_names.items()
        }
        direction_fields = {
            name: arrays[name][y, x].copy()
            for name in (
                "action_target_y",
                "action_target_x",
                "action_target_ow_id",
                "action_target_kind",
                "action_target_source",
                "action_target_distance",
                "action_target_confidence",
                "action_direction_y",
                "action_direction_x",
                "action_direction_executable",
                "action_direction_score",
                "action_direction_distance_delta",
                "action_direction_hazard",
                "action_direction_opportunity",
            )
        }
        agent_context = {
            name: arrays[name][y, x].copy() for name in arrays if name.startswith("agent_")
        }
        oracle_context = {
            name: arrays[name][y, x].copy()
            for name in arrays
            if name.startswith("oracle_") or name.startswith("dense_oracle_")
        }
        decisions = SourceDecisionBatch(
            source_state_id=sid,
            run_id=self.run_id,
            condition=self.condition,
            seed=int(self.cfg.world.seed),
            tick=tick,
            source_flat=flat.copy(),
            source_y=y,
            source_x=x,
            decision_sequence=arrays["decision_sequence"][y, x].copy(),
            ow_id=arrays["pre_ow_id"][y, x].copy(),
            lineage_id=arrays["pre_lineage_id"][y, x].copy(),
            selected_action=arrays["selected_action"][y, x].copy(),
            selected_probability=arrays["selected_probability"][y, x].copy(),
            direction_fields=direction_fields,
            agent_context_fields=agent_context,
            oracle_context_fields=oracle_context,
            **selected_candidate,
        )
        if tuple(decisions.policy_legal.shape[1:]) != (22,):
            raise RuntimeError("factual v2 source must contain 22 candidates per decision")
        if tuple(decisions.direction_fields["action_direction_y"].shape[1:]) != (2, 8):
            raise RuntimeError("factual v2 source must contain 16 direction rows per decision")
        factual_evidence = {name: value.copy() for name, value in arrays.items()}
        evidence_bytes = sum(int(value.nbytes) for value in factual_evidence.values())
        self.sources.append(CollectedSource(source, decisions, factual_evidence))
        self.source_copy_bytes += source.nbytes + evidence_bytes
        self.source_copy_count += 1
        run.transfer_ledger.record_d2d(
            source.nbytes + evidence_bytes,
            kind="counterfactual_source",
            tick=tick,
            source_stream="counterfactual-source-copy",
            synchronization="event",
            scheduled=True,
            graph_compatible=False,
            reason="complete Phase 3 decision-boundary source capture",
        )

    def close(self) -> None:
        if self._source_stream is not None:
            self._source_stream.synchronize()

    def after_postdecision(self, run: Any) -> None:
        """Capture due factual anchor endpoints D2D for later independent hashing."""
        required = {int(item) for item in self.cfg.counterfactual.horizons}
        for values in self.cfg.counterfactual.family_horizons.values():
            required.update(int(item) for item in values)
        for collected in self.sources:
            offset = int(run.ds.tick) - int(collected.decisions.tick) + 1
            if offset not in required or offset in collected.factual_horizons:
                continue
            decision_event = run.streams.record(run.streams.compute)
            stream = self._stream(run)
            run.streams.wait(stream, decision_event)
            ready = run.streams.new_event()
            endpoint = capture_source_state(
                run.ds,
                f"{collected.state.source_state_id}:factual-h{offset}",
                stream=stream,
                ready_event=ready,
            )
            run.streams.wait(run.streams.compute, ready)
            collected.factual_horizons[offset] = endpoint
            self.source_copy_bytes += endpoint.nbytes
            run.transfer_ledger.record_d2d(
                endpoint.nbytes,
                kind="counterfactual_source",
                tick=int(run.ds.tick),
                source_stream="counterfactual-source-copy",
                synchronization="event",
                scheduled=True,
                graph_compatible=False,
                reason=f"factual selected-anchor endpoint H={offset}",
            )
