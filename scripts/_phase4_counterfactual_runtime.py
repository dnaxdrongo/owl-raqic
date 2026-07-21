"""Provide performance adapters around the certified counterfactual engine.

Per-tick packets and horizon snapshots remain on the branch stream until each
rollout completes. The adapters do not change actions, random streams, state
transitions, outcomes, events, contributions, hashes, or recorder contracts.
"""

from __future__ import annotations

import copy
import hashlib
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.counterfactual.evidence import (  # noqa: E402
    BranchEvidencePacket,
    capture_branch_evidence,
    transfer_branch_evidence,
)
from owl.counterfactual.forcing import (  # noqa: E402
    ForcedActionBatch,
    build_forced_action_batch,
    inject_forced_actions,
)
from owl.counterfactual.outcomes import (  # noqa: E402
    OutcomeDevicePacket,
    capture_outcome_device,
    transfer_outcome,
)
from owl.counterfactual.scheduler import BranchResult, CounterfactualScheduler  # noqa: E402
from owl.counterfactual.schema import STATE_HASH_VERSION, BranchStatus, branch_id  # noqa: E402
from owl.counterfactual.state_clone import (  # noqa: E402
    CounterfactualSourceState,
    build_clone_manifest,
    ordered_array_groups,
)
from owl.counterfactual.state_hash import (  # noqa: E402
    StateHashResult,
    _array_leaf,
    _canonical_json_with_transfer,
    _lp,
    _metadata_subset,
    compare_state_science,
    differing_leaves,
    hash_state,
)
from owl.record.cadc_capture import capture_selected_intent  # noqa: E402


def _capture_horizon_state(run: Any, source_state_id: str) -> CounterfactualSourceState:
    """Snapshot complete state on the current branch stream without host transfer.

    The counterfactual source-capture helper performs a quadratic pointer-pair audit on
    every capture. Branch isolation was already proved when the mutable branch
    was created; horizon snapshots use distinct backend ``copy`` allocations
    and are checked against the reference path by parity tests.
    """

    ds = run.ds
    source_manifest = build_clone_manifest(ds, require_phase25=True)
    # ``OWLDeviceState`` has no manifest, so canonical hashing intentionally
    # uses state_hash._metadata_subset's fixed live-state allow-list. Keep the
    # complete scientific metadata payload for cloning/comparison, but make
    # the snapshot hash use that same allow-list instead of adding
    # ``defer_host_metrics`` only because this object has a clone manifest.
    manifest = replace(source_manifest, metadata_names=())
    with run.streams.compute:
        arrays = {name: value.copy() for name, value in ds.arrays.items()}
        patch_arrays = {name: value.copy() for name, value in ds.patch_arrays.items()}
        global_arrays = {name: value.copy() for name, value in ds.global_arrays.items()}
    return CounterfactualSourceState(
        source_state_id=source_state_id,
        backend=ds.backend,
        arrays=arrays,
        patch_arrays=patch_arrays,
        global_arrays=global_arrays,
        scalars=copy.deepcopy(ds.scalars),
        metadata={
            name: copy.deepcopy(ds.metadata[name])
            for name in source_manifest.metadata_names
        },
        manifest=manifest,
    )


def _outside_allowed_array_changes(
    current: Any,
    source: Any,
    allowed: frozenset[str],
) -> tuple[str, ...]:
    """Verify unchanged clone leaves on-device with one compact synchronization."""

    current_groups = dict(ordered_array_groups(current))
    source_groups = dict(ordered_array_groups(source))
    failures: list[str] = []
    pending: list[tuple[str, Any]] = []
    xp = current.backend.xp
    for group in sorted(set(current_groups) | set(source_groups)):
        current_mapping = current_groups.get(group, {})
        source_mapping = source_groups.get(group, {})
        for name in sorted(set(current_mapping) | set(source_mapping)):
            key = f"{group}.{name}"
            if key in allowed:
                continue
            if name not in current_mapping or name not in source_mapping:
                failures.append(key)
                continue
            left = current_mapping[name]
            right = source_mapping[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                failures.append(key)
                continue
            left_bytes = left.reshape(-1).view(xp.uint8)
            right_bytes = right.reshape(-1).view(xp.uint8)
            pending.append((key, xp.all(left_bytes == right_bytes)))
    if pending:
        flags = current.backend.asnumpy(xp.stack([value for _, value in pending]))
        failures.extend(
            key for (key, _), equal in zip(pending, flags, strict=True) if not bool(equal)
        )
    return tuple(sorted(failures))


def _rehash_forced_state(
    current: Any,
    source: Any,
    baseline: StateHashResult,
    *,
    allowed_changed_leaves: frozenset[str],
    chunk_bytes: int = 4 * 1024**2,
) -> StateHashResult:
    """Rebuild the exact canonical root while transferring only allowed leaves.

    The high-level force seam may modify only readout arrays. Every other array
    is compared bit-for-bit on the active backend, while scalar and registered
    metadata leaves are recomputed canonically. Reusing their already-proved
    leaf digests removes a complete-state D2H transfer for every branch without
    weakening unexpected-mutation detection.
    """

    unexpected = _outside_allowed_array_changes(
        current,
        source,
        allowed_changed_leaves,
    )
    if unexpected:
        raise RuntimeError(
            "forced action mutated fields outside the registered seam: "
            f"{unexpected}"
        )
    baseline_leaves = dict(baseline.leaf_hashes)
    leaves: list[tuple[str, str]] = []
    total = 0
    device_to_host = 0
    for group, mapping in ordered_array_groups(current):
        for name, value in mapping.items():
            key = f"{group}.{name}"
            total += int(value.nbytes)
            if key in allowed_changed_leaves:
                leaf = _array_leaf(group, name, value, chunk_bytes)
                if value.__class__.__module__.startswith("cupy"):
                    device_to_host += int(value.nbytes)
            else:
                if key not in baseline_leaves:
                    raise RuntimeError(f"baseline state hash is missing leaf {key}")
                leaf = baseline_leaves[key]
            leaves.append((key, leaf))
    for group, value in (
        ("scalars", current.scalars),
        ("metadata", _metadata_subset(current)),
    ):
        encoded, transferred = _canonical_json_with_transfer(value)
        device_to_host += transferred
        leaf = hashlib.sha256(_lp(group.encode()) + _lp(encoded)).hexdigest()
        if baseline_leaves.get(group) != leaf:
            raise RuntimeError(
                f"forced action mutated {group} outside the registered seam"
            )
        leaves.append((group, leaf))
    root = hashlib.sha256(_lp(STATE_HASH_VERSION.encode()))
    for name, leaf in leaves:
        root.update(_lp(name.encode()))
        root.update(_lp(bytes.fromhex(leaf)))
    return StateHashResult(
        algorithm=STATE_HASH_VERSION,
        root=root.hexdigest(),
        leaf_hashes=tuple(leaves),
        array_bytes=total,
        device_to_host_bytes=device_to_host,
    )


class DeferredTransferCounterfactualScheduler(CounterfactualScheduler):
    """Preserve counterfactual results while reducing per-tick CUDA synchronization."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._source_pre_hashes: dict[str, StateHashResult] = {}
        self._factual_horizon_hashes: dict[tuple[str, int], StateHashResult] = {}

    def _cached_source_pre_hash(self, source_id: str, state: Any) -> StateHashResult:
        with self._ledger_lock:
            cached = self._source_pre_hashes.get(source_id)
            if cached is None:
                cached = hash_state(state)
                self._source_pre_hashes[source_id] = cached
            return cached

    def _cached_factual_horizon_hash(
        self,
        source_id: str,
        horizon: int,
        state: Any,
    ) -> StateHashResult:
        key = (source_id, int(horizon))
        with self._ledger_lock:
            cached = self._factual_horizon_hashes.get(key)
            if cached is None:
                cached = hash_state(state)
                self._factual_horizon_hashes[key] = cached
            return cached

    def _execute_branch(
        self,
        collected: Any,
        decision_index: int,
        decision_id: str,
        forced_action: int,
        repeat_index: int,
        seed: int,
        *,
        anchor: bool,
    ) -> BranchResult:
        decisions = collected.decisions
        bid = branch_id(
            collected.state.source_state_id,
            decision_id,
            repeat_index,
            forced_action,
            seed,
            "anchor" if anchor else "paired",
        )
        run = self._branch_context(collected, seed)
        forced: ForcedActionBatch = build_forced_action_batch(
            decisions, [decision_index], [forced_action]
        )
        pre = self._cached_source_pre_hash(collected.state.source_state_id, run.ds)
        result = BranchResult(
            branch_id=bid,
            source_state_id=collected.state.source_state_id,
            source_decision_id=decision_id,
            repeat_index=repeat_index,
            branch_seed=seed,
            factual_action=int(
                self.template_run.ds.backend.asnumpy(forced.factual_action)[0]
            ),
            forced_action=forced_action,
            anchor=anchor,
            status=BranchStatus.RUNNING,
            pre_force_hash=pre,
            post_force_hash=pre,
        )
        started = perf_counter()
        device_evidence: list[BranchEvidencePacket] = []
        device_outcomes: dict[int, OutcomeDevicePacket] = {}
        horizon_states: dict[int, CounterfactualSourceState] = {}
        pending_device_bytes = 0
        max_pending_bytes = int(self.cfg.counterfactual.max_pending_bytes)

        def reserve_pending(nbytes: int) -> None:
            nonlocal pending_device_bytes
            pending_device_bytes += int(nbytes)
            if pending_device_bytes > max_pending_bytes:
                raise MemoryError(
                    "deferred branch evidence exceeds max_pending_bytes: "
                    f"{pending_device_bytes} > {max_pending_bytes}"
                )

        try:
            valid = inject_forced_actions(run.ds, forced)
            result.validation_passed = bool(run.ds.backend.asnumpy(valid)[0])
            if not result.validation_passed:
                raise RuntimeError("forced action failed source validation")
            allowed_force_leaves = frozenset(
                {"arrays.readout", "arrays.raqic_readout"}
            )
            result.post_force_hash = _rehash_forced_state(
                run.ds,
                collected.state,
                pre,
                allowed_changed_leaves=allowed_force_leaves,
            )
            result.force_changed_leaves = differing_leaves(pre, result.post_force_hash)
            if not set(result.force_changed_leaves) <= allowed_force_leaves:
                raise RuntimeError(
                    "forced action mutated fields outside the registered seam: "
                    f"{result.force_changed_leaves}"
                )
            capture_selected_intent(run.cadc_buffer, run.ds, run.cfg)
            horizons = self._horizons(forced_action)
            first_death = run.ds.xp.asarray(-1, dtype=run.ds.xp.int64)
            y = forced.source_y[0]
            x = forced.source_x[0]
            source_baseline = {
                "coordinate_y": y.copy(),
                "coordinate_x": x.copy(),
                "health": run.ds.health[y, x].copy(),
                "resource": run.ds.resource[y, x].copy(),
                "boundary": run.ds.boundary[y, x].copy(),
                "integration": run.ds.integration[y, x].copy(),
                "memory": run.ds.memory[y, x].copy(),
            }

            def capture_completed(offset: int) -> None:
                nonlocal first_death
                ow_id = forced.ow_id[0]
                alive = run.ds.xp.any(
                    (run.ds.occupancy == ow_id)
                    & (run.ds.health > 0.0)
                    & (~run.ds.obstacle)
                )
                first_death = run.ds.xp.where(
                    (first_death < 0) & (~alive), int(run.ds.tick), first_death
                )
                evidence = capture_branch_evidence(run.cadc_buffer)
                device_evidence.append(evidence)
                reserve_pending(evidence.nbytes)
                if offset not in horizons:
                    return
                outcome = capture_outcome_device(
                    run.ds,
                    forced,
                    horizon=offset,
                    source_tick=decisions.tick,
                    first_death_tick=first_death,
                    source_baseline=source_baseline,
                )
                device_outcomes[offset] = outcome
                reserve_pending(outcome.nbytes)
                snapshot = _capture_horizon_state(
                    run,
                    f"{bid}:horizon:{offset}",
                )
                horizon_states[offset] = snapshot
                reserve_pending(snapshot.nbytes)

            with run.streams.compute:
                run._segment_actions()
                run._segment_postdecision()
                capture_completed(1)
                for offset in range(2, max(horizons) + 1):
                    run._step_device([])
                    capture_completed(offset)

            # One synchronization replaces the reference path's per-tick sync.
            run.streams.compute.synchronize()
            for packet in device_evidence:
                host = transfer_branch_evidence(run.ds.backend, packet)
                result.evidence.append(host)
                if run.ds.is_gpu:
                    run.transfer_ledger.record_d2h(
                        host.nbytes,
                        kind="counterfactual_event",
                        tick=host.tick,
                        source_stream="counterfactual-branch-lane",
                        synchronization="deferred_branch_end",
                        scheduled=True,
                        graph_compatible=False,
                        reason="bounded deferred branch event/contribution packet",
                    )
            for offset in horizons:
                result.outcomes[offset] = transfer_outcome(
                    run.ds.backend, device_outcomes[offset]
                )
                snapshot = horizon_states[offset]
                result.horizon_hashes[offset] = hash_state(snapshot)
                if run.ds.is_gpu:
                    run.transfer_ledger.record_d2h(
                        result.outcomes[offset].nbytes,
                        kind="counterfactual_outcome",
                        tick=int(snapshot.scalars["tick"]),
                        source_stream="counterfactual-branch-lane",
                        synchronization="deferred_branch_end",
                        scheduled=True,
                        graph_compatible=False,
                        reason="compact deferred branch horizon outcome vector",
                    )
                    run.transfer_ledger.record_d2h(
                        result.horizon_hashes[offset].device_to_host_bytes,
                        kind="counterfactual_hash",
                        tick=int(snapshot.scalars["tick"]),
                        source_stream="counterfactual-hash",
                        synchronization="deferred_branch_end",
                        scheduled=True,
                        graph_compatible=False,
                        reason="canonical complete-state Merkle leaves",
                    )
                if anchor and offset in collected.factual_horizons:
                    factual_endpoint = collected.factual_horizons[offset]
                    result.anchor_exact_hash_matches[offset] = (
                        result.horizon_hashes[offset].root
                        == self._cached_factual_horizon_hash(
                            collected.state.source_state_id,
                            offset,
                            factual_endpoint,
                        ).root
                    )
                    result.anchor_matches[offset] = compare_state_science(
                        snapshot, factual_endpoint
                    ).passed
            if (
                anchor
                and self.cfg.counterfactual.require_anchor_equivalence
                and not all(
                    result.anchor_matches.get(horizon, False)
                    for horizon in result.horizon_hashes
                    if horizon in collected.factual_horizons
                )
            ):
                raise AssertionError("selected anchor differs from factual remainder")
            if any(bool(packet.event_overflow[0]) for packet in result.evidence):
                raise OverflowError("branch event buffer overflow")
            result.status = BranchStatus.COMPLETED
        except Exception as exc:
            result.status = BranchStatus.FAILED
            result.failure = f"{type(exc).__name__}: {exc}"
            result.failure_traceback = tuple(traceback.format_exc().splitlines())
        finally:
            result.runtime_seconds = perf_counter() - started
            run.streams.synchronize_all()
            with self._ledger_lock:
                for record in run.transfer_ledger.records:
                    self.transfer_ledger.record(
                        kind=record.kind,
                        direction=record.direction,
                        tick=record.tick,
                        nbytes=record.bytes,
                        source_stream=record.source_stream,
                        synchronization=record.synchronization,
                        scheduled=record.scheduled,
                        graph_compatible=record.graph_compatible,
                        reason=record.reason,
                    )
            horizon_states.clear()
            device_evidence.clear()
            device_outcomes.clear()
            run.close(checkpoint=False)
        return result
