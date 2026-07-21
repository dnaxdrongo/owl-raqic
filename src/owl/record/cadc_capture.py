"""Backend-native, observational CADC capture helpers."""

from __future__ import annotations

from typing import Any

from owl.record.cadc_device_buffer import (
    TRACKED_CONTRIBUTION_FIELDS,
    CADCDeviceBuffer,
)
from owl.record.cadc_schema import (
    ABSENT_INT,
    CADCEventCode,
    CaptureStageCode,
    ContributionCode,
    ReasonCode,
)
from owl.science.action_contract import candidate_target_context


def _copy(buffer: CADCDeviceBuffer, name: str, value: Any) -> None:
    destination = buffer.arrays[name]
    destination[...] = value.astype(destination.dtype, copy=False)


def _local_tensor(field: Any, *, radius: int, mode: str, xp: Any, absent: Any) -> Any:
    """Assemble an exact local tensor on the active backend."""
    h, w = map(int, field.shape[:2])
    y, x = xp.indices((h, w), dtype=xp.int32)
    gathered = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            target_y = y + int(dy)
            target_x = x + int(dx)
            if mode == "toroidal":
                value = field[target_y % h, target_x % w]
            else:
                valid = (
                    (target_y >= 0)
                    & (target_y < h)
                    & (target_x >= 0)
                    & (target_x < w)
                )
                safe_y = xp.clip(target_y, 0, h - 1)
                safe_x = xp.clip(target_x, 0, w - 1)
                value = field[safe_y, safe_x]
                if mode != "reflective":
                    shape = (*valid.shape, *((1,) if field.ndim == 3 else ()))
                    value = xp.where(valid.reshape(shape), value, absent)
            gathered.append(value)
    stacked = xp.stack(tuple(gathered), axis=2)
    return stacked.reshape(h, w, -1)


def capture_agent_oracle_context(buffer: CADCDeviceBuffer, ds: Any, cfg: Any) -> None:
    """Freeze the exact utility sensing primitives and separate same-stage truth."""
    from owl.gpu.stages.utility_gpu import _drives

    xp = ds.xp
    drives, alive = _drives(ds, cfg)
    for name in (
        "food_pressure",
        "toxin_pressure",
        "crowding",
        "novelty",
        "hunger",
        "pain",
        "boundary_stress",
        "social_need",
    ):
        _copy(buffer, f"agent_{name}", drives[name])
    _copy(buffer, "pre_alive", alive)
    _copy(buffer, "agent_signal_reception", ds.signal_reception)
    _copy(buffer, "agent_signal_memory", ds.signal_memory)
    for source, destination in (
        ("food_mean", "agent_sensed_food_mean"),
        ("toxin_mean", "agent_sensed_toxin_mean"),
        ("alive_density", "agent_sensed_alive_density"),
    ):
        _copy(buffer, destination, ds.arrays[source])
    for name in ("memory", "phase", "health", "resource", "boundary", "integration"):
        _copy(buffer, f"agent_{name}", getattr(ds, name))

    _copy(buffer, "oracle_food", ds.food)
    _copy(buffer, "oracle_toxin", ds.toxin)
    _copy(buffer, "oracle_waste", ds.arrays.get("waste", xp.zeros_like(ds.health)))
    _copy(buffer, "oracle_signal", ds.signal)
    _copy(buffer, "oracle_occupancy", ds.occupancy)
    _copy(buffer, "oracle_obstacle", ds.obstacle)
    if "dense_oracle_food" in buffer.arrays:
        radius = int(cfg.recording.cadc.exact_local_radius)
        mode = str(cfg.world.boundary_mode)
        dense_sources = {
            "food": ds.food,
            "toxin": ds.toxin,
            "waste": ds.arrays.get("waste", xp.zeros_like(ds.health)),
            "health": ds.health,
            "resource": ds.resource,
            "signal": ds.signal,
            "occupancy": ds.occupancy,
            "obstacle": ds.obstacle,
        }
        for name, value in dense_sources.items():
            absent = ABSENT_INT if name == "occupancy" else False if name == "obstacle" else 0
            _copy(
                buffer,
                f"dense_oracle_{name}",
                _local_tensor(value, radius=radius, mode=mode, xp=xp, absent=absent),
            )
    received = xp.sum(ds.signal_reception, axis=-1) > 0
    _record_dense_event(
        buffer,
        CADCEventCode.SIGNAL_RECEPTION,
        received,
        stage=CaptureStageCode.POST_SENSING,
        payload0=xp.sum(ds.signal_reception, axis=-1),
    )
    buffer.tick = int(ds.tick)
    buffer.stage_code = int(CaptureStageCode.POST_SENSING)


def capture_tick_open(buffer: CADCDeviceBuffer, ds: Any) -> None:
    """Initialize source coordinates and the per-OW reconciliation baseline."""
    xp = ds.xp
    alive = (ds.health > 0.0) & (~ds.obstacle)
    source_y, source_x = xp.indices(ds.health.shape, dtype=xp.int32)
    _copy(buffer, "current_y", xp.where(alive, source_y, ABSENT_INT))
    _copy(buffer, "current_x", xp.where(alive, source_x, ABSENT_INT))
    capture_stage_before(buffer, ds)
    buffer.arrays["tick_start"][...] = buffer.arrays["stage_before"]
    buffer.arrays["contribution_delta"].fill(0)
    for name in (
        "event_active",
        "event_stage_code",
        "event_reason_code",
        "event_payload",
        "event_count",
        "event_overflow",
    ):
        buffer.arrays[name].fill(0)
    for name in (
        "event_source_y",
        "event_source_x",
        "event_target_y",
        "event_target_x",
        "event_target_ow_id",
    ):
        buffer.arrays[name].fill(ABSENT_INT)
    buffer.tick = int(ds.tick)
    buffer.stage_code = int(CaptureStageCode.TICK_OPEN)


def capture_prechoice_candidates(buffer: CADCDeviceBuffer, ds: Any, cfg: Any) -> None:
    """Freeze identity, policy mask, candidate targets and pre-choice reasons."""
    xp = ds.xp
    _action_transition_context_from_device = None
    if bool(cfg.action_transitions.enabled):
        from owl.gpu.stages.action_transitions_gpu import (
            action_transition_context_from_device,
        )

        _action_transition_context_from_device = action_transition_context_from_device
    alive = (ds.health > 0.0) & (~ds.obstacle)
    fallback_id = xp.arange(ds.health.size, dtype=xp.int64).reshape(ds.health.shape)
    ow_id = xp.where(ds.occupancy >= 0, ds.occupancy, fallback_id)
    _copy(buffer, "pre_ow_id", xp.where(alive, ow_id, -1))
    source_flat = xp.arange(ds.health.size, dtype=xp.int64).reshape(ds.health.shape)
    _copy(
        buffer,
        "decision_sequence",
        xp.where(alive, xp.int64(int(ds.tick) * int(ds.health.size)) + source_flat, -1),
    )
    source_y, source_x = xp.indices(ds.health.shape, dtype=xp.int32)
    _copy(buffer, "current_y", xp.where(alive, source_y, ABSENT_INT))
    _copy(buffer, "current_x", xp.where(alive, source_x, ABSENT_INT))
    _copy(buffer, "pre_lineage_id", ds.arrays.get("lineage_id", xp.full_like(ow_id, -1)))
    _copy(buffer, "pre_parent_id", ds.arrays.get("parent_id", xp.full_like(ow_id, -1)))
    _copy(buffer, "pre_ow_type", ds.arrays.get("ow_type", xp.zeros_like(ds.health, dtype=xp.int16)))
    _copy(buffer, "pre_age", ds.arrays.get("age", xp.zeros_like(ds.health, dtype=xp.int32)))
    _copy(
        buffer,
        "pre_development_stage",
        ds.arrays.get("development_stage", xp.zeros_like(ds.health, dtype=xp.int16)),
    )
    policy = ds.arrays.get("_policy_legal_bool", ds.arrays["_authority_bool"])
    _copy(buffer, "policy_legal", policy)
    _copy(buffer, "candidate_utility", ds.arrays["pre_utilities"])
    for name in (
        "memory",
        "phase",
        "health",
        "resource",
        "boundary",
        "integration",
        "threshold",
        "activation",
        "phase_coherence",
    ):
        destination = f"agent_{name}"
        if destination in buffer.arrays and name in ds.arrays:
            _copy(buffer, destination, ds.arrays[name])
    for name in (
        "mobility",
        "metabolism",
        "predation",
        "grazing",
        "cooperation",
        "aggression",
        "curiosity",
        "reproduction_rate",
        "toxin_resistance",
        "memory_capacity",
        "coupling_strength",
        "emit_strength",
        "emit_efficiency",
        "receive_sensitivity",
        "signal_precision",
        "honesty_bias",
        "deception_bias",
    ):
        _copy(buffer, f"agent_trait_{name}", ds.arrays[name])
    _copy(
        buffer,
        "agent_parent_intention",
        ds.arrays.get("raqic_parent_intention", xp.zeros_like(ds.pre_utilities)),
    )
    _copy(buffer, "agent_prior_probability", ds.possibility)
    context = candidate_target_context(
        ds.health,
        ds.resource,
        ds.obstacle,
        ds.occupancy,
        ds.food,
        ds.toxin,
        ds.parent_id,
        policy,
        boundary_mode=str(cfg.world.boundary_mode),
        diagonal_movement_enabled=bool(cfg.actions.diagonal_movement_enabled),
        xp=xp,
        action_transition_context=(
            _action_transition_context_from_device(ds)
            if bool(cfg.action_transitions.enabled)
            else None
        ),
        action_transition_config=cfg.action_transitions,
        movement_cost=float(cfg.resources.movement_cost),
    )
    mapping = {
        "candidate_target_kind": context.target_kind,
        "candidate_proposed_y": context.proposed_y,
        "candidate_proposed_x": context.proposed_x,
        "candidate_resolved_y": context.resolved_y,
        "candidate_resolved_x": context.resolved_x,
        "candidate_target_ow_id": context.target_ow_id,
        "candidate_destination_occupancy": context.destination_occupancy,
        "candidate_destination_obstacle": context.destination_obstacle,
        "candidate_destination_food": context.destination_food,
        "candidate_destination_toxin": context.destination_toxin,
        "candidate_opportunity_count": context.opportunity_count,
        "candidate_executable": context.executable,
        "candidate_reason_code": context.reason_code,
    }
    if bool(cfg.action_transitions.enabled):
        mapping.update(
            {
                "candidate_target_source": context.target_source,
                "candidate_target_distance": context.target_distance,
                "candidate_target_confidence": context.target_confidence,
                "candidate_compiled_action": context.compiled_action,
            }
        )
    for name, value in mapping.items():
        _copy(buffer, name, value)
    if bool(cfg.action_transitions.enabled):
        for name in (
            "active_sense_food_memory",
            "active_sense_toxin_memory",
            "active_sense_alive_memory",
            "active_sense_ttl",
        ):
            _copy(buffer, f"agent_{name}", ds.arrays[name])
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
        ):
            _copy(buffer, name, ds.arrays[name])
    buffer.tick = int(ds.tick)
    buffer.stage_code = int(CaptureStageCode.PRE_CHOICE)


def capture_selected_intent(buffer: CADCDeviceBuffer, ds: Any, cfg: Any) -> None:
    """Freeze selected policy intent after all decision backends have committed."""
    xp = ds.xp
    selected = ds.raqic_readout if "raqic_readout" in ds.arrays else ds.readout
    selected = selected.astype(xp.int16, copy=False)
    _copy(buffer, "selected_action", selected)
    probabilities = ds.arrays.get("raqic_probabilities", ds.possibility)
    picked = xp.take_along_axis(probabilities, selected[..., None], axis=-1)[..., 0]
    _copy(buffer, "selected_probability", picked)
    sy = xp.take_along_axis(
        buffer.arrays["candidate_resolved_y"], selected[..., None], axis=-1
    )[..., 0]
    sx = xp.take_along_axis(
        buffer.arrays["candidate_resolved_x"], selected[..., None], axis=-1
    )[..., 0]
    target_id = xp.take_along_axis(
        buffer.arrays["candidate_target_ow_id"], selected[..., None], axis=-1
    )[..., 0]
    _copy(buffer, "selected_target_y", sy)
    _copy(buffer, "selected_target_x", sx)
    _copy(buffer, "selected_target_ow_id", target_id)
    if bool(cfg.action_transitions.enabled):
        for name in (
            "information_new_cell_count",
            "information_new_target_count",
            "information_memory_changed",
            "information_execution_success",
            "information_no_new_information",
            "information_sensed_food_before",
            "information_sensed_food_after",
            "information_sensed_toxin_before",
            "information_sensed_toxin_after",
            "information_sensed_alive_before",
            "information_sensed_alive_after",
            "intent_target_distance_after",
            "intent_known_hazard_after",
            "intent_contact_opportunity_after",
        ):
            buffer.arrays[name].fill(0)
        compiled = xp.take_along_axis(
            buffer.arrays["candidate_compiled_action"], selected[..., None], axis=-1
        )[..., 0]
        _copy(buffer, "compiled_execution_action", compiled)
        family = xp.where(selected == 20, 0, xp.where(selected == 21, 1, 0)).astype(
            xp.int32
        )
        high_level = (selected == 20) | (selected == 21)
        family_index = family[..., None]
        for destination, source in (
            ("intent_target_y", "action_target_y"),
            ("intent_target_x", "action_target_x"),
            ("intent_target_ow_id", "action_target_ow_id"),
            ("intent_target_kind", "action_target_kind"),
            ("intent_target_source", "action_target_source"),
            ("intent_target_distance_before", "action_target_distance"),
        ):
            value = xp.take_along_axis(
                buffer.arrays[source], family_index, axis=-1
            )[..., 0]
            absent = 0 if destination == "intent_target_distance_before" else ABSENT_INT
            buffer.arrays[destination][...] = xp.where(high_level, value, absent)
        confidence = xp.take_along_axis(
            buffer.arrays["action_target_confidence"], family_index, axis=-1
        )[..., 0]
        buffer.arrays["intent_known_hazard_before"][...] = xp.where(
            selected == 20, confidence, 0
        )
        distance = buffer.arrays["intent_target_distance_before"]
        buffer.arrays["intent_contact_opportunity_before"][...] = xp.where(
            selected == 21, (distance <= 1).astype(ds.health.dtype), 0
        )
        for destination, source in (
            ("information_sensed_food_before", "active_sense_food_memory"),
            ("information_sensed_toxin_before", "active_sense_toxin_memory"),
            ("information_sensed_alive_before", "active_sense_alive_memory"),
        ):
            buffer.arrays[destination][...] = xp.where(
                selected == 1, ds.arrays[source], 0
            )
    for name in ("attempted_action", "realized_action"):
        buffer.arrays[name].fill(ABSENT_INT)
    for name in (
        "realized_target_y",
        "realized_target_x",
        "realized_target_ow_id",
        "execution_reason_code",
    ):
        buffer.arrays[name].fill(ABSENT_INT)
    buffer.arrays["execution_success"].fill(False)
    information = living = buffer.arrays["pre_alive"] > 0
    information = information & ((selected == 1) | (selected == 11))
    buffer.arrays["information_active"][...] = information
    buffer.arrays["information_kind"][...] = xp.where(information, selected, 0).astype(xp.int8)
    buffer.arrays["information_pre_observation_ref"][...] = xp.where(
        information, buffer.arrays["decision_sequence"], ABSENT_INT
    )
    buffer.arrays["information_post_memory_ref"].fill(ABSENT_INT)
    buffer.arrays["information_pre_signal_sum"][...] = xp.where(
        information, xp.sum(buffer.arrays["agent_signal_reception"], axis=-1), 0
    )
    buffer.arrays["information_post_signal_memory_sum"].fill(0)
    buffer.arrays["information_memory_delta"].fill(0)
    buffer.arrays["information_followup_tick"][...] = xp.where(
        information, int(ds.tick) + 1, ABSENT_INT
    )
    # Sensing is applied automatically before action selection. Communication emissions
    # are aggregated into a field, so sender-specific receiver provenance is
    # unavailable unless a future scientific contract supplies it.
    buffer.arrays["information_timing_code"][...] = xp.where(information, 1, 0)
    buffer.arrays["information_receiver_count"][...] = xp.where(
        information & (selected == 11), ABSENT_INT, 0
    )
    buffer.arrays["information_receiver_link_status"][...] = xp.where(
        information & (selected == 11), 2, xp.where(information, 1, 0)
    ).astype(xp.int8)
    for name in (
        "amount_consumed",
        "amount_transferred",
        "amount_repaired",
        "amount_damaged",
        "amount_emitted",
        "amount_received",
        "direct_cost",
    ):
        buffer.arrays[name].fill(0)
    passive = living & (selected == 0)
    buffer.arrays["attempted_action"][passive] = selected[passive]
    buffer.arrays["realized_action"][passive] = selected[passive]
    buffer.arrays["execution_success"][passive] = True
    buffer.arrays["execution_reason_code"][passive] = int(ReasonCode.NONE)
    if not bool(cfg.action_transitions.enabled):
        unsupported = living & ((selected == 1) | (selected == 20) | (selected == 21))
        buffer.arrays["execution_reason_code"][unsupported] = int(
            ReasonCode.NO_EXECUTION_CONTRACT
        )
    else:
        targeted = living & ((selected == 20) | (selected == 21))
        _record_dense_event(
            buffer,
            CADCEventCode.ACTION_TARGET_ACQUIRED,
            targeted,
            stage=CaptureStageCode.POST_SELECTION,
            target_y=buffer.arrays["intent_target_y"],
            target_x=buffer.arrays["intent_target_x"],
            target_ow_id=buffer.arrays["intent_target_ow_id"],
            payload0=buffer.arrays["intent_target_distance_before"],
            payload1=buffer.arrays["intent_known_hazard_before"],
            payload2=buffer.arrays["intent_contact_opportunity_before"],
        )
    buffer.arrays["information_observation_before"][...] = xp.where(
        information[..., None], buffer.arrays["agent_signal_reception"], 0
    )
    buffer.arrays["information_memory_before"][...] = xp.where(
        information[..., None], buffer.arrays["agent_signal_memory"], 0
    )
    buffer.stage_code = int(CaptureStageCode.POST_SELECTION)


def capture_information_post_memory(buffer: CADCDeviceBuffer, ds: Any) -> None:
    """Freeze SENSE/COMMUNICATE post-memory primitives without scoring them."""
    xp = ds.xp
    active = buffer.arrays["information_active"]
    y = xp.maximum(buffer.arrays["current_y"], 0)
    x = xp.maximum(buffer.arrays["current_x"], 0)
    post = xp.sum(ds.signal_memory[y, x, :], axis=-1)
    pre = xp.sum(buffer.arrays["agent_signal_memory"], axis=-1)
    buffer.arrays["information_post_signal_memory_sum"][...] = xp.where(active, post, 0)
    buffer.arrays["information_memory_delta"][...] = xp.where(active, post - pre, 0)
    memory_after = ds.signal_memory[y, x, :]
    received = buffer.arrays["agent_signal_reception"]
    selected = buffer.arrays["selected_action"]
    emitted = ds.signal_emission[y, x, :]
    buffer.arrays["information_memory_after"][...] = xp.where(
        active[..., None], memory_after, 0
    )
    buffer.arrays["information_received_channels"][...] = xp.where(
        active[..., None], received, 0
    )
    buffer.arrays["information_emitted_channels"][...] = xp.where(
        (active & (selected == 11))[..., None], emitted, 0
    )
    buffer.arrays["information_amount_received"][...] = xp.where(
        active, xp.sum(received, axis=-1), 0
    )
    buffer.arrays["information_post_memory_ref"][...] = xp.where(
        active, buffer.arrays["decision_sequence"], ABSENT_INT
    )


def capture_movement_execution(
    buffer: CADCDeviceBuffer, ds: Any, cfg: Any, plan: Any
) -> None:
    """Copy the already-computed MovementPlan into source-keyed evidence arrays."""
    xp = ds.xp
    if int(plan.mover_y.shape[0]) == 0:
        return
    sy, sx = plan.mover_y, plan.mover_x
    ty, tx = plan.target_y, plan.target_x
    selected = buffer.arrays["selected_action"][sy, sx]
    buffer.arrays["attempted_action"][sy, sx] = selected
    buffer.arrays["realized_target_y"][sy, sx] = ty
    buffer.arrays["realized_target_x"][sy, sx] = tx
    target_id = ds.occupancy[ty, tx]
    buffer.arrays["realized_target_ow_id"][sy, sx] = xp.where(target_id >= 0, target_id, -1)
    buffer.arrays["execution_success"][sy, sx] = plan.accepted
    buffer.arrays["realized_action"][sy, sx] = xp.where(plan.accepted, selected, ABSENT_INT)
    selected_reason = xp.take_along_axis(
        buffer.arrays["candidate_reason_code"][sy, sx, :], selected[:, None], axis=-1
    )[:, 0]
    reason = xp.where(
        plan.accepted,
        int(ReasonCode.NONE),
        xp.where(
            selected_reason != int(ReasonCode.NONE),
            selected_reason,
            int(ReasonCode.CONFLICT_LOST),
        ),
    )
    buffer.arrays["execution_reason_code"][sy, sx] = reason.astype(xp.int16)
    cost = xp.where(
        plan.accepted,
        float(cfg.resources.movement_cost),
        0.5 * float(cfg.resources.movement_cost),
    )
    buffer.arrays["direct_cost"][sy, sx] = cost.astype(ds.health.dtype)
    _record_sparse_event(
        buffer,
        CADCEventCode.MOVEMENT_ATTEMPT,
        sy,
        sx,
        ty,
        tx,
        stage=CaptureStageCode.MOVEMENT,
        reason=reason,
        payload0=cost,
    )
    success_idx = xp.nonzero(plan.accepted)[0]
    _record_sparse_event(
        buffer,
        CADCEventCode.MOVEMENT_SUCCESS,
        sy[success_idx],
        sx[success_idx],
        ty[success_idx],
        tx[success_idx],
        stage=CaptureStageCode.MOVEMENT,
    )
    failed_idx = xp.nonzero(~plan.accepted)[0]
    _record_sparse_event(
        buffer,
        CADCEventCode.MOVEMENT_REJECTION,
        sy[failed_idx],
        sx[failed_idx],
        ty[failed_idx],
        tx[failed_idx],
        stage=CaptureStageCode.MOVEMENT,
        reason=reason[failed_idx],
    )
    collision_idx = xp.nonzero(plan.collision)[0]
    _record_sparse_event(
        buffer,
        CADCEventCode.COLLISION,
        sy[collision_idx],
        sx[collision_idx],
        ty[collision_idx],
        tx[collision_idx],
        stage=CaptureStageCode.COLLISION_INHIBITION,
    )
    keep = xp.nonzero(plan.accepted)[0]
    if int(keep.shape[0]):
        ay, ax, by, bx = sy[keep], sx[keep], ty[keep], tx[keep]
        buffer.arrays["current_y"][ay, ax] = by
        buffer.arrays["current_x"][ay, ax] = bx


def capture_action_transition_execution(
    buffer: CADCDeviceBuffer, ds: Any, cfg: Any, active_sense: Any
) -> None:
    """Finalize v1 SENSE/FLEE/PURSUE evidence from authoritative results."""
    if not bool(cfg.action_transitions.enabled):
        return
    xp = ds.xp
    selected = buffer.arrays["selected_action"]
    living = buffer.arrays["pre_alive"] > 0
    high_level = living & ((selected == 20) | (selected == 21))
    missing = high_level & (buffer.arrays["attempted_action"] == ABSENT_INT)
    buffer.arrays["attempted_action"][missing] = selected[missing]
    selected_reason = xp.take_along_axis(
        buffer.arrays["candidate_reason_code"], selected[..., None], axis=-1
    )[..., 0]
    buffer.arrays["execution_reason_code"][missing] = xp.where(
        selected_reason[missing] == int(ReasonCode.NONE),
        int(ReasonCode.STAGE_NOT_ATTEMPTED),
        selected_reason[missing],
    ).astype(xp.int16)

    current_y = xp.maximum(buffer.arrays["current_y"], 0)
    current_x = xp.maximum(buffer.arrays["current_x"], 0)
    target_y = xp.maximum(buffer.arrays["intent_target_y"], 0)
    target_x = xp.maximum(buffer.arrays["intent_target_x"], 0)
    dy = xp.abs(current_y - target_y)
    dx = xp.abs(current_x - target_x)
    if str(cfg.world.boundary_mode) == "toroidal":
        h, w = buffer.world_shape
        dy = xp.minimum(dy, h - dy)
        dx = xp.minimum(dx, w - dx)
    distance_after = xp.maximum(dy, dx).astype(ds.health.dtype)
    valid_target = high_level & (buffer.arrays["intent_target_y"] >= 0)
    buffer.arrays["intent_target_distance_after"][...] = xp.where(
        valid_target, distance_after, 0
    )
    buffer.arrays["intent_known_hazard_after"][...] = xp.where(
        selected == 20, ds.toxin[current_y, current_x], 0
    )
    buffer.arrays["intent_contact_opportunity_after"][...] = xp.where(
        selected == 21, (distance_after <= 1).astype(ds.health.dtype), 0
    )

    sense = living & (selected == 1)
    attempted = sense & active_sense.attempted
    success = sense & active_sense.success
    buffer.arrays["attempted_action"][attempted] = 1
    buffer.arrays["realized_action"][success] = 1
    buffer.arrays["execution_success"][sense] = success[sense]
    reason = xp.where(
        success,
        xp.where(
            active_sense.no_new_information,
            int(ReasonCode.ACTIVE_SENSE_NO_NEW_INFORMATION),
            int(ReasonCode.NONE),
        ),
        xp.where(
            attempted,
            int(ReasonCode.INSUFFICIENT_RESOURCE),
            int(ReasonCode.STAGE_NOT_ATTEMPTED),
        ),
    ).astype(xp.int16)
    buffer.arrays["execution_reason_code"][sense] = reason[sense]
    buffer.arrays["realized_target_y"][success] = buffer.arrays["current_y"][success]
    buffer.arrays["realized_target_x"][success] = buffer.arrays["current_x"][success]
    buffer.arrays["realized_target_ow_id"][success] = buffer.arrays["pre_ow_id"][success]
    buffer.arrays["direct_cost"][sense] += active_sense.cost[sense]
    buffer.arrays["information_new_cell_count"][sense] = (
        active_sense.newly_observed_count[sense]
    )
    buffer.arrays["information_new_target_count"][sense] = (
        active_sense.newly_observed_target_count[sense]
    )
    buffer.arrays["information_memory_changed"][sense] = active_sense.memory_changed[sense]
    buffer.arrays["information_execution_success"][sense] = success[sense]
    buffer.arrays["information_no_new_information"][sense] = (
        active_sense.no_new_information[sense]
    )
    for destination, source in (
        ("information_sensed_food_after", "active_sense_food_memory"),
        ("information_sensed_toxin_after", "active_sense_toxin_memory"),
        ("information_sensed_alive_after", "active_sense_alive_memory"),
    ):
        buffer.arrays[destination][sense] = ds.arrays[source][sense]
    _record_dense_event(
        buffer,
        CADCEventCode.ACTIVE_SENSE_ATTEMPT,
        attempted,
        stage=CaptureStageCode.ACTIVE_SENSE,
        reason=reason,
        payload0=active_sense.cost,
        payload1=active_sense.newly_observed_count,
        payload2=active_sense.newly_observed_target_count,
    )
    _record_dense_event(
        buffer,
        CADCEventCode.ACTIVE_SENSE_SUCCESS,
        success,
        stage=CaptureStageCode.ACTIVE_SENSE,
        reason=reason,
        payload0=active_sense.cost,
        payload1=active_sense.newly_observed_count,
        payload2=active_sense.newly_observed_target_count,
        payload3=active_sense.memory_changed.astype(ds.health.dtype),
    )


def _gather_by_decision(buffer: CADCDeviceBuffer, ds: Any, name: str) -> Any:
    xp = ds.xp
    y = xp.maximum(buffer.arrays["current_y"], 0)
    x = xp.maximum(buffer.arrays["current_x"], 0)
    if name == "signal_emission":
        return xp.sum(ds.signal_emission[y, x, :], axis=-1)
    source = ds.arrays.get(name)
    if source is None:
        return xp.zeros(buffer.world_shape, dtype=ds.health.dtype)
    return source[y, x]


def _event_slot(buffer: CADCDeviceBuffer, code: CADCEventCode) -> int:
    return buffer.event_codes.index(int(code))


def _record_dense_event(
    buffer: CADCDeviceBuffer,
    code: CADCEventCode,
    mask: Any,
    *,
    stage: CaptureStageCode,
    target_y: Any | None = None,
    target_x: Any | None = None,
    target_ow_id: Any | None = None,
    reason: Any | None = None,
    payload0: Any | None = None,
    payload1: Any | None = None,
    payload2: Any | None = None,
    payload3: Any | None = None,
) -> None:
    xp = buffer.xp
    slot = _event_slot(buffer, code)
    h, w = buffer.world_shape
    source_y, source_x = xp.indices((h, w), dtype=xp.int32)
    flat_mask = mask.reshape(-1)
    buffer.arrays["event_active"][slot, ...] |= flat_mask
    buffer.arrays["event_source_y"][slot, ...] = xp.where(
        flat_mask, source_y.reshape(-1), buffer.arrays["event_source_y"][slot]
    )
    buffer.arrays["event_source_x"][slot, ...] = xp.where(
        flat_mask, source_x.reshape(-1), buffer.arrays["event_source_x"][slot]
    )
    buffer.arrays["event_stage_code"][slot, ...] = xp.where(
        flat_mask, int(stage), buffer.arrays["event_stage_code"][slot]
    )
    if target_y is not None:
        buffer.arrays["event_target_y"][slot, ...] = xp.where(
            flat_mask, target_y.reshape(-1), buffer.arrays["event_target_y"][slot]
        )
    if target_x is not None:
        buffer.arrays["event_target_x"][slot, ...] = xp.where(
            flat_mask, target_x.reshape(-1), buffer.arrays["event_target_x"][slot]
        )
    if target_ow_id is not None:
        buffer.arrays["event_target_ow_id"][slot, ...] = xp.where(
            flat_mask, target_ow_id.reshape(-1), buffer.arrays["event_target_ow_id"][slot]
        )
    if reason is not None:
        reason_value = reason.reshape(-1) if hasattr(reason, "shape") else reason
        buffer.arrays["event_reason_code"][slot, ...] = xp.where(
            flat_mask, reason_value, buffer.arrays["event_reason_code"][slot]
        )
    for payload_index, payload in enumerate((payload0, payload1, payload2, payload3)):
        if payload is not None:
            payload_value = payload.reshape(-1) if hasattr(payload, "shape") else payload
            buffer.arrays["event_payload"][slot, ..., payload_index] = xp.where(
                flat_mask,
                payload_value,
                buffer.arrays["event_payload"][slot, ..., payload_index],
            )


def _record_sparse_event(
    buffer: CADCDeviceBuffer,
    code: CADCEventCode,
    source_y: Any,
    source_x: Any,
    target_y: Any,
    target_x: Any,
    *,
    stage: CaptureStageCode,
    reason: Any | None = None,
    payload0: Any | None = None,
    target_ow_id: Any | None = None,
    slot_index: Any | None = None,
) -> None:
    slot = _event_slot(buffer, code)
    if int(source_y.shape[0]) == 0:
        return
    index = (
        source_y.astype(buffer.xp.int64) * int(buffer.world_shape[1]) + source_x
        if slot_index is None
        else slot_index
    )
    buffer.arrays["event_active"][slot, index] = True
    buffer.arrays["event_stage_code"][slot, index] = int(stage)
    buffer.arrays["event_source_y"][slot, index] = source_y
    buffer.arrays["event_source_x"][slot, index] = source_x
    buffer.arrays["event_target_y"][slot, index] = target_y
    buffer.arrays["event_target_x"][slot, index] = target_x
    if target_ow_id is None:
        valid = (
            (target_y >= 0)
            & (target_y < int(buffer.world_shape[0]))
            & (target_x >= 0)
            & (target_x < int(buffer.world_shape[1]))
        )
        safe_y = buffer.xp.clip(target_y, 0, int(buffer.world_shape[0]) - 1)
        safe_x = buffer.xp.clip(target_x, 0, int(buffer.world_shape[1]) - 1)
        target_ow_id = buffer.xp.where(
            valid, buffer.arrays["oracle_occupancy"][safe_y, safe_x], ABSENT_INT
        )
    buffer.arrays["event_target_ow_id"][slot, index] = target_ow_id
    if reason is not None:
        buffer.arrays["event_reason_code"][slot, index] = reason
    if payload0 is not None:
        buffer.arrays["event_payload"][slot, index, 0] = payload0


def capture_stage_before(buffer: CADCDeviceBuffer, ds: Any) -> None:
    """Snapshot tracked values at current OW coordinates before one stage."""
    for index, name in enumerate(TRACKED_CONTRIBUTION_FIELDS):
        buffer.arrays["stage_before"][..., index] = _gather_by_decision(buffer, ds, name)
    buffer.arrays["stage_before_parent_id"][...] = _gather_by_decision(buffer, ds, "parent_id")


def capture_stage_after(
    buffer: CADCDeviceBuffer,
    ds: Any,
    contribution: ContributionCode,
    *,
    actions: tuple[int, ...] = (),
) -> None:
    """Record one named stage delta and action-family execution classification."""
    xp = ds.xp
    slot = buffer.contribution_codes.index(int(contribution))
    after = xp.stack(
        tuple(_gather_by_decision(buffer, ds, name) for name in TRACKED_CONTRIBUTION_FIELDS),
        axis=-1,
    )
    delta = after - buffer.arrays["stage_before"]
    if contribution == ContributionCode.MOVEMENT:
        identity_slot = buffer.contribution_codes.index(int(ContributionCode.IDENTITY_TRANSFER))
        for index in (2, 3, 7):
            buffer.arrays["contribution_delta"][identity_slot, ..., index] = delta[..., index]
            delta[..., index] = 0
    living = buffer.arrays["pre_alive"] > 0
    buffer.arrays["contribution_delta"][slot, ...] = xp.where(
        living[..., None], delta, 0
    ).astype(ds.health.dtype)
    if not actions:
        return
    selected = buffer.arrays["selected_action"]
    owned = living & xp.isin(selected, xp.asarray(actions, dtype=selected.dtype))
    buffer.arrays["attempted_action"][owned] = selected[owned]
    health_delta = delta[..., 0]
    resource_delta = delta[..., 1]
    food_delta = delta[..., 2]
    integration_delta = delta[..., 5]
    boundary_delta = delta[..., 6]
    signal_delta = delta[..., 7]
    success = xp.zeros(buffer.world_shape, dtype=bool)
    for action in actions:
        mask = owned & (selected == int(action))
        if int(action) == 10:  # FEED
            action_success = (resource_delta > 0) | (food_delta < 0)
        elif int(action) == 11:  # COMMUNICATE
            action_success = signal_delta > 0
        elif int(action) == 12:  # INHIBIT
            action_success = (resource_delta < 0) | (signal_delta > 0)
        elif int(action) == 13:  # INTEGRATE
            action_success = (integration_delta > 0) | (boundary_delta > 0)
        elif int(action) == 14:  # REPAIR
            action_success = (health_delta > 0) | (boundary_delta > 0)
        elif int(action) == 15:  # REPRODUCE
            action_success = resource_delta < 0
        elif int(action) == 16:  # INGEST
            action_success = resource_delta > 0
        else:
            parent_after = _gather_by_decision(buffer, ds, "parent_id")
            action_success = (xp.any(delta != 0, axis=-1)) | (
                parent_after != buffer.arrays["stage_before_parent_id"]
            )
        success |= mask & action_success
        event_code = {
            10: CADCEventCode.FEEDING,
            11: CADCEventCode.SIGNAL_EMIT,
            12: CADCEventCode.INHIBITION,
            13: CADCEventCode.INTEGRATION,
            14: CADCEventCode.REPAIR,
            15: CADCEventCode.BIRTH,
            17: CADCEventCode.EXPULSION,
            18: CADCEventCode.SPLIT,
            19: CADCEventCode.MERGE,
        }.get(int(action))
        if event_code is not None:
            _record_dense_event(
                buffer,
                event_code,
                mask & action_success,
                stage={
                    10: CaptureStageCode.FEEDING,
                    11: CaptureStageCode.COMMUNICATION,
                    12: CaptureStageCode.COLLISION_INHIBITION,
                    13: CaptureStageCode.REPAIR_INTEGRATE,
                    14: CaptureStageCode.REPAIR_INTEGRATE,
                    15: CaptureStageCode.REPRODUCTION,
                    17: CaptureStageCode.TOPOLOGY,
                    18: CaptureStageCode.TOPOLOGY,
                    19: CaptureStageCode.TOPOLOGY,
                }[int(action)],
                payload0=xp.maximum(resource_delta, 0),
            )
    buffer.arrays["execution_success"][owned] = success[owned]
    buffer.arrays["realized_action"][owned] = xp.where(
        success[owned], selected[owned], ABSENT_INT
    )
    buffer.arrays["execution_reason_code"][owned] = xp.where(
        success[owned], int(ReasonCode.NONE), int(ReasonCode.NO_EFFECT)
    ).astype(xp.int16)
    buffer.arrays["amount_consumed"][owned] += xp.maximum(-food_delta[owned], 0)
    buffer.arrays["amount_transferred"][owned] += xp.maximum(resource_delta[owned], 0)
    buffer.arrays["amount_repaired"][owned] += xp.maximum(
        health_delta[owned], 0
    ) + xp.maximum(boundary_delta[owned], 0)
    buffer.arrays["amount_damaged"][owned] += xp.maximum(-health_delta[owned], 0)
    buffer.arrays["amount_emitted"][owned] += xp.maximum(signal_delta[owned], 0)
    buffer.arrays["direct_cost"][owned] += xp.maximum(-resource_delta[owned], 0)
    y = buffer.arrays["current_y"]
    x = buffer.arrays["current_x"]
    buffer.arrays["realized_target_y"][success] = y[success]
    buffer.arrays["realized_target_x"][success] = x[success]
    buffer.arrays["realized_target_ow_id"][success] = buffer.arrays["pre_ow_id"][success]


def capture_ingestion_execution(
    buffer: CADCDeviceBuffer,
    ds: Any,
    predator_y: Any,
    predator_x: Any,
    target_y: Any,
    target_x: Any,
    eligible: Any,
    success: Any,
    probability: Any,
    transfer: Any,
) -> None:
    """Record exact already-drawn ingestion outcomes from collision resolution."""
    xp = ds.xp
    idx = xp.nonzero(eligible)[0]
    if int(idx.shape[0]) == 0:
        return
    py, px, ty, tx = predator_y[idx], predator_x[idx], target_y[idx], target_x[idx]
    selected = buffer.arrays["selected_action"][py, px]
    buffer.arrays["attempted_action"][py, px] = selected
    ok = success[idx]
    buffer.arrays["execution_success"][py, px] = ok
    buffer.arrays["realized_action"][py, px] = xp.where(ok, selected, ABSENT_INT)
    buffer.arrays["execution_reason_code"][py, px] = xp.where(
        ok, int(ReasonCode.NONE), int(ReasonCode.STOCHASTIC_GATE_FAILED)
    )
    buffer.arrays["realized_target_y"][py, px] = ty
    buffer.arrays["realized_target_x"][py, px] = tx
    buffer.arrays["realized_target_ow_id"][py, px] = ds.occupancy[ty, tx]
    buffer.arrays["amount_transferred"][py, px] += xp.where(ok, transfer[idx], 0)
    _record_sparse_event(
        buffer,
        CADCEventCode.INGESTION,
        py,
        px,
        ty,
        tx,
        stage=CaptureStageCode.COLLISION_INHIBITION,
        reason=xp.where(ok, int(ReasonCode.NONE), int(ReasonCode.STOCHASTIC_GATE_FAILED)),
        payload0=probability[idx],
        target_ow_id=buffer.arrays["oracle_occupancy"][ty, tx],
        slot_index=idx,
    )


def capture_reproduction_execution(
    buffer: CADCDeviceBuffer, ds: Any, result: dict[str, Any]
) -> None:
    """Record the existing reproduction plan without recomputing its RNG draws."""
    xp = ds.xp
    plan = result.get("_cadc_plan")
    if plan is None or int(plan.parent_y.shape[0]) == 0:
        return
    py, px, ty, tx = plan.parent_y, plan.parent_x, plan.target_y, plan.target_x
    selected = buffer.arrays["selected_action"][py, px]
    buffer.arrays["attempted_action"][py, px] = selected
    buffer.arrays["execution_success"][py, px] = plan.accepted
    buffer.arrays["realized_action"][py, px] = xp.where(
        plan.accepted, selected, ABSENT_INT
    )
    reason = xp.where(
        plan.accepted,
        int(ReasonCode.NONE),
        xp.where(
            ~plan.has_target,
            int(ReasonCode.NO_TARGET),
            xp.where(
                ~plan.gate,
                int(ReasonCode.STOCHASTIC_GATE_FAILED),
                int(ReasonCode.CONFLICT_LOST),
            ),
        ),
    )
    buffer.arrays["execution_reason_code"][py, px] = reason.astype(xp.int16)
    buffer.arrays["realized_target_y"][py, px] = ty
    buffer.arrays["realized_target_x"][py, px] = tx
    target_id = ds.occupancy[ty, tx]
    buffer.arrays["realized_target_ow_id"][py, px] = xp.where(
        plan.accepted, target_id, ABSENT_INT
    )
    accepted = xp.nonzero(plan.accepted)[0]
    if int(accepted.shape[0]):
        ay, ax, by, bx = py[accepted], px[accepted], ty[accepted], tx[accepted]
        child_resource = ds.resource[by, bx]
        buffer.arrays["amount_transferred"][ay, ax] += child_resource
        _record_sparse_event(
            buffer,
            CADCEventCode.BIRTH,
            ay,
            ax,
            by,
            bx,
            stage=CaptureStageCode.REPRODUCTION,
            payload0=child_resource,
            target_ow_id=ds.occupancy[by, bx],
        )


def capture_topology_execution(buffer: CADCDeviceBuffer, ds: Any, events: Any) -> None:
    """Record accepted entries from the existing bounded topology buffer."""
    xp = ds.xp
    active = events.active & events.accepted
    idx = xp.nonzero(active)[0]
    if int(idx.shape[0]) == 0:
        return
    sy, sx = events.source_y[idx], events.source_x[idx]
    ty, tx = events.target_y[idx], events.target_x[idx]
    event_type = events.event_type[idx]
    selected = buffer.arrays["selected_action"][sy, sx]
    buffer.arrays["attempted_action"][sy, sx] = selected
    buffer.arrays["realized_action"][sy, sx] = selected
    buffer.arrays["execution_success"][sy, sx] = True
    buffer.arrays["execution_reason_code"][sy, sx] = int(ReasonCode.NONE)
    buffer.arrays["realized_target_y"][sy, sx] = ty
    buffer.arrays["realized_target_x"][sy, sx] = tx
    buffer.arrays["realized_target_ow_id"][sy, sx] = ds.occupancy[ty, tx]
    from owl.gpu.stages.topology_gpu import TOPOLOGY_EXPEL, TOPOLOGY_MERGE, TOPOLOGY_SPLIT

    for event_value, event_code in (
        (TOPOLOGY_MERGE, CADCEventCode.MERGE),
        (TOPOLOGY_SPLIT, CADCEventCode.SPLIT),
        (TOPOLOGY_EXPEL, CADCEventCode.EXPULSION),
    ):
        keep = xp.nonzero(event_type == int(event_value))[0]
        _record_sparse_event(
            buffer,
            event_code,
            sy[keep],
            sx[keep],
            ty[keep],
            tx[keep],
            stage=CaptureStageCode.TOPOLOGY,
            payload0=events.priority[idx][keep],
            target_ow_id=ds.occupancy[ty[keep], tx[keep]],
        )


def capture_death_event(buffer: CADCDeviceBuffer, ds: Any, dead: Any) -> None:
    """Freeze identity and coordinates before death cleanup clears them."""
    xp = ds.xp
    target_y = xp.maximum(buffer.arrays["current_y"], 0)
    target_x = xp.maximum(buffer.arrays["current_x"], 0)
    dead_by_decision = dead[target_y, target_x]
    target_ow_id = ds.occupancy[target_y, target_x]
    target_health = ds.health[target_y, target_x]
    starvation_debt = ds.arrays.get("starvation_debt", xp.zeros_like(ds.health))[
        target_y, target_x
    ]
    starvation_criterion = (starvation_debt >= 1.0) & (target_health <= 0.05)
    boundary_failure = ds.boundary[target_y, target_x] <= 0
    integration_failure = ds.integration[target_y, target_x] < 0
    criterion_count = (
        (target_health <= 0).astype(xp.int8)
        + starvation_criterion.astype(xp.int8)
        + boundary_failure.astype(xp.int8)
        + integration_failure.astype(xp.int8)
    )
    reason = xp.where(
        dead_by_decision & (criterion_count == 1),
        int(ReasonCode.NONE),
        int(ReasonCode.CAUSE_AMBIGUOUS),
    )
    _record_dense_event(
        buffer,
        CADCEventCode.DEATH,
        dead_by_decision,
        stage=CaptureStageCode.DEATH,
        target_y=target_y,
        target_x=target_x,
        target_ow_id=target_ow_id,
        reason=reason,
        payload0=target_health,
        payload1=starvation_debt,
        payload2=ds.boundary[target_y, target_x],
        payload3=ds.integration[target_y, target_x],
    )


def capture_damage_evidence(
    buffer: CADCDeviceBuffer,
    ds: Any,
    starvation_health_damage: Any,
    toxin_health_damage: Any,
) -> None:
    """Record the exact already-computed metabolism damage components."""
    xp = ds.xp
    y = xp.maximum(buffer.arrays["current_y"], 0)
    x = xp.maximum(buffer.arrays["current_x"], 0)
    living = buffer.arrays["pre_alive"] > 0
    starvation = starvation_health_damage[y, x]
    toxin = toxin_health_damage[y, x]
    target_id = ds.occupancy[y, x]
    _record_dense_event(
        buffer,
        CADCEventCode.STARVATION_EVIDENCE,
        living & (starvation > 0),
        stage=CaptureStageCode.METABOLISM_DAMAGE,
        target_y=y,
        target_x=x,
        target_ow_id=target_id,
        payload0=starvation,
    )
    _record_dense_event(
        buffer,
        CADCEventCode.TOXIN_DAMAGE_EVIDENCE,
        living & (toxin > 0),
        stage=CaptureStageCode.METABOLISM_DAMAGE,
        target_y=y,
        target_x=x,
        target_ow_id=target_id,
        payload0=toxin,
    )


def finalize_tick_reconciliation(buffer: CADCDeviceBuffer, ds: Any) -> None:
    """Store the explicitly named residual after all tracked stage channels."""
    xp = ds.xp
    after = xp.stack(
        tuple(_gather_by_decision(buffer, ds, name) for name in TRACKED_CONTRIBUTION_FIELDS),
        axis=-1,
    )
    buffer.arrays["tick_end"][...] = after
    named = xp.sum(buffer.arrays["contribution_delta"], axis=0)
    residual = after - buffer.arrays["tick_start"] - named
    slot = buffer.contribution_codes.index(int(ContributionCode.RESIDUAL))
    living_at_choice = buffer.arrays["pre_alive"] > 0
    buffer.arrays["contribution_delta"][slot, ...] = xp.where(
        living_at_choice[..., None], residual, 0
    ).astype(ds.health.dtype)
    unclassified = living_at_choice & (buffer.arrays["execution_reason_code"] < 0)
    buffer.arrays["execution_reason_code"][unclassified] = int(ReasonCode.STAGE_NOT_ATTEMPTED)
    count = xp.sum(buffer.arrays["event_active"], dtype=xp.int64)
    buffer.arrays["event_count"][0] = count
    buffer.arrays["event_overflow"][0] = xp.maximum(count - int(buffer.event_capacity), 0)
    buffer.stage_code = int(CaptureStageCode.TICK_COMMIT)
