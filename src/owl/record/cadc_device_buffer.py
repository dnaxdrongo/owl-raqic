"""Recorder-owned backend arrays for factual CADC capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from owl.core.actions import Action
from owl.record.cadc_schema import (
    ABSENT_INT,
    CADC_SCHEMA_DIGEST,
    CADCEventCode,
    ContributionCode,
    schema_contract_for_config,
)

TRACKED_CONTRIBUTION_FIELDS: tuple[str, ...] = (
    "health",
    "resource",
    "food",
    "toxin",
    "waste",
    "integration",
    "boundary",
    "signal_emission",
)
CONTRIBUTION_CODES: tuple[int, ...] = tuple(int(code) for code in ContributionCode)
EVENT_CODES: tuple[int, ...] = tuple(int(code) for code in CADCEventCode)


@dataclass
class CADCDeviceBuffer:
    """Fixed-shape observation buffer that never owns scientific state."""

    xp: Any
    world_shape: tuple[int, int]
    channel_count: int
    arrays: dict[str, Any] = field(default_factory=dict)
    tick: int = 0
    stage_code: int = 0
    schema_digest: str = CADC_SCHEMA_DIGEST
    schema_version: str = "owl.cadc.factual.v1"
    event_capacity: int = 0
    registered_event_codes: tuple[int, ...] = EVENT_CODES
    registered_contribution_codes: tuple[int, ...] = CONTRIBUTION_CODES

    @classmethod
    def create(cls, ds: Any, cfg: Any) -> CADCDeviceBuffer:
        xp = ds.xp
        h, w = map(int, ds.health.shape)
        channels = int(ds.signal_reception.shape[-1])
        actions = len(Action)
        f = ds.health.dtype
        arrays: dict[str, Any] = {}
        action_transitions = bool(cfg.action_transitions.enabled)
        schema_version, schema_digest, event_codes, contribution_codes = (
            schema_contract_for_config(cfg)
        )

        def zeros(name: str, shape: tuple[int, ...], dtype: Any) -> None:
            arrays[name] = xp.zeros(shape, dtype=dtype)

        for name in (
            "pre_alive",
            "agent_food_pressure",
            "agent_toxin_pressure",
            "agent_crowding",
            "agent_novelty",
            "agent_hunger",
            "agent_pain",
            "agent_boundary_stress",
            "agent_social_need",
            "agent_memory",
            "agent_phase",
            "agent_health",
            "agent_resource",
            "agent_boundary",
            "agent_integration",
            "oracle_food",
            "oracle_toxin",
            "oracle_waste",
            "agent_threshold",
            "agent_activation",
            "agent_phase_coherence",
            "agent_sensed_food_mean",
            "agent_sensed_toxin_mean",
            "agent_sensed_alive_density",
        ):
            zeros(name, (h, w), f)
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
            zeros(f"agent_trait_{name}", (h, w), f)
        for name in ("agent_signal_reception", "agent_signal_memory", "oracle_signal"):
            zeros(name, (h, w, channels), f)

        zeros("pre_ow_id", (h, w), xp.int64)
        zeros("pre_lineage_id", (h, w), xp.int64)
        zeros("pre_parent_id", (h, w), xp.int64)
        zeros("pre_ow_type", (h, w), xp.int16)
        zeros("pre_age", (h, w), xp.int32)
        zeros("pre_development_stage", (h, w), xp.int16)
        zeros("oracle_occupancy", (h, w), xp.int64)
        zeros("oracle_obstacle", (h, w), bool)
        zeros("policy_legal", (h, w, actions), bool)
        zeros("candidate_executable", (h, w, actions), bool)
        zeros("candidate_reason_code", (h, w, actions), xp.int16)
        zeros("candidate_target_kind", (h, w, actions), xp.int8)
        zeros("candidate_proposed_y", (h, w, actions), xp.int32)
        zeros("candidate_proposed_x", (h, w, actions), xp.int32)
        zeros("candidate_resolved_y", (h, w, actions), xp.int32)
        zeros("candidate_resolved_x", (h, w, actions), xp.int32)
        zeros("candidate_target_ow_id", (h, w, actions), xp.int64)
        zeros("candidate_destination_occupancy", (h, w, actions), xp.int64)
        zeros("candidate_destination_obstacle", (h, w, actions), bool)
        zeros("candidate_destination_food", (h, w, actions), f)
        zeros("candidate_destination_toxin", (h, w, actions), f)
        zeros("candidate_opportunity_count", (h, w, actions), xp.int16)
        zeros("candidate_utility", (h, w, actions), f)
        zeros("agent_parent_intention", (h, w, actions), f)
        zeros("agent_prior_probability", (h, w, actions), f)
        zeros("decision_sequence", (h, w), xp.int64)
        if action_transitions:
            zeros("candidate_target_source", (h, w, actions), xp.int16)
            zeros("candidate_target_distance", (h, w, actions), f)
            zeros("candidate_target_confidence", (h, w, actions), f)
            zeros("candidate_compiled_action", (h, w, actions), xp.int16)
            arrays["candidate_compiled_action"].fill(ABSENT_INT)
            for name in (
                "agent_active_sense_food_memory",
                "agent_active_sense_toxin_memory",
                "agent_active_sense_alive_memory",
            ):
                zeros(name, (h, w), f)
            zeros("agent_active_sense_ttl", (h, w), xp.int32)
            for name, dtype in (
                ("action_target_y", xp.int32),
                ("action_target_x", xp.int32),
                ("action_target_ow_id", xp.int64),
                ("action_target_kind", xp.int16),
                ("action_target_source", xp.int16),
            ):
                zeros(name, (h, w, 2), dtype)
            zeros("action_target_distance", (h, w, 2), f)
            zeros("action_target_confidence", (h, w, 2), f)
            for name, dtype in (
                ("action_direction_y", xp.int32),
                ("action_direction_x", xp.int32),
                ("action_direction_executable", bool),
                ("action_direction_score", f),
                ("action_direction_distance_delta", f),
                ("action_direction_hazard", f),
                ("action_direction_opportunity", f),
            ):
                zeros(name, (h, w, 2, 8), dtype)
            for name in (
                "action_target_y",
                "action_target_x",
                "action_target_ow_id",
                "action_direction_y",
                "action_direction_x",
            ):
                arrays[name].fill(ABSENT_INT)
        for name in (
            "selected_action",
            "attempted_action",
            "realized_action",
            "execution_reason_code",
        ):
            zeros(name, (h, w), xp.int16)
            arrays[name].fill(ABSENT_INT)
        for name in (
            "selected_target_y",
            "selected_target_x",
            "realized_target_y",
            "realized_target_x",
            "current_y",
            "current_x",
        ):
            zeros(name, (h, w), xp.int32)
            arrays[name].fill(ABSENT_INT)
        for name in ("selected_target_ow_id", "realized_target_ow_id"):
            zeros(name, (h, w), xp.int64)
            arrays[name].fill(ABSENT_INT)
        zeros("selected_probability", (h, w), f)
        zeros("execution_success", (h, w), bool)
        if action_transitions:
            zeros("compiled_execution_action", (h, w), xp.int16)
            arrays["compiled_execution_action"].fill(ABSENT_INT)
            for name, dtype in (
                ("intent_target_y", xp.int32),
                ("intent_target_x", xp.int32),
                ("intent_target_ow_id", xp.int64),
                ("intent_target_kind", xp.int16),
                ("intent_target_source", xp.int16),
            ):
                zeros(name, (h, w), dtype)
                arrays[name].fill(ABSENT_INT)
            for name in (
                "intent_target_distance_before",
                "intent_target_distance_after",
                "intent_known_hazard_before",
                "intent_known_hazard_after",
                "intent_contact_opportunity_before",
                "intent_contact_opportunity_after",
            ):
                zeros(name, (h, w), f)
        zeros("information_active", (h, w), bool)
        zeros("information_kind", (h, w), xp.int8)
        zeros("information_pre_observation_ref", (h, w), xp.int64)
        zeros("information_post_memory_ref", (h, w), xp.int64)
        zeros("information_pre_signal_sum", (h, w), f)
        zeros("information_post_signal_memory_sum", (h, w), f)
        zeros("information_memory_delta", (h, w), f)
        zeros("information_followup_tick", (h, w), xp.int64)
        zeros("information_timing_code", (h, w), xp.int8)
        zeros("information_receiver_count", (h, w), xp.int32)
        zeros("information_receiver_link_status", (h, w), xp.int8)
        zeros("information_amount_received", (h, w), f)
        if action_transitions:
            zeros("information_new_cell_count", (h, w), xp.int32)
            zeros("information_new_target_count", (h, w), xp.int32)
            zeros("information_memory_changed", (h, w), bool)
            zeros("information_execution_success", (h, w), bool)
            zeros("information_no_new_information", (h, w), bool)
            for name in (
                "information_sensed_food_before",
                "information_sensed_food_after",
                "information_sensed_toxin_before",
                "information_sensed_toxin_after",
                "information_sensed_alive_before",
                "information_sensed_alive_after",
            ):
                zeros(name, (h, w), f)
        for name in (
            "information_observation_before",
            "information_memory_before",
            "information_memory_after",
            "information_emitted_channels",
            "information_received_channels",
        ):
            zeros(name, (h, w, channels), f)
        arrays["information_pre_observation_ref"].fill(ABSENT_INT)
        arrays["information_post_memory_ref"].fill(ABSENT_INT)
        arrays["information_followup_tick"].fill(ABSENT_INT)
        arrays["information_receiver_count"].fill(ABSENT_INT)
        for name in (
            "amount_consumed",
            "amount_transferred",
            "amount_repaired",
            "amount_damaged",
            "amount_emitted",
            "amount_received",
            "direct_cost",
        ):
            zeros(name, (h, w), f)
        zeros(
            "contribution_delta",
            (len(contribution_codes), h, w, len(TRACKED_CONTRIBUTION_FIELDS)),
            f,
        )
        zeros("stage_before", (h, w, len(TRACKED_CONTRIBUTION_FIELDS)), f)
        zeros("tick_start", (h, w, len(TRACKED_CONTRIBUTION_FIELDS)), f)
        zeros("tick_end", (h, w, len(TRACKED_CONTRIBUTION_FIELDS)), f)
        zeros("stage_before_parent_id", (h, w), xp.int64)
        event_shape = (len(event_codes), h * w)
        zeros("event_active", event_shape, bool)
        zeros("event_stage_code", event_shape, xp.int16)
        zeros("event_reason_code", event_shape, xp.int16)
        zeros("event_source_y", event_shape, xp.int32)
        zeros("event_source_x", event_shape, xp.int32)
        zeros("event_target_y", event_shape, xp.int32)
        zeros("event_target_x", event_shape, xp.int32)
        zeros("event_target_ow_id", event_shape, xp.int64)
        zeros("event_payload", (*event_shape, 4), f)
        zeros("event_count", (1,), xp.int64)
        zeros("event_overflow", (1,), xp.int64)
        for name in (
            "event_source_y",
            "event_source_x",
            "event_target_y",
            "event_target_x",
            "event_target_ow_id",
        ):
            arrays[name].fill(ABSENT_INT)
        arrays["decision_sequence"].fill(ABSENT_INT)
        arrays["pre_ow_id"].fill(ABSENT_INT)
        arrays["candidate_target_ow_id"].fill(ABSENT_INT)
        arrays["candidate_destination_occupancy"].fill(ABSENT_INT)
        if bool(cfg.recording.cadc.include_dense_context):
            radius = int(cfg.recording.cadc.exact_local_radius)
            local_cells = (2 * radius + 1) ** 2
            for name in (
                "food",
                "toxin",
                "waste",
                "health",
                "resource",
                "signal",
            ):
                width = local_cells * channels if name == "signal" else local_cells
                zeros(f"dense_oracle_{name}", (h, w, width), f)
            zeros("dense_oracle_occupancy", (h, w, local_cells), xp.int64)
            zeros("dense_oracle_obstacle", (h, w, local_cells), bool)
            arrays["dense_oracle_occupancy"].fill(ABSENT_INT)
        buffer = cls(
            xp=xp,
            world_shape=(h, w),
            channel_count=channels,
            arrays=arrays,
            schema_digest=schema_digest,
            schema_version=schema_version,
            event_capacity=int(cfg.recording.cadc.event_capacity_per_tick),
            registered_event_codes=event_codes,
            registered_contribution_codes=contribution_codes,
        )
        maximum = int(cfg.recording.cadc.max_device_buffer_bytes)
        if buffer.nbytes > maximum:
            raise MemoryError(
                f"CADC device buffer requires {buffer.nbytes:,} bytes, configured maximum is "
                f"{maximum:,}"
            )
        return buffer

    @property
    def nbytes(self) -> int:
        return sum(int(getattr(array, "nbytes", 0)) for array in self.arrays.values())

    def pointer_arrays(self) -> dict[str, Any]:
        """Expose recorder pointers for graph pointer-stability audits."""
        return self.arrays

    @property
    def contribution_codes(self) -> tuple[int, ...]:
        return self.registered_contribution_codes

    @property
    def contribution_fields(self) -> tuple[str, ...]:
        return TRACKED_CONTRIBUTION_FIELDS

    @property
    def event_codes(self) -> tuple[int, ...]:
        return self.registered_event_codes
