"""State containers for the array-first Observer-Window Life engine.

The core implementation rule is: model observer windows conceptually, compute them as
dense arrays. Cell-level state lives in :class:`WorldState`; patch/global state is
aggregated into :class:`PatchState` and :class:`GlobalState`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.integer]
BoolArray = npt.NDArray[np.bool_]


@dataclass(slots=True)
class PatchState:
    """Aggregated patch-level observer-window arrays.

    Shape convention:
    - scalar patch fields: ``(patch_height, patch_width)``
    - patch possibility and policy bias: ``(patch_height, patch_width, num_actions)``
    - patch signal pressure: ``(patch_height, patch_width, num_channels)``
    """

    activation: FloatArray
    memory: FloatArray
    phase: FloatArray
    possibility: FloatArray
    integration: FloatArray
    resource: FloatArray
    health: FloatArray
    boundary: FloatArray
    signal_pressure: FloatArray
    synchrony: FloatArray
    coherence: FloatArray
    cross_scale: FloatArray
    intention: IntArray
    policy_bias: FloatArray
    crisis: float = 0.0
    carrying_pressure: float = 0.0
    starvation_pressure: float = 0.0
    food_deficit: float = 0.0
    intention_scores: FloatArray | None = None
    centroid_y: FloatArray | None = None
    centroid_x: FloatArray | None = None
    velocity_y: FloatArray | None = None
    velocity_x: FloatArray | None = None
    prediction_error: FloatArray | None = None
    alive_density: FloatArray | None = None
    food_mean: FloatArray | None = None
    starvation_debt_mean: FloatArray | None = None
    reproduction_fraction: FloatArray | None = None
    movement_fraction: FloatArray | None = None
    feed_fraction: FloatArray | None = None
    death_pressure: FloatArray | None = None
    patch_crisis: FloatArray | None = None
    patch_carrying_pressure: FloatArray | None = None
    noetic_B: FloatArray | None = None
    noetic_M: FloatArray | None = None
    noetic_P: FloatArray | None = None
    noetic_C: FloatArray | None = None
    noetic_K: FloatArray | None = None
    noetic_Theta: FloatArray | None = None
    noetic_N: FloatArray | None = None


@dataclass(slots=True)
class GlobalState:
    """Apex/global observer-window summary variables.

    The global state is a compact summary, not a controller that overwrites cell
    readouts. Later top-down code may use ``policy_bias`` as a weak logit bias.
    """

    integration: float
    readout: int
    intention: int
    fragmentation: float
    diversity: float
    complexity: float
    signal_pressure: FloatArray
    policy_bias: FloatArray
    crisis: float = 0.0
    carrying_pressure: float = 0.0
    starvation_pressure: float = 0.0
    food_deficit: float = 0.0
    intention_scores: FloatArray | None = None


@dataclass(slots=True)
class EventRecord:
    """Sparse event record used by topology and collision handlers.

    Events are intentionally sparse Python records. Dense cell dynamics stay in
    NumPy arrays; rare topology changes may be processed as event records.
    """

    kind: str
    tick: int
    source: tuple[int, int] | None = None
    target: tuple[int, int] | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OWRecord:
    """Optional sparse/mobile multi-cell observer-window record.

    This is reserved for later multi-cell or object-level OWs. It is not used
    for the ordinary cell hot loop.
    """

    id: int
    type_id: int
    pos_y: int
    pos_x: int
    occupied_cells: list[tuple[int, int]]
    parent_id: int | None
    children: list[int]
    traits: FloatArray
    alive: bool
    genome: FloatArray | None = None
    resource: float = 0.0
    health: float = 1.0
    boundary: float = 1.0


@dataclass(slots=True)
class WorldState:
    """Dense arrays for cell-level physical, possibility, and fractal layers.

    Shape convention:
    - most cell fields: ``(height, width)``
    - action/possibility fields: ``(height, width, num_actions)``
    - communication-channel fields: ``(height, width, num_channels)``

    The physical survival layer is represented by resource, health, boundary,
    food, toxin, obstacle, and occupancy. The possibility layer is represented
    by ``possibility`` and ``readout``. The fractal/mosaic layer is represented
    by phase, integration, patch state, and global state.
    """

    activation: FloatArray
    memory: FloatArray
    phase: FloatArray
    threshold: FloatArray
    readout: IntArray
    integration: FloatArray
    resource: FloatArray
    health: FloatArray
    boundary: FloatArray
    age: IntArray
    ow_type: IntArray
    lineage_id: IntArray
    parent_id: IntArray

    possibility: FloatArray

    signal: FloatArray
    signal_emission: FloatArray
    signal_reception: FloatArray
    signal_memory: FloatArray
    channel_receptivity: FloatArray
    channel_emission_bias: FloatArray
    channel_trust_local: FloatArray

    food: FloatArray
    toxin: FloatArray
    obstacle: BoolArray
    occupancy: IntArray
    noise: FloatArray

    mobility: FloatArray
    metabolism: FloatArray
    predation: FloatArray
    grazing: FloatArray
    cooperation: FloatArray
    aggression: FloatArray
    curiosity: FloatArray
    reproduction_rate: FloatArray
    toxin_resistance: FloatArray
    memory_capacity: FloatArray
    coupling_strength: FloatArray

    emit_strength: FloatArray
    emit_efficiency: FloatArray
    receive_sensitivity: FloatArray
    signal_precision: FloatArray
    honesty_bias: FloatArray
    deception_bias: FloatArray

    patches: PatchState
    global_state: GlobalState

    event_queue: list[EventRecord] = field(default_factory=list)
    mobile_ows: dict[int, OWRecord] = field(default_factory=dict)
    tick: int = 0

    # Advanced-build optional dense fields. These remain optional so alternate
    # tests/snapshots/manual WorldState constructors keep working. Runtime code
    # calls ``owl.core.advanced.ensure_advanced_fields`` before using them.
    digestion: FloatArray | None = None
    waste: FloatArray | None = None
    age_stress: FloatArray | None = None
    last_intake: FloatArray | None = None
    last_death_mask: BoolArray | None = None
    last_utilities: FloatArray | None = None
    last_logits: FloatArray | None = None
    last_action_probabilities: FloatArray | None = None
    action_cooldown: FloatArray | None = None
    signal_source_id: IntArray | None = None
    neighbor_trust: FloatArray | None = None
    deception_memory: FloatArray | None = None
    source_confidence: FloatArray | None = None
    phase_frequency: FloatArray | None = None
    phase_lag: FloatArray | None = None
    same_scale_weight: FloatArray | None = None
    parent_weight: FloatArray | None = None
    prediction_error: FloatArray | None = None
    alive_density: FloatArray | None = None
    food_mean: FloatArray | None = None
    starvation_debt_mean: FloatArray | None = None
    reproduction_fraction: FloatArray | None = None
    movement_fraction: FloatArray | None = None
    feed_fraction: FloatArray | None = None
    death_pressure: FloatArray | None = None
    patch_crisis: FloatArray | None = None
    patch_carrying_pressure: FloatArray | None = None
    starvation_debt: FloatArray | None = None
    last_movement_action: IntArray | None = None
    movement_loop_score: FloatArray | None = None
    genome: FloatArray | None = None
    development_stage: FloatArray | None = None
    symbiosis: FloatArray | None = None
    next_ow_id: int = 1
    pre_resource: FloatArray | None = None
    pre_health: FloatArray | None = None
    pre_food: FloatArray | None = None
    pre_starvation_debt: FloatArray | None = None
    pre_authority: FloatArray | None = None
    pre_utilities: FloatArray | None = None
    pre_parent_bias: FloatArray | None = None
    last_decision_urgency: FloatArray | None = None
    last_survival_value: FloatArray | None = None
    last_homeostatic_error: FloatArray | None = None
    last_macro_probabilities: FloatArray | None = None
    last_chosen_macro: IntArray | None = None
    noetic_B: FloatArray | None = None
    noetic_M: FloatArray | None = None
    noetic_P: FloatArray | None = None
    noetic_C: FloatArray | None = None
    noetic_K: FloatArray | None = None
    noetic_Theta: FloatArray | None = None
    noetic_N: FloatArray | None = None
    # Action-transition state is allocated only when its explicit contract is enabled.
    # Baseline runs leave these arrays unallocated.
    active_sense_food_memory: FloatArray | None = None
    active_sense_toxin_memory: FloatArray | None = None
    active_sense_alive_memory: FloatArray | None = None
    active_sense_ttl: IntArray | None = None
    active_sense_new_cell_count: IntArray | None = None
    active_sense_new_target_count: IntArray | None = None
    action_target_y: IntArray | None = None
    action_target_x: IntArray | None = None
    action_target_ow_id: IntArray | None = None
    action_target_kind: IntArray | None = None
    action_target_source: IntArray | None = None
    action_target_distance: FloatArray | None = None
    action_target_confidence: FloatArray | None = None
    action_direction_y: IntArray | None = None
    action_direction_x: IntArray | None = None
    action_direction_executable: BoolArray | None = None
    action_direction_score: FloatArray | None = None
    action_direction_distance_delta: FloatArray | None = None
    action_direction_hazard: FloatArray | None = None
    action_direction_opportunity: FloatArray | None = None
    flee_compiled_action: IntArray | None = None
    pursue_compiled_action: IntArray | None = None
    compiled_execution_action: IntArray | None = None
    # RAQIC integration optional dense fields. These are allocated only when
    # ``owl.raqic.state.ensure_raqic_fields`` is called and otherwise remain
    # ``None`` preserves baseline behavior when the optional feature is disabled.
    raqic_probabilities: FloatArray | None = None
    raqic_readout: IntArray | None = None
    raqic_record_action: IntArray | None = None
    raqic_record_readout: IntArray | None = None
    raqic_record_confidence: FloatArray | None = None
    raqic_score: FloatArray | None = None
    raqic_phase: FloatArray | None = None
    raqic_parent_intention: FloatArray | None = None
    raqic_audit_flags: IntArray | None = None
    raqic_trace_error: FloatArray | None = None
    raqic_min_eigenvalue: FloatArray | None = None
    raqic_backend_code: IntArray | None = None
    raqic_legacy_shadow_possibility: FloatArray | None = None
    raqic_legacy_shadow_readout: IntArray | None = None
    raqic_compare_l1: FloatArray | None = None
    raqic_compare_kl: FloatArray | None = None
    raqic_debug_density_diag: FloatArray | None = None
    raqic_patch_intention: FloatArray | None = None
    raqic_patch_record_aggregate: FloatArray | None = None
    raqic_patch_confidence: FloatArray | None = None
    raqic_global_intention: FloatArray | None = None
    raqic_global_record_aggregate: FloatArray | None = None
    # Optional actualization context and diagnostics. Phase context
    # is coordinate-resident and derived only from prior-tick RAQIC records.
    raqic_patch_action_phase: FloatArray | None = None
    raqic_patch_action_coherence: FloatArray | None = None
    raqic_global_action_phase: FloatArray | None = None
    raqic_global_action_coherence: FloatArray | None = None
    raqic_parent_action_phase: FloatArray | None = None
    raqic_parent_action_coherence: FloatArray | None = None
    raqic_pre_mixer_probabilities: FloatArray | None = None
    raqic_utility_innovation: FloatArray | None = None
    raqic_phase_alignment: FloatArray | None = None
    raqic_resonant_parent_intention: FloatArray | None = None
    raqic_interference_delta_l1: FloatArray | None = None
    raqic_policy_kl: FloatArray | None = None
    raqic_utility_projection_fraction: FloatArray | None = None
    raqic_utility_score_cosine: FloatArray | None = None
    raqic_utility_orthogonality_residual: FloatArray | None = None
    raqic_utility_innovation_norm: FloatArray | None = None
    raqic_interference_norm_error: FloatArray | None = None
    raqic_interference_illegal_mass: FloatArray | None = None
    raqic_shadow_probabilities: FloatArray | None = None
    raqic_shadow_readout: IntArray | None = None
    global_crisis: float = 0.0
    global_carrying_pressure: float = 0.0


def field_shape(state: WorldState) -> tuple[int, int]:
    """Return the cell-grid shape as ``(height, width)``.

    The authoritative grid shape is the health field, because a cell that has no
    health cannot be alive. Shape mismatches are checked in  initialization
    and later invariant tests.
    """
    shape = state.health.shape
    if len(shape) != 2:
        raise ValueError(f"state.health must be two-dimensional, got shape {shape}")
    return int(shape[0]), int(shape[1])


def action_shape(state: WorldState) -> tuple[int, int, int]:
    """Return the action-probability tensor shape.

    Expected shape is ``(height, width, num_actions)``.
    """
    shape = state.possibility.shape
    if len(shape) != 3:
        raise ValueError(f"state.possibility must be three-dimensional, got shape {shape}")
    return int(shape[0]), int(shape[1]), int(shape[2])


def channel_shape(state: WorldState) -> tuple[int, int, int]:
    """Return the communication-channel tensor shape.

    Expected shape is ``(height, width, num_channels)``.
    """
    shape = state.signal.shape
    if len(shape) != 3:
        raise ValueError(f"state.signal must be three-dimensional, got shape {shape}")
    return int(shape[0]), int(shape[1]), int(shape[2])


def clone_for_baseline(state: WorldState) -> WorldState:
    """Create an independent deep copy for comparative experiment conditions.

    All dense NumPy arrays are copied. Sparse event/mobile records are copied
    recursively so later experiments can branch from the same initial condition
    without sharing mutable state.
    """
    return copy.deepcopy(state)
