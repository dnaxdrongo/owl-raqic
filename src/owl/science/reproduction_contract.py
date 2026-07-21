"""Backend-neutral deterministic reproduction transition.

The scalar CPU path and NumPy/CuPy array paths call this same equation-level
transition.  It preserves the biological inheritance rules while replacing
order-dependent generator consumption with a versioned counter-RNG contract.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from owl.core.actions import Action
from owl.core.traits import TRAIT_FIELD_NAMES
from owl.gpu.field_registry import FIELD_REGISTRY
from owl.science.action_contract import reproduction_plan
from owl_raqic.random_contract import RNGStream, normal01


@dataclass(frozen=True)
class ReproductionDiagnostics:
    candidates: int
    accepted: int
    child_ids: tuple[int, ...]
    parents: tuple[tuple[int, int], ...] = ()
    targets: tuple[tuple[int, int], ...] = ()
    plan: Any | None = None
    accepted_indices: Any | None = None


def _array(mapping: MutableMapping[str, Any], name: str) -> Any:
    return mapping.get(name)


def apply_reproduction_arrays(
    arrays: MutableMapping[str, Any],
    scalars: MutableMapping[str, Any],
    cfg: Any,
    *,
    tick: int,
    xp: Any,
    patch_shape: tuple[int, int],
) -> ReproductionDiagnostics:
    """Apply one simultaneous reproduction transition to array mappings."""
    if not bool(cfg.reproduction.enabled):
        return ReproductionDiagnostics(0, 0, ())

    required = (
        "readout",
        "health",
        "resource",
        "boundary",
        "integration",
        "reproduction_rate",
        "obstacle",
        "occupancy",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError(f"reproduction contract missing arrays: {missing}")

    plan = reproduction_plan(
        arrays["readout"],
        arrays["health"],
        arrays["resource"],
        arrays["boundary"],
        arrays["integration"],
        arrays["reproduction_rate"],
        arrays["obstacle"],
        arrays["occupancy"],
        min_resource=float(cfg.reproduction.min_resource),
        min_health=float(cfg.reproduction.min_health),
        min_boundary=float(cfg.reproduction.min_boundary),
        min_integration=float(cfg.reproduction.min_integration),
        boundary_mode=str(cfg.world.boundary_mode),
        seed=int(cfg.world.seed),
        tick=int(tick),
        xp=xp,
    )
    candidates = int(plan.parent_y.shape[0])
    keep = xp.nonzero(plan.accepted)[0]
    accepted = int(keep.shape[0])
    if accepted == 0:
        return ReproductionDiagnostics(
            candidates,
            0,
            (),
            plan=plan,
            accepted_indices=keep,
        )

    py = plan.parent_y[keep]
    px = plan.parent_x[keep]
    cy = plan.target_y[keep]
    cx = plan.target_x[keep]
    parent_ow = plan.parent_ow_id[keep]

    # Snapshot and copy all persistent cell-owned inheritance fields before any
    # mutation. Decision/readout arrays are initialized separately below.
    for name, spec in FIELD_REGISTRY.items():
        if not spec.copy_on_reproduction or name not in arrays:
            continue
        arr = arrays[name]
        if getattr(arr, "ndim", 0) < 2 or tuple(arr.shape[:2]) != tuple(arrays["health"].shape):
            continue
        values = arr[py, px, ...].copy()
        arr[cy, cx, ...] = values

    resource = arrays["resource"]
    parent_resource = xp.clip(
        resource[py, px].astype(xp.float64), 0.0, float(cfg.resources.max_resource)
    )
    child_resource = float(cfg.reproduction.child_resource_fraction) * parent_resource
    resource[py, px] = xp.clip(
        parent_resource - child_resource, 0.0, float(cfg.resources.max_resource)
    ).astype(resource.dtype)
    resource[cy, cx] = xp.clip(child_resource, 0.0, float(cfg.resources.max_resource)).astype(
        resource.dtype
    )

    # Core newborn equations, matching the scientific CPU model.
    if "activation" in arrays:
        arrays["activation"][cy, cx] = (0.5 * arrays["activation"][py, px]).astype(
            arrays["activation"].dtype
        )
    if "memory" in arrays:
        arrays["memory"][cy, cx] = xp.clip(
            float(cfg.reproduction.memory_inheritance) * arrays["memory"][py, px], 0.0, 1.0
        ).astype(arrays["memory"].dtype)
    if "phase" in arrays:
        phase_noise = normal01(
            int(cfg.world.seed),
            int(tick),
            parent_ow,
            RNGStream.REPRODUCTION_TIE,
            10,
            xp=xp,
            dtype=xp.float64,
        ) * float(cfg.phase.phase_noise_sigma)
        arrays["phase"][cy, cx] = xp.mod(
            arrays["phase"][py, px].astype(xp.float64) + phase_noise,
            2.0 * xp.pi,
        ).astype(arrays["phase"].dtype)
    if "threshold" in arrays:
        threshold_noise = normal01(
            int(cfg.world.seed),
            int(tick),
            parent_ow,
            RNGStream.REPRODUCTION_TIE,
            11,
            xp=xp,
            dtype=xp.float64,
        ) * (0.5 * float(cfg.reproduction.mutation_sigma))
        arrays["threshold"][cy, cx] = xp.clip(
            arrays["threshold"][py, px].astype(xp.float64) + threshold_noise, 0.0, 1.0
        ).astype(arrays["threshold"].dtype)
    if "integration" in arrays:
        arrays["integration"][cy, cx] = xp.clip(
            0.5 * arrays["integration"][py, px], 0.0, 1.0
        ).astype(arrays["integration"].dtype)
    arrays["health"][cy, cx] = xp.asarray(
        float(cfg.reproduction.initial_child_health), dtype=arrays["health"].dtype
    )
    arrays["boundary"][cy, cx] = xp.asarray(
        float(cfg.reproduction.initial_child_boundary), dtype=arrays["boundary"].dtype
    )
    if "age" in arrays:
        arrays["age"][cy, cx] = 0

    # Mutate scalar and channel traits with stable slots. The result is
    # invariant to parent iteration order, chunking, and device partition.
    sigma = float(cfg.reproduction.mutation_sigma)
    for index, name in enumerate(TRAIT_FIELD_NAMES):
        arr = _array(arrays, name)
        if arr is None:
            continue
        noise = (
            normal01(
                int(cfg.world.seed),
                int(tick),
                parent_ow,
                RNGStream.REPRODUCTION_TIE,
                100 + index,
                xp=xp,
                dtype=xp.float64,
            )
            * sigma
        )
        arr[cy, cx] = xp.clip(arr[py, px].astype(xp.float64) + noise, 0.0, 1.0).astype(arr.dtype)

    channel_sigma = float(cfg.reproduction.channel_mutation_sigma)
    for field_index, name in enumerate(
        ("channel_emission_bias", "channel_receptivity", "channel_trust_local")
    ):
        arr = _array(arrays, name)
        if arr is None:
            continue
        effective_sigma = (
            min(channel_sigma, 0.02) if name == "channel_trust_local" else channel_sigma
        )
        channels = int(arr.shape[-1])
        for channel in range(channels):
            noise = (
                normal01(
                    int(cfg.world.seed),
                    int(tick),
                    parent_ow,
                    RNGStream.REPRODUCTION_TIE,
                    1000 + field_index * 256 + channel,
                    xp=xp,
                    dtype=xp.float64,
                )
                * effective_sigma
            )
            arr[cy, cx, channel] = xp.clip(
                arr[py, px, channel].astype(xp.float64) + noise, 0.0, 1.0
            ).astype(arr.dtype)

    if "signal_emission" in arrays:
        arrays["signal_emission"][cy, cx, ...] = 0.0
    if "signal_reception" in arrays:
        arrays["signal_reception"][cy, cx, ...] = 0.0
    if "signal_memory" in arrays:
        arrays["signal_memory"][cy, cx, ...] = xp.clip(
            float(cfg.reproduction.memory_inheritance) * arrays["signal_memory"][py, px, ...],
            0.0,
            1.0,
        ).astype(arrays["signal_memory"].dtype)

    # Globally unique identity in the single-process contract. Distributed
    # execution replaces this scalar range with its certified global allocator.
    occupancy = arrays["occupancy"]
    if getattr(xp, "__name__", "") == "numpy":
        max_seen = int(xp.max(occupancy)) if occupancy.size else 0
        start = max(int(scalars.get("next_ow_id", 1)), max_seen + 1, 1)
    else:
        # Persistent GPU construction initializes next_ow_id above every live
        # identity. Trust that authoritative scalar so births do not introduce
        # a device scalar synchronization in the tick hot path.
        start = max(int(scalars.get("next_ow_id", 1)), 1)
    child_ids = xp.arange(start, start + accepted, dtype=occupancy.dtype)
    occupancy[cy, cx] = child_ids
    scalars["next_ow_id"] = start + accepted

    if "lineage_id" in arrays:
        parent_lineage = arrays["lineage_id"][py, px]
        arrays["lineage_id"][cy, cx] = xp.where(
            parent_lineage >= 0, parent_lineage, parent_ow.astype(parent_lineage.dtype)
        )
    if "parent_id" in arrays:
        ph, pw = patch_shape
        h, w = arrays["health"].shape
        psy = max(1, h // int(ph))
        psx = max(1, w // int(pw))
        arrays["parent_id"][cy, cx] = (cy // psy) * int(pw) + (cx // psx)

    rest = int(Action.REST)
    for name in (
        "readout",
        "raqic_readout",
        "raqic_record_action",
        "raqic_record_readout",
        "raqic_legacy_shadow_readout",
    ):
        arr = _array(arrays, name)
        if arr is not None:
            arr[cy, cx] = rest
    for name in (
        "possibility",
        "raqic_probabilities",
        "raqic_parent_intention",
        "raqic_legacy_shadow_possibility",
        "last_action_probabilities",
        "last_macro_probabilities",
    ):
        arr = _array(arrays, name)
        if arr is not None:
            arr[cy, cx, ...] = 0.0
            if arr.shape[-1] > rest:
                arr[cy, cx, rest] = 1.0
    for name in ("raqic_score", "raqic_phase"):
        arr = _array(arrays, name)
        if arr is not None:
            arr[cy, cx, ...] = 0.0
    for name in ("starvation_debt", "movement_loop_score"):
        arr = _array(arrays, name)
        if arr is not None:
            arr[cy, cx] = 0.0
    if "last_movement_action" in arrays:
        arrays["last_movement_action"][cy, cx] = rest
    for name in (
        "active_sense_food_memory",
        "active_sense_toxin_memory",
        "active_sense_alive_memory",
        "active_sense_ttl",
        "active_sense_new_cell_count",
        "active_sense_new_target_count",
        "action_target_distance",
        "action_target_confidence",
        "action_direction_executable",
        "action_direction_score",
        "action_direction_distance_delta",
        "action_direction_hazard",
        "action_direction_opportunity",
    ):
        arr = _array(arrays, name)
        if arr is not None:
            arr[cy, cx, ...] = 0
    for name in (
        "flee_compiled_action",
        "pursue_compiled_action",
        "compiled_execution_action",
        "action_target_y",
        "action_target_x",
        "action_target_ow_id",
        "action_target_kind",
        "action_target_source",
        "action_direction_y",
        "action_direction_x",
    ):
        arr = _array(arrays, name)
        if arr is not None:
            arr[cy, cx, ...] = -1

    if getattr(xp, "__name__", "") == "numpy":
        ids_np = tuple(int(value) for value in xp.asarray(child_ids))
        parents = tuple(
            (int(y), int(x)) for y, x in zip(py, px, strict=True)
        )
        targets = tuple(
            (int(y), int(x)) for y, x in zip(cy, cx, strict=True)
        )
    else:
        # Device-native factual evidence is emitted from ``plan`` and ``keep``
        # at the packet boundary. Sparse host tuples support reference validation only.
        ids_np = ()
        parents = ()
        targets = ()
    return ReproductionDiagnostics(
        candidates,
        accepted,
        ids_np,
        parents,
        targets,
        plan,
        keep,
    )
