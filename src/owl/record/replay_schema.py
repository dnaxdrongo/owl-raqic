"""Define explicit column schemas for replay materialization.

The scientific engine owns the values.  This module only classifies existing
arrays and compiles deterministic Arrow schemas for bounded columnar writers.
PyArrow is imported lazily so importing the scientific core does not require
recording extras.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

REPLAY_COLUMNAR_SCHEMA_VERSION = "owl.replay.columnar.v1"
ROW_ORDER = "tick,cell_flat_c,action_index"

STATE_FIELDS: tuple[str, ...] = (
    "health",
    "resource",
    "toxin",
    "food",
    "waste",
    "integration",
    "boundary",
    "age",
    "ow_type",
    "lineage_id",
    "parent_id",
    "development_stage",
    "starvation_debt",
    "readout",
    "raqic_readout",
    "raqic_record_confidence",
)

WORLD_ACTION_FIELDS: tuple[str, ...] = (
    "last_utilities",
    "pre_utilities",
    "last_logits",
    "last_action_probabilities",
    "possibility",
    "raqic_score",
    "raqic_utility_innovation",
    "raqic_resonant_parent_intention",
    "raqic_pre_mixer_probabilities",
    "raqic_probabilities",
    "raqic_phase",
    "raqic_phase_alignment",
    "raqic_parent_intention",
    "raqic_shadow_probabilities",
    "raqic_parent_action_phase",
    "raqic_parent_action_coherence",
)

WORLD_SCALAR_DIAGNOSTICS: tuple[str, ...] = (
    "raqic_utility_innovation_norm",
    "raqic_utility_projection_fraction",
    "raqic_utility_score_cosine",
    "raqic_utility_orthogonality_residual",
    "raqic_policy_kl",
    "raqic_interference_delta_l1",
    "raqic_interference_norm_error",
    "raqic_interference_illegal_mass",
)

PATCH_ACTION_FIELDS: tuple[str, ...] = (
    "raqic_patch_action_phase",
    "raqic_patch_action_coherence",
)

GLOBAL_ACTION_FIELDS: tuple[str, ...] = (
    "raqic_global_action_phase",
    "raqic_global_action_coherence",
)

SELECTED_DECISION_FIELDS: dict[str, str] = {
    "last_utilities": "selected_utility",
    "pre_utilities": "selected_pre_utility",
    "raqic_score": "selected_raqic_score",
    "raqic_probabilities": "selected_final_probability",
    "raqic_pre_mixer_probabilities": "selected_pre_mixer_probability",
    "raqic_phase": "selected_phase",
    "raqic_utility_innovation": "selected_utility_innovation",
    "raqic_resonant_parent_intention": "selected_resonance_contribution",
    "raqic_shadow_probabilities": "selected_shadow_probability",
}


class ReplayShapeClass(StrEnum):
    RUN_SCALAR = "run_scalar"
    WORLD_IDENTITY = "world_identity"
    WORLD_SCALAR = "world_scalar"
    WORLD_ACTION = "world_action"
    PATCH_ACTION = "patch_action"
    GLOBAL_ACTION = "global_action"
    DERIVED = "derived"


@dataclass(frozen=True)
class ReplayColumnSpec:
    name: str
    source_field: str | None
    shape_class: ReplayShapeClass
    numpy_dtype: str | None
    nullable: bool = False
    semantic_granularity: str = "row"


@dataclass(frozen=True)
class CompiledReplaySchema:
    state_schema: Any
    decision_schema: Any
    action_math_schema: Any | None
    state_specs: tuple[ReplayColumnSpec, ...]
    decision_specs: tuple[ReplayColumnSpec, ...]
    action_math_specs: tuple[ReplayColumnSpec, ...]
    action_count: int
    world_shape: tuple[int, int]
    patch_shape: tuple[int, int] | None
    schema_digest: str

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_COLUMNAR_SCHEMA_VERSION,
            "row_order": ROW_ORDER,
            "action_count": self.action_count,
            "world_shape": list(self.world_shape),
            "patch_shape": None if self.patch_shape is None else list(self.patch_shape),
            "schema_digest": self.schema_digest,
            "state_columns": [spec.name for spec in self.state_specs],
            "decision_columns": [spec.name for spec in self.decision_specs],
            "action_math_columns": [spec.name for spec in self.action_math_specs],
        }


def _pa() -> Any:
    import pyarrow as pa

    return pa


def arrow_type_for_numpy(dtype: np.dtype[Any]) -> Any:
    pa = _pa()
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return pa.bool_()
    if dtype.kind == "i":
        return {
            1: pa.int8(),
            2: pa.int16(),
            4: pa.int32(),
            8: pa.int64(),
        }[dtype.itemsize]
    if dtype.kind == "u":
        return {
            1: pa.uint8(),
            2: pa.uint16(),
            4: pa.uint32(),
            8: pa.uint64(),
        }[dtype.itemsize]
    if dtype.kind == "f":
        return {2: pa.float16(), 4: pa.float32(), 8: pa.float64()}[dtype.itemsize]
    raise TypeError(f"unsupported replay dtype: {dtype}")


def _field(
    name: str, dtype: Any, *, nullable: bool = False, metadata: dict[str, str] | None = None
) -> Any:
    pa = _pa()
    encoded = None
    if metadata:
        encoded = {str(key).encode(): str(value).encode() for key, value in metadata.items()}
    return pa.field(name, dtype, nullable=nullable, metadata=encoded)


def _dictionary_string(index_bits: int = 16) -> Any:
    pa = _pa()
    index = pa.int8() if index_bits == 8 else pa.int16()
    return pa.dictionary(index, pa.string())


def _require_world_shape(array: np.ndarray, world_shape: tuple[int, int], name: str) -> np.ndarray:
    h, w = world_shape
    if array.shape == (h, w):
        return array
    if array.shape == (h * w,):
        return array.reshape(h, w)
    raise ValueError(
        f"{name} must have world-scalar shape {(h, w)} or {(h * w,)}, got {array.shape}"
    )


def _classify_action_field(
    name: str,
    array: np.ndarray,
    *,
    world_shape: tuple[int, int],
    action_count: int,
) -> tuple[ReplayShapeClass, tuple[int, int] | None]:
    h, w = world_shape
    if name in WORLD_ACTION_FIELDS or name in {"authority", "_authority_bool"}:
        if array.shape == (h, w, action_count):
            return ReplayShapeClass.WORLD_ACTION, None
        if array.shape == (h * w, action_count):
            return ReplayShapeClass.WORLD_ACTION, None
        raise ValueError(
            f"{name} must have world-action shape {(h, w, action_count)} or "
            f"{(h * w, action_count)}, got {array.shape}"
        )
    if name in WORLD_SCALAR_DIAGNOSTICS:
        _require_world_shape(array, world_shape, name)
        return ReplayShapeClass.WORLD_SCALAR, None
    if name in PATCH_ACTION_FIELDS:
        if array.ndim != 3 or int(array.shape[-1]) != action_count:
            raise ValueError(
                f"{name} must have patch-action rank/width (...,{action_count}), got {array.shape}"
            )
        ph, pw = int(array.shape[0]), int(array.shape[1])
        if ph <= 0 or pw <= 0 or h % ph or w % pw:
            raise ValueError(f"{name} patch shape {(ph, pw)} must divide world shape {world_shape}")
        return ReplayShapeClass.PATCH_ACTION, (ph, pw)
    if name in GLOBAL_ACTION_FIELDS:
        if array.shape != (action_count,):
            raise ValueError(
                f"{name} must have global-action shape {(action_count,)}, got {array.shape}"
            )
        return ReplayShapeClass.GLOBAL_ACTION, None
    raise KeyError(name)


def compile_replay_schema(
    arrays: dict[str, np.ndarray] | Any,
    *,
    world_shape: tuple[int, int],
    action_names: tuple[str, ...],
    recording_tier: str,
) -> CompiledReplaySchema:
    """Compile deterministic schemas and fail before row materialization."""

    pa = _pa()
    action_count = len(action_names)
    if action_count <= 0 and recording_tier in {"analysis_full", "analysis_sampled", "debug_full"}:
        raise ValueError("action_names are required for action-math recording")
    if len(set(action_names)) != len(action_names):
        raise ValueError("action_names must be unique and ordered")

    normalized = {str(name): np.asarray(value) for name, value in arrays.items()}
    h, w = world_shape
    health = _require_world_shape(normalized["health"], world_shape, "health")
    occupancy = _require_world_shape(
        normalized.get("occupancy", np.full((h, w), -1)), world_shape, "occupancy"
    )
    if health.shape != occupancy.shape:
        raise ValueError("health and occupancy shapes differ")

    condition_type = _dictionary_string(8)
    base_fields = [
        _field("condition", condition_type, metadata={"granularity": "run"}),
        _field("seed", pa.int64()),
        _field("tick", pa.int64()),
        _field("ow_id", pa.int64()),
        _field("y", pa.int32()),
        _field("x", pa.int32()),
    ]
    base_specs = [
        ReplayColumnSpec(
            "condition", None, ReplayShapeClass.RUN_SCALAR, None, semantic_granularity="run"
        ),
        ReplayColumnSpec(
            "seed", None, ReplayShapeClass.RUN_SCALAR, "int64", semantic_granularity="run"
        ),
        ReplayColumnSpec(
            "tick", None, ReplayShapeClass.RUN_SCALAR, "int64", semantic_granularity="tick"
        ),
        ReplayColumnSpec(
            "ow_id",
            "occupancy",
            ReplayShapeClass.WORLD_IDENTITY,
            "int64",
            semantic_granularity="ow_tick",
        ),
        ReplayColumnSpec(
            "y", None, ReplayShapeClass.DERIVED, "int32", semantic_granularity="ow_tick"
        ),
        ReplayColumnSpec(
            "x", None, ReplayShapeClass.DERIVED, "int32", semantic_granularity="ow_tick"
        ),
    ]

    state_fields = list(base_fields)
    state_specs = list(base_specs)
    for name in STATE_FIELDS:
        value = normalized.get(name)
        if value is None:
            continue
        scalar = _require_world_shape(value, world_shape, name)
        state_fields.append(
            _field(
                name,
                arrow_type_for_numpy(scalar.dtype),
                metadata={"source_dtype": str(scalar.dtype), "granularity": "ow_tick"},
            )
        )
        state_specs.append(
            ReplayColumnSpec(
                name,
                name,
                ReplayShapeClass.WORLD_SCALAR,
                str(scalar.dtype),
                semantic_granularity="ow_tick",
            )
        )

    decision_fields = list(state_fields)
    decision_specs = list(state_specs)
    selected_source = normalized.get("raqic_readout", normalized.get("readout"))
    if selected_source is None:
        selected_dtype = np.dtype(np.int16)
    else:
        selected_dtype = _require_world_shape(
            selected_source, world_shape, "selected readout"
        ).dtype
    decision_fields.append(_field("selected_action", arrow_type_for_numpy(selected_dtype)))
    decision_specs.append(
        ReplayColumnSpec(
            "selected_action",
            "raqic_readout|readout",
            ReplayShapeClass.DERIVED,
            str(selected_dtype),
        )
    )
    authority_name = (
        "_authority_bool"
        if "_authority_bool" in normalized
        else "authority"
        if "authority" in normalized
        else None
    )
    if authority_name is not None:
        _classify_action_field(
            authority_name,
            normalized[authority_name],
            world_shape=world_shape,
            action_count=action_count,
        )
        decision_fields.append(_field("legal_action_count", pa.int32()))
        decision_specs.append(
            ReplayColumnSpec(
                "legal_action_count", authority_name, ReplayShapeClass.DERIVED, "int32"
            )
        )
    for source, destination in SELECTED_DECISION_FIELDS.items():
        value = normalized.get(source)
        if value is None:
            continue
        shape_class, _ = _classify_action_field(
            source, value, world_shape=world_shape, action_count=action_count
        )
        decision_fields.append(
            _field(
                destination,
                arrow_type_for_numpy(value.dtype),
                metadata={"source_field": source, "source_dtype": str(value.dtype)},
            )
        )
        decision_specs.append(
            ReplayColumnSpec(
                destination,
                source,
                shape_class,
                str(value.dtype),
                semantic_granularity="selected_action",
            )
        )
    for name in WORLD_SCALAR_DIAGNOSTICS:
        value = normalized.get(name)
        if value is None:
            continue
        _require_world_shape(value, world_shape, name)
        decision_fields.append(
            _field(
                name, arrow_type_for_numpy(value.dtype), metadata={"source_dtype": str(value.dtype)}
            )
        )
        decision_specs.append(
            ReplayColumnSpec(
                name,
                name,
                ReplayShapeClass.WORLD_SCALAR,
                str(value.dtype),
                semantic_granularity="ow_tick",
            )
        )

    action_specs: list[ReplayColumnSpec] = []
    action_fields: list[Any] = []
    patch_shape: tuple[int, int] | None = None
    if recording_tier in {"analysis_full", "analysis_sampled", "debug_full"}:
        action_fields = [
            _field("condition", condition_type, metadata={"granularity": "run"}),
            _field("seed", pa.int64()),
            _field("tick", pa.int64()),
            _field("ow_id", pa.int64()),
            _field("action_index", pa.int16() if action_count <= 32767 else pa.int32()),
            _field("action_name", _dictionary_string(16), metadata={"action_order": "manifest"}),
            _field("selected", pa.bool_()),
        ]
        action_specs = [
            ReplayColumnSpec(
                "condition", None, ReplayShapeClass.RUN_SCALAR, None, semantic_granularity="run"
            ),
            ReplayColumnSpec(
                "seed", None, ReplayShapeClass.RUN_SCALAR, "int64", semantic_granularity="run"
            ),
            ReplayColumnSpec(
                "tick", None, ReplayShapeClass.RUN_SCALAR, "int64", semantic_granularity="tick"
            ),
            ReplayColumnSpec(
                "ow_id",
                "occupancy",
                ReplayShapeClass.WORLD_IDENTITY,
                "int64",
                semantic_granularity="ow_action_tick",
            ),
            ReplayColumnSpec(
                "action_index",
                None,
                ReplayShapeClass.DERIVED,
                "int16" if action_count <= 32767 else "int32",
                semantic_granularity="action",
            ),
            ReplayColumnSpec(
                "action_name", None, ReplayShapeClass.DERIVED, None, semantic_granularity="action"
            ),
            ReplayColumnSpec(
                "selected",
                "raqic_readout|readout",
                ReplayShapeClass.DERIVED,
                "bool",
                semantic_granularity="action",
            ),
        ]
        if authority_name is not None:
            action_fields.append(_field("legal", pa.bool_()))
            action_specs.append(
                ReplayColumnSpec(
                    "legal",
                    authority_name,
                    ReplayShapeClass.WORLD_ACTION,
                    "bool",
                    semantic_granularity="action",
                )
            )
        for name in (
            *WORLD_ACTION_FIELDS,
            *WORLD_SCALAR_DIAGNOSTICS,
            *PATCH_ACTION_FIELDS,
            *GLOBAL_ACTION_FIELDS,
        ):
            value = normalized.get(name)
            if value is None:
                continue
            shape_class, detected_patch = _classify_action_field(
                name, value, world_shape=world_shape, action_count=action_count
            )
            if detected_patch is not None:
                if patch_shape is not None and patch_shape != detected_patch:
                    raise ValueError(
                        f"inconsistent patch shapes: {patch_shape} and {detected_patch}"
                    )
                patch_shape = detected_patch
            action_fields.append(
                _field(
                    name,
                    arrow_type_for_numpy(value.dtype),
                    metadata={
                        "source_dtype": str(value.dtype),
                        "shape_class": shape_class.value,
                        "granularity": "action"
                        if shape_class != ReplayShapeClass.WORLD_SCALAR
                        else "ow_tick_repeated",
                    },
                )
            )
            action_specs.append(
                ReplayColumnSpec(
                    name,
                    name,
                    shape_class,
                    str(value.dtype),
                    semantic_granularity="action"
                    if shape_class != ReplayShapeClass.WORLD_SCALAR
                    else "ow_tick_repeated",
                )
            )

    schema_payload = {
        "version": REPLAY_COLUMNAR_SCHEMA_VERSION,
        "row_order": ROW_ORDER,
        "world_shape": list(world_shape),
        "patch_shape": None if patch_shape is None else list(patch_shape),
        "action_names": list(action_names),
        "state": [(item.name, item.numpy_dtype, item.shape_class.value) for item in state_specs],
        "decision": [
            (item.name, item.numpy_dtype, item.shape_class.value) for item in decision_specs
        ],
        "action": [(item.name, item.numpy_dtype, item.shape_class.value) for item in action_specs],
    }
    digest = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = {
        b"owl.replay.columnar.schema_version": REPLAY_COLUMNAR_SCHEMA_VERSION.encode(),
        b"owl.replay.row_order": ROW_ORDER.encode(),
        b"owl.replay.schema_digest": digest.encode(),
    }
    return CompiledReplaySchema(
        state_schema=pa.schema(state_fields, metadata=metadata),
        decision_schema=pa.schema(decision_fields, metadata=metadata),
        action_math_schema=pa.schema(action_fields, metadata=metadata) if action_fields else None,
        state_specs=tuple(state_specs),
        decision_specs=tuple(decision_specs),
        action_math_specs=tuple(action_specs),
        action_count=action_count,
        world_shape=world_shape,
        patch_shape=patch_shape,
        schema_digest=digest,
    )
