"""Zarr recording interface for simulation histories.

The recorder writes selected dense fields at scheduled ticks. If the optional
``zarr`` dependency is installed, arrays are stored in a Zarr group. In minimal
runtime environments without Zarr, the same public API falls back to a directory
of ``.npy`` arrays plus JSON metadata so recording tests and headless debugging
remain available.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from owl.core.config import SimulationConfig
from owl.core.state import WorldState, action_shape, channel_shape, field_shape
from owl.record.metrics import collect_metrics


def _try_import_zarr() -> Any | None:
    """Import zarr lazily so the module remains importable without it."""
    try:
        import zarr

        return zarr
    except Exception:
        return None


def _zarr_require_group(group: Any, name: str) -> Any:
    """Return an existing or newly-created subgroup across zarr v2/v3 APIs."""
    if hasattr(group, "require_group"):
        return group.require_group(name)
    if hasattr(group, "create_group"):
        try:
            return group.create_group(name)
        except Exception:
            try:
                return group[name]
            except Exception:
                raise
    try:
        return group[name]
    except Exception as exc:
        raise TypeError("zarr group object does not expose require_group/create_group") from exc


def _zarr_create_array(
    group: Any, name: str, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: str
) -> Any:
    """Create a Zarr array across zarr-python v2/v3 API differences.

    ``name`` may contain slash-separated subgroup components. Creating the
    subgroup hierarchy explicitly is more robust than relying on backend-specific
    path parsing and works for both Zarr v2 and v3 style group objects.
    """
    parts = [part for part in str(name).split("/") if part]
    if not parts:
        raise ValueError("zarr array name cannot be empty")
    parent = group
    for part in parts[:-1]:
        parent = _zarr_require_group(parent, part)
    array_name = parts[-1]

    if hasattr(parent, "create_array"):
        return parent.create_array(name=array_name, shape=shape, chunks=chunks, dtype=dtype)
    if hasattr(parent, "create_dataset"):
        return parent.create_dataset(array_name, shape=shape, chunks=chunks, dtype=dtype)
    raise TypeError("zarr group object does not expose create_array or create_dataset")


def _safe_chunk(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return conservative chunks with a single record along time axis."""
    if len(shape) == 1:
        return (min(max(shape[0], 1), 1024),)
    chunks = [1]
    for size in shape[1:]:
        chunks.append(min(max(int(size), 1), 128))
    return tuple(chunks)


class ZarrRecorder:
    """Chunked recorder for selected dense fields.

    Parameters
    ----------
    path:
        Output directory or Zarr store path.
    state:
        Initial world state used to determine shapes and dtypes. This function
        does not mutate state.
    max_steps:
        Maximum number of ticks expected for the run.
    record_every:
        Positive recording period. ``maybe_record`` records when
        ``state.tick % record_every == 0``.
    """

    def __init__(self, path: str, state: WorldState, max_steps: int, record_every: int):
        if int(max_steps) < 0:
            raise ValueError(f"max_steps must be nonnegative, got {max_steps!r}")
        if int(record_every) <= 0:
            raise ValueError(f"record_every must be positive, got {record_every!r}")

        self.path = Path(path)
        if self.path.exists() and self.path.is_file():
            raise ValueError(f"recording path points to a file: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.record_every = int(record_every)
        self.max_steps = int(max_steps)
        self.n_records = int(max_steps) // self.record_every + 2
        self.index = 0
        self.closed = False
        self.metrics: list[dict[str, Any]] = []

        h, w = field_shape(state)
        ah, aw, actions = action_shape(state)
        ch, cw, channels = channel_shape(state)
        if (ah, aw) != (h, w):
            raise ValueError("state.possibility spatial shape must match cell shape")
        if (ch, cw) != (h, w):
            raise ValueError("state.signal spatial shape must match cell shape")

        patch_h, patch_w = state.patches.integration.shape
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("state.patches.integration must have positive shape")

        self.zarr = _try_import_zarr()
        self.root: Any | None = None
        self._arrays: dict[str, Any] = {}
        self._fallback = self.zarr is None

        self._field_specs: dict[str, tuple[tuple[int, ...], str]] = {
            "tick": ((self.n_records,), "i8"),
            "state/integration": ((self.n_records, h, w), "f4"),
            "state/resource": ((self.n_records, h, w), "f4"),
            "state/health": ((self.n_records, h, w), "f4"),
            "state/boundary": ((self.n_records, h, w), "f4"),
            "state/memory": ((self.n_records, h, w), "f4"),
            "state/readout": ((self.n_records, h, w), "i2"),
            "state/possibility": ((self.n_records, h, w, actions), "f4"),
            "environment/food": ((self.n_records, h, w), "f4"),
            "environment/toxin": ((self.n_records, h, w), "f4"),
            "communication/signal": ((self.n_records, h, w, channels), "f4"),
            "communication/signal_reception": ((self.n_records, h, w, channels), "f4"),
            "patch/integration": ((self.n_records, patch_h, patch_w), "f4"),
            "patch/synchrony": ((self.n_records, patch_h, patch_w), "f4"),
            "patch/coherence": ((self.n_records, patch_h, patch_w), "f4"),
            "global/integration": ((self.n_records,), "f4"),
            "global/fragmentation": ((self.n_records,), "f4"),
            "global/diversity": ((self.n_records,), "f4"),
            "global/complexity": ((self.n_records,), "f4"),
            "global/crisis": ((self.n_records,), "f4"),
            "global/carrying_pressure": ((self.n_records,), "f4"),
            "global/starvation_pressure": ((self.n_records,), "f4"),
            "global/food_deficit": ((self.n_records,), "f4"),
        }
        for patch_name in (
            "patch_crisis",
            "patch_carrying_pressure",
            "alive_density",
            "food_mean",
            "starvation_debt_mean",
            "reproduction_fraction",
            "movement_fraction",
            "feed_fraction",
            "death_pressure",
            "noetic_B",
            "noetic_M",
            "noetic_P",
            "noetic_C",
            "noetic_K",
            "noetic_Theta",
            "noetic_N",
        ):
            arr = getattr(state.patches, patch_name, None)
            if isinstance(arr, np.ndarray):
                self._field_specs["patch/" + patch_name.replace("patch_", "")] = (
                    (self.n_records, *arr.shape),
                    "f4",
                )

        optional_specs: dict[str, tuple[str, str]] = {
            "state/occupancy": ("occupancy", "i4"),
            "state/lineage_id": ("lineage_id", "i4"),
            "state/last_utilities": ("last_utilities", "f4"),
            "state/last_logits": ("last_logits", "f4"),
            "state/last_action_probabilities": ("last_action_probabilities", "f4"),
            "state/action_cooldown": ("action_cooldown", "f4"),
            "state/last_intake": ("last_intake", "f4"),
            "state/digestion": ("digestion", "f4"),
            "state/waste": ("waste", "f4"),
            "state/starvation_debt": ("starvation_debt", "f4"),
            "state/last_movement_action": ("last_movement_action", "i4"),
            "state/movement_loop_score": ("movement_loop_score", "f4"),
            "state/genome": ("genome", "f4"),
            "state/development_stage": ("development_stage", "f4"),
            "state/pre_resource": ("pre_resource", "f4"),
            "state/pre_health": ("pre_health", "f4"),
            "state/pre_food": ("pre_food", "f4"),
            "state/pre_starvation_debt": ("pre_starvation_debt", "f4"),
            "state/pre_authority": ("pre_authority", "f4"),
            "state/pre_utilities": ("pre_utilities", "f4"),
            "state/pre_parent_bias": ("pre_parent_bias", "f4"),
            "state/last_survival_value": ("last_survival_value", "f4"),
            "state/last_decision_urgency": ("last_decision_urgency", "f4"),
            "state/last_homeostatic_error": ("last_homeostatic_error", "f4"),
            "state/last_macro_probabilities": ("last_macro_probabilities", "f4"),
            "state/last_chosen_macro": ("last_chosen_macro", "i4"),
            "state/noetic_B": ("noetic_B", "f4"),
            "state/noetic_M": ("noetic_M", "f4"),
            "state/noetic_P": ("noetic_P", "f4"),
            "state/noetic_C": ("noetic_C", "f4"),
            "state/noetic_K": ("noetic_K", "f4"),
            "state/noetic_Theta": ("noetic_Theta", "f4"),
            "state/noetic_N": ("noetic_N", "f4"),
            "state/raqic_probabilities": ("raqic_probabilities", "f4"),
            "state/raqic_readout": ("raqic_readout", "i2"),
            "state/raqic_record_action": ("raqic_record_action", "i2"),
            "state/raqic_record_readout": ("raqic_record_readout", "i4"),
            "state/raqic_record_confidence": ("raqic_record_confidence", "f4"),
            "state/raqic_score": ("raqic_score", "f4"),
            "state/raqic_phase": ("raqic_phase", "f4"),
            "state/raqic_parent_intention": ("raqic_parent_intention", "f4"),
            "state/raqic_audit_flags": ("raqic_audit_flags", "i4"),
            "state/raqic_trace_error": ("raqic_trace_error", "f4"),
            "state/raqic_min_eigenvalue": ("raqic_min_eigenvalue", "f4"),
            "state/raqic_backend_code": ("raqic_backend_code", "i4"),
            "state/raqic_legacy_shadow_possibility": ("raqic_legacy_shadow_possibility", "f4"),
            "state/raqic_legacy_shadow_readout": ("raqic_legacy_shadow_readout", "i2"),
            "state/raqic_compare_l1": ("raqic_compare_l1", "f4"),
            "state/raqic_compare_kl": ("raqic_compare_kl", "f4"),
            "state/raqic_patch_intention": ("raqic_patch_intention", "f4"),
            "state/raqic_patch_record_aggregate": ("raqic_patch_record_aggregate", "f4"),
            "state/raqic_patch_confidence": ("raqic_patch_confidence", "f4"),
            "state/raqic_global_intention": ("raqic_global_intention", "f4"),
            "state/raqic_global_record_aggregate": ("raqic_global_record_aggregate", "f4"),
            "state/raqic_patch_action_phase": ("raqic_patch_action_phase", "f8"),
            "state/raqic_patch_action_coherence": ("raqic_patch_action_coherence", "f8"),
            "state/raqic_global_action_phase": ("raqic_global_action_phase", "f8"),
            "state/raqic_global_action_coherence": ("raqic_global_action_coherence", "f8"),
            "state/raqic_parent_action_phase": ("raqic_parent_action_phase", "f8"),
            "state/raqic_parent_action_coherence": ("raqic_parent_action_coherence", "f8"),
            "state/raqic_pre_mixer_probabilities": (
                "raqic_pre_mixer_probabilities",
                "f8",
            ),
            "state/raqic_utility_innovation": ("raqic_utility_innovation", "f8"),
            "state/raqic_phase_alignment": ("raqic_phase_alignment", "f8"),
            "state/raqic_resonant_parent_intention": (
                "raqic_resonant_parent_intention",
                "f8",
            ),
            "state/raqic_interference_delta_l1": ("raqic_interference_delta_l1", "f8"),
            "state/raqic_policy_kl": ("raqic_policy_kl", "f8"),
            "state/raqic_utility_projection_fraction": (
                "raqic_utility_projection_fraction",
                "f8",
            ),
            "state/raqic_utility_score_cosine": ("raqic_utility_score_cosine", "f8"),
            "state/raqic_utility_orthogonality_residual": (
                "raqic_utility_orthogonality_residual",
                "f8",
            ),
            "state/raqic_utility_innovation_norm": ("raqic_utility_innovation_norm", "f8"),
            "state/raqic_interference_norm_error": (
                "raqic_interference_norm_error",
                "f8",
            ),
            "state/raqic_interference_illegal_mass": (
                "raqic_interference_illegal_mass",
                "f8",
            ),
            "state/raqic_shadow_probabilities": ("raqic_shadow_probabilities", "f8"),
            "state/raqic_shadow_readout": ("raqic_shadow_readout", "i2"),
        }
        self._optional_record_fields: dict[str, str] = {}
        for zarr_name, (attr_name, dtype) in optional_specs.items():
            arr = getattr(state, attr_name, None)
            if isinstance(arr, np.ndarray):
                self._field_specs[zarr_name] = ((self.n_records, *arr.shape), dtype)
                self._optional_record_fields[zarr_name] = attr_name

        attrs = {
            "format": "observer-window-life-recording-v1",
            "record_every": self.record_every,
            "max_steps": self.max_steps,
            "n_records_allocated": self.n_records,
            "height": h,
            "width": w,
            "actions": actions,
            "channels": channels,
        }

        if self.zarr is not None:
            self.root = self.zarr.open_group(str(self.path), mode="w")
            try:
                self.root.attrs.update(attrs)
            except Exception:
                for key, value in attrs.items():
                    self.root.attrs[key] = value
            for name, (shape, dtype) in self._field_specs.items():
                self._arrays[name] = _zarr_create_array(
                    self.root, name, shape, _safe_chunk(shape), dtype
                )
        else:
            self.path.mkdir(parents=True, exist_ok=True)
            self._attrs = attrs
            for name, (shape, dtype) in self._field_specs.items():
                self._arrays[name] = np.zeros(shape, dtype=np.dtype(dtype))

        # Stable aliases used by tests and downstream analysis code. ``N`` is a
        # shorthand for the integration field in supported replay data.
        self.arr_tick = self._arrays["tick"]
        self.arr_N = self._arrays["state/integration"]
        self.arr_integration = self._arrays["state/integration"]
        self.arr_health = self._arrays["state/health"]
        self.arr_resource = self._arrays["state/resource"]
        self.arr_readout = self._arrays["state/readout"]
        self.arr_signal = self._arrays["communication/signal"]

    def _write_array(self, name: str, index: int, value: np.ndarray | float | int) -> None:
        """Write one record slice to a backing array."""
        array = self._arrays[name]
        if name == "tick" or name.startswith("global/"):
            array[index] = value
        else:
            array[index, ...] = value

    def maybe_record(self, state: WorldState) -> None:
        """Record selected fields if the current tick matches schedule.

        Mutates the recorder only. ``WorldState`` is read but not modified.
        """
        if self.closed:
            raise RuntimeError("cannot record after recorder.close()")
        if int(state.tick) % self.record_every != 0:
            return
        if self.index >= self.n_records:
            raise RuntimeError(
                f"recorder capacity exceeded: index={self.index}, allocated={self.n_records}, "
                "increase max_steps or record_every"
            )

        h, w = field_shape(state)
        _, _, channels = channel_shape(state)
        expected_signal = self._field_specs["communication/signal"][0][1:]
        if (h, w, channels) != expected_signal:
            raise ValueError(
                f"state signal shape {(h, w, channels)} does not match "
                f"recorder shape {expected_signal}"
            )

        i = self.index
        self._write_array("tick", i, int(state.tick))
        self._write_array("state/integration", i, state.integration)
        self._write_array("state/resource", i, state.resource)
        self._write_array("state/health", i, state.health)
        self._write_array("state/boundary", i, state.boundary)
        self._write_array("state/memory", i, state.memory)
        self._write_array("state/readout", i, state.readout)
        self._write_array("state/possibility", i, state.possibility)
        self._write_array("environment/food", i, state.food)
        self._write_array("environment/toxin", i, state.toxin)
        self._write_array("communication/signal", i, state.signal)
        self._write_array("communication/signal_reception", i, state.signal_reception)
        self._write_array("patch/integration", i, state.patches.integration)
        self._write_array("patch/synchrony", i, state.patches.synchrony)
        self._write_array("patch/coherence", i, state.patches.coherence)
        self._write_array("global/integration", i, float(state.global_state.integration))
        self._write_array("global/fragmentation", i, float(state.global_state.fragmentation))
        self._write_array("global/diversity", i, float(state.global_state.diversity))
        self._write_array("global/complexity", i, float(state.global_state.complexity))
        self._write_array("global/crisis", i, float(getattr(state.global_state, "crisis", 0.0)))
        self._write_array(
            "global/carrying_pressure",
            i,
            float(getattr(state.global_state, "carrying_pressure", 0.0)),
        )
        self._write_array(
            "global/starvation_pressure",
            i,
            float(getattr(state.global_state, "starvation_pressure", 0.0)),
        )
        self._write_array(
            "global/food_deficit", i, float(getattr(state.global_state, "food_deficit", 0.0))
        )
        for patch_name in (
            "patch_crisis",
            "patch_carrying_pressure",
            "alive_density",
            "food_mean",
            "starvation_debt_mean",
            "reproduction_fraction",
            "movement_fraction",
            "feed_fraction",
            "death_pressure",
            "noetic_B",
            "noetic_M",
            "noetic_P",
            "noetic_C",
            "noetic_K",
            "noetic_Theta",
            "noetic_N",
        ):
            arr = getattr(state.patches, patch_name, None)
            zname = "patch/" + patch_name.replace("patch_", "")
            if isinstance(arr, np.ndarray) and zname in self._arrays:
                self._write_array(zname, i, arr)
        for name, attr_name in self._optional_record_fields.items():
            arr = getattr(state, attr_name, None)
            if isinstance(arr, np.ndarray):
                self._write_array(name, i, arr)
        self.metrics.append(collect_metrics(state, _minimal_cfg_from_state(state)))
        self.index += 1

    def close(self) -> None:
        """Finalize recorder resources.

        For real Zarr stores this updates metadata and closes the underlying
        store if the installed backend exposes a close method. For fallback
        stores this writes ``.npy`` arrays and JSON metadata.
        """
        if self.closed:
            return

        if self._fallback:
            self.path.mkdir(parents=True, exist_ok=True)
            attrs = dict(self._attrs)
            attrs["recorded_count"] = int(self.index)
            attrs["backend"] = "numpy-directory-fallback"
            (self.path / "metadata.json").write_text(
                json.dumps(attrs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (self.path / "metrics.json").write_text(
                json.dumps(self.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            for name, array in self._arrays.items():
                filename = name.replace("/", "__") + ".npy"
                np.save(
                    self.path / filename,
                    np.asarray(array)[: self.index] if array.shape[0] == self.n_records else array,
                )
        else:
            assert self.root is not None
            with contextlib.suppress(Exception):
                self.root.attrs["recorded_count"] = int(self.index)
            with contextlib.suppress(Exception):
                (self.path / "metrics.json").write_text(
                    json.dumps(self.metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            with contextlib.suppress(Exception):
                self.root.store.close()

        self.closed = True


def _minimal_cfg_from_state(state: WorldState) -> Any:
    """Construct a minimal config-like object for metrics.

    ``collect_metrics`` only needs channel count, epsilon, and resource scale.
    In normal code callers use :func:`create_recorder`, but the recorder public
    constructor intentionally preserves the skeleton signature without cfg.
    """
    channels = int(state.signal.shape[-1])
    return SimpleNamespace(
        communication=SimpleNamespace(num_channels=channels),
        actions=SimpleNamespace(epsilon=1e-8),
        resources=SimpleNamespace(max_resource=1.0),
    )


def create_recorder(
    cfg: SimulationConfig, state: WorldState, max_steps: int
) -> ZarrRecorder | None:
    """Construct a recorder from config if recording is enabled.

    Parameters
    ----------
    cfg:
        Simulation configuration. ``cfg.recording.enabled`` controls whether a
        recorder is returned.
    state:
        Initial world state used for shape allocation.
    max_steps:
        Expected run length.

    Returns
    -------
    ZarrRecorder | None
        Recorder instance, or ``None`` when recording is disabled.
    """
    if not cfg.recording.enabled:
        return None
    return ZarrRecorder(
        path=cfg.recording.zarr_path,
        state=state,
        max_steps=max_steps,
        record_every=cfg.recording.record_every,
    )
