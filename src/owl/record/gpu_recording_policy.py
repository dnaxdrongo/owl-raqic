from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from owl.record.snapshots import save_snapshot

_VALID_LEVELS = {
    "metrics_only",
    "metrics_plus_events",
    "sampled_cells",
    "patch_summaries",
    "full_snapshot_decimated",
    "debug_full_every_tick",
}


@dataclass(frozen=True)
class GPURecordingPolicy:
    level: str
    every: int
    sample_limit: int
    record_measurement_records: bool
    record_action_probabilities: bool
    persist_quantum_state: bool
    record_actualization_diagnostics: bool
    output_dir: Path

    @classmethod
    def from_config(cls, cfg: Any) -> GPURecordingPolicy:
        level = str(
            getattr(
                cfg.raqic,
                "full_gpu_recording_level_v07",
                getattr(cfg.raqic, "full_gpu_recording_level", "metrics_only"),
            )
        )
        legacy_map = {
            "summary_gpu": "metrics_only",
            "sampled_gpu": "sampled_cells",
            "full_gpu_snapshot": "full_snapshot_decimated",
        }
        level = legacy_map.get(level, level)
        if level not in _VALID_LEVELS:
            raise ValueError(f"unknown full GPU recording level: {level}")
        metrics_path = Path(cfg.recording.metrics_path)
        return cls(
            level=level,
            every=max(1, int(getattr(cfg.raqic, "full_gpu_record_every", 1))),
            sample_limit=max(1, int(getattr(cfg.raqic, "gpu_audit_limit", 16) or 16)),
            record_measurement_records=bool(getattr(cfg.raqic, "record_measurement_records", True)),
            record_action_probabilities=bool(
                getattr(cfg.raqic, "record_action_probabilities", True)
            ),
            persist_quantum_state=bool(getattr(cfg.raqic, "persist_quantum_state", True)),
            record_actualization_diagnostics=bool(
                getattr(cfg.raqic, "record_actualization_diagnostics", False)
            ),
            output_dir=metrics_path.parent / "gpu_records",
        )

    def due(self, tick: int) -> bool:
        if self.level == "debug_full_every_tick":
            return True
        return int(tick) % self.every == 0

    def metric_due(self, tick: int) -> bool:
        return self.due(tick)

    def _event_summary(self, run: Any) -> dict[str, Any]:
        ds, xp = run.ds, run.ds.xp
        action_count = int(ds.possibility.shape[-1])
        live = (ds.health > 0) & (~ds.obstacle)
        histogram = xp.bincount(
            xp.where(live, ds.readout, 0).reshape(-1).astype(xp.int32),
            minlength=action_count,
        )
        topology = ds.metadata.get("last_topology_events")
        topo_count = 0
        if topology is not None:
            topo_count = int(ds.backend.asnumpy(topology.device_count).reshape(-1)[0])
        return {
            "action_histogram": ds.backend.asnumpy(histogram).astype(int).tolist(),
            "topology_event_count": topo_count,
            "visual_event_overflow": int(ds.scalars.get("visual_event_overflow", 0)),
        }

    def _sampled_cells(self, run: Any) -> dict[str, Any]:
        ds, xp = run.ds, run.ds.xp
        live = ((ds.health > 0) & (~ds.obstacle)).reshape(-1)
        scores = ds.arrays.get("raqic_record_confidence", ds.integration).reshape(-1)
        # Select a deterministic high-confidence/low-index sample at the
        # recording boundary without copying full decision tensors.
        rank = xp.where(live, scores, -xp.inf)
        limit = min(self.sample_limit, int(rank.size))
        indices = xp.argsort(rank)[-limit:]
        valid = live[indices]
        indices = indices[valid]
        payload: dict[str, Any] = {
            "flat_indices": ds.backend.asnumpy(indices).astype(int).tolist(),
            "health": ds.backend.asnumpy(ds.health.reshape(-1)[indices]).tolist(),
            "resource": ds.backend.asnumpy(ds.resource.reshape(-1)[indices]).tolist(),
            "readout": ds.backend.asnumpy(ds.readout.reshape(-1)[indices]).astype(int).tolist(),
        }
        if self.record_action_probabilities and "raqic_probabilities" in ds.arrays:
            actions = int(ds.raqic_probabilities.shape[-1])
            payload["probabilities"] = ds.backend.asnumpy(
                ds.raqic_probabilities.reshape(-1, actions)[indices]
            ).tolist()
        if self.record_measurement_records:
            record_names = [
                "raqic_record_confidence",
                "raqic_trace_error",
                "raqic_min_eigenvalue",
                "raqic_audit_flags",
            ]
            if self.record_actualization_diagnostics:
                record_names.extend(
                    [
                        "raqic_pre_mixer_probabilities",
                        "raqic_utility_innovation",
                        "raqic_phase_alignment",
                        "raqic_resonant_parent_intention",
                        "raqic_interference_delta_l1",
                        "raqic_policy_kl",
                        "raqic_utility_projection_fraction",
                        "raqic_utility_score_cosine",
                        "raqic_utility_orthogonality_residual",
                        "raqic_utility_innovation_norm",
                        "raqic_interference_norm_error",
                        "raqic_interference_illegal_mass",
                        "raqic_shadow_probabilities",
                        "raqic_shadow_readout",
                    ]
                )
            for name in record_names:
                if name in ds.arrays:
                    array = ds.arrays[name]
                    payload[name] = ds.backend.asnumpy(
                        array.reshape((array.shape[0] * array.shape[1], *array.shape[2:]))[indices]
                    ).tolist()
        return payload

    def _patch_summaries(self, run: Any) -> dict[str, Any]:
        selected = {}
        for name in (
            "health",
            "resource",
            "integration",
            "coherence",
            "intention",
            "policy_bias",
            "phase",
        ):
            array = run.ds.patch_arrays.get(name)
            if array is not None:
                selected[name] = run.ds.backend.asnumpy(array).tolist()
        if self.record_actualization_diagnostics:
            for name in (
                "raqic_patch_action_phase",
                "raqic_patch_action_coherence",
                "raqic_global_action_phase",
                "raqic_global_action_coherence",
            ):
                array = run.ds.arrays.get(name)
                if array is not None:
                    selected[name] = run.ds.backend.asnumpy(array).tolist()
        return selected

    def record_tick(self, run: Any, diagnostics: dict[str, Any]) -> dict[str, Any] | None:
        tick = int(run.ds.tick)
        if not self.due(tick) or run.async_writer is None:
            return None
        payload: dict[str, Any] = {
            "schema_version": "owl.gpu-record.v096",
            "kind": self.level,
            "tick": tick,
            "backend": run.ds.backend.name,
            "actualization_variant": str(
                getattr(run.cfg.raqic, "actualization_variant", "stable_baseline")
            ),
            "record_actualization_diagnostics": self.record_actualization_diagnostics,
        }
        if self.level in {
            "metrics_plus_events",
            "sampled_cells",
            "patch_summaries",
            "full_snapshot_decimated",
            "debug_full_every_tick",
        }:
            payload["events"] = self._event_summary(run)
        if self.level == "sampled_cells":
            payload["sampled_cells"] = self._sampled_cells(run)
        elif self.level == "patch_summaries":
            payload["patches"] = self._patch_summaries(run)
        elif self.level in {"full_snapshot_decimated", "debug_full_every_tick"}:
            state = run.checkpoint(force=True, count=False)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"snapshot_{tick:08d}.npz"
            save_snapshot(state, str(path))
            payload["snapshot"] = str(path)
            payload["quantum_fields_persisted"] = self.persist_quantum_state
        run.async_writer.write(payload)
        return payload
