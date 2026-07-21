from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from owl.core.actions import Action
from owl.gpu.streams import StreamBundle, TransferTicket


class MetricIndex(IntEnum):
    TICK = 0
    ALIVE_COUNT = 1
    FOOD_TOTAL = 2
    TOXIN_TOTAL = 3
    MEAN_HEALTH = 4
    MEAN_RESOURCE = 5
    MEAN_INTEGRATION = 6
    RAQIC_MAX_ROW_ERROR = 7
    FALLBACK_COUNT = 8
    TOPOLOGY_OVERFLOW = 9
    VISUAL_EVENT_OVERFLOW = 10
    GRAPH_REPLAY_COUNT = 11
    RAQIC_ELIGIBLE_COUNT = 12
    RAQIC_PROCESSED_COUNT = 13
    NAN_COUNT_HEALTH = 14
    NAN_COUNT_RESOURCE = 15
    DEAD_NONREST_COUNT = 16
    MEAN_RAQIC_UTILITY_INNOVATION_L1 = 17
    MEAN_RAQIC_PHASE_ALIGNMENT = 18
    MEAN_RAQIC_INTERFERENCE_DELTA_L1 = 19
    MEAN_RAQIC_POLICY_KL = 20
    MEAN_RAQIC_UTILITY_PROJECTION_FRACTION = 21
    MEAN_RAQIC_UTILITY_SCORE_COSINE = 22
    MAX_RAQIC_INTERFERENCE_NORM_ERROR = 23
    MAX_RAQIC_INTERFERENCE_ILLEGAL_MASS = 24
    RAQIC_SHADOW_READOUT_CHANGE_FRACTION = 25


METRIC_NAMES = tuple(m.name.lower() for m in MetricIndex)


@dataclass
class DeviceMetricSlab:
    """Compact device-resident metrics with asynchronous host transfer."""

    backend: Any
    slab: Any
    host_owner: Any = None

    @classmethod
    def create(cls, backend: Any) -> DeviceMetricSlab:
        return cls(
            backend=backend, slab=backend.xp.zeros((len(MetricIndex),), dtype=backend.xp.float64)
        )

    def update(self, ds: Any, *, fallback_count: int = 0, graph_replay_count: int = 0) -> None:
        xp = self.backend.xp
        live = (ds.health > 0) & (~ds.obstacle)
        live_count = xp.sum(live, dtype=xp.float64)
        safe_count = xp.maximum(live_count, xp.asarray(1.0, dtype=xp.float64))
        live_f = live.astype(xp.float64)
        self.slab[MetricIndex.TICK] = xp.asarray(ds.tick, dtype=xp.float64)
        self.slab[MetricIndex.ALIVE_COUNT] = live_count
        self.slab[MetricIndex.FOOD_TOTAL] = xp.sum(ds.food, dtype=xp.float64)
        self.slab[MetricIndex.TOXIN_TOTAL] = xp.sum(ds.toxin, dtype=xp.float64)
        self.slab[MetricIndex.MEAN_HEALTH] = (
            xp.sum(ds.health.astype(xp.float64) * live_f) / safe_count
        )
        self.slab[MetricIndex.MEAN_RESOURCE] = (
            xp.sum(ds.resource.astype(xp.float64) * live_f) / safe_count
        )
        self.slab[MetricIndex.MEAN_INTEGRATION] = (
            xp.sum(ds.integration.astype(xp.float64) * live_f) / safe_count
        )
        probs = ds.arrays.get("raqic_probabilities")
        if probs is not None:
            sums = xp.sum(probs.astype(xp.float64), axis=-1)
            err = xp.where(live, xp.abs(sums - 1.0), 0.0)
            self.slab[MetricIndex.RAQIC_MAX_ROW_ERROR] = xp.max(err)
        else:
            self.slab[MetricIndex.RAQIC_MAX_ROW_ERROR] = 0.0
        self.slab[MetricIndex.FALLBACK_COUNT] = float(fallback_count)
        self.slab[MetricIndex.TOPOLOGY_OVERFLOW] = xp.asarray(
            ds.scalars.get("topology_overflow", 0), dtype=xp.float64
        )
        self.slab[MetricIndex.VISUAL_EVENT_OVERFLOW] = xp.asarray(
            ds.scalars.get("visual_event_overflow", 0), dtype=xp.float64
        )
        self.slab[MetricIndex.GRAPH_REPLAY_COUNT] = float(graph_replay_count)
        eligible = ds.arrays.get("_raqic_eligible_count")
        processed = ds.arrays.get("_raqic_processed_count")
        self.slab[MetricIndex.RAQIC_ELIGIBLE_COUNT] = (
            live_count if eligible is None else xp.asarray(eligible, dtype=xp.float64)
        )
        self.slab[MetricIndex.RAQIC_PROCESSED_COUNT] = (
            live_count if processed is None else xp.asarray(processed, dtype=xp.float64)
        )
        self.slab[MetricIndex.NAN_COUNT_HEALTH] = xp.sum(~xp.isfinite(ds.health), dtype=xp.float64)
        self.slab[MetricIndex.NAN_COUNT_RESOURCE] = xp.sum(
            ~xp.isfinite(ds.resource), dtype=xp.float64
        )
        readout = ds.arrays.get("raqic_readout")
        self.slab[MetricIndex.DEAD_NONREST_COUNT] = (
            xp.asarray(0.0, dtype=xp.float64)
            if readout is None
            else xp.sum((~live) & (readout != int(Action.REST)), dtype=xp.float64)
        )

        def live_mean(name: str, *, reduce_action: str = "mean") -> Any:
            value = ds.arrays.get(name)
            if value is None:
                return xp.asarray(0.0, dtype=xp.float64)
            cast = value.astype(xp.float64)
            if cast.ndim == live.ndim + 1:
                if reduce_action == "l1":
                    cast = xp.sum(xp.abs(cast), axis=-1, dtype=xp.float64)
                else:
                    cast = xp.mean(cast, axis=-1, dtype=xp.float64)
            return xp.sum(cast * live_f, dtype=xp.float64) / safe_count

        self.slab[MetricIndex.MEAN_RAQIC_UTILITY_INNOVATION_L1] = live_mean(
            "raqic_utility_innovation", reduce_action="l1"
        )
        self.slab[MetricIndex.MEAN_RAQIC_PHASE_ALIGNMENT] = live_mean("raqic_phase_alignment")
        self.slab[MetricIndex.MEAN_RAQIC_INTERFERENCE_DELTA_L1] = live_mean(
            "raqic_interference_delta_l1"
        )
        self.slab[MetricIndex.MEAN_RAQIC_POLICY_KL] = live_mean("raqic_policy_kl")
        self.slab[MetricIndex.MEAN_RAQIC_UTILITY_PROJECTION_FRACTION] = live_mean(
            "raqic_utility_projection_fraction"
        )
        self.slab[MetricIndex.MEAN_RAQIC_UTILITY_SCORE_COSINE] = live_mean(
            "raqic_utility_score_cosine"
        )
        norm_error = ds.arrays.get("raqic_interference_norm_error")
        illegal_mass = ds.arrays.get("raqic_interference_illegal_mass")
        self.slab[MetricIndex.MAX_RAQIC_INTERFERENCE_NORM_ERROR] = (
            xp.asarray(0.0, dtype=xp.float64)
            if norm_error is None
            else xp.max(xp.where(live, xp.abs(norm_error), 0.0))
        )
        self.slab[MetricIndex.MAX_RAQIC_INTERFERENCE_ILLEGAL_MASS] = (
            xp.asarray(0.0, dtype=xp.float64)
            if illegal_mass is None
            else xp.max(xp.where(live, xp.abs(illegal_mass), 0.0))
        )
        shadow_readout = ds.arrays.get("raqic_shadow_readout")
        self.slab[MetricIndex.RAQIC_SHADOW_READOUT_CHANGE_FRACTION] = (
            xp.asarray(0.0, dtype=xp.float64)
            if shadow_readout is None or readout is None
            else xp.sum(live & (shadow_readout != readout), dtype=xp.float64) / safe_count
        )

    def transfer_async(
        self, streams: StreamBundle, *, metadata: dict[str, Any] | None = None
    ) -> TransferTicket:
        host, owner = streams.pinned_array(self.slab.shape, dtype=np.float64)
        if not self.backend.is_gpu:
            np.copyto(host, np.asarray(self.slab, dtype=np.float64))
            event = streams.new_event()
            event.record(streams.transfer)
            return TransferTicket(host, event, owner, metadata or {})

        ready = streams.record(streams.compute)
        streams.wait(streams.transfer, ready)
        with streams.transfer:
            # CuPy ndarray.get supports nonblocking copies only when ``out`` is
            # page-locked host memory. Fall back to a blocking copy only if an
            # Some supported CuPy installations do not accept this keyword.
            try:
                self.slab.get(out=host, stream=streams.transfer, blocking=False)
            except TypeError:  # pragma: no cover - version-dependent GPU host
                np.copyto(host, self.backend.asnumpy(self.slab))
            done = streams.record(streams.transfer)
        return TransferTicket(host, done, owner, metadata or {})

    @staticmethod
    def decode(array: np.ndarray, *, backend: str, persistent: bool = True) -> dict[str, Any]:
        values = np.asarray(array, dtype=np.float64)
        out: dict[str, Any] = {name: float(values[i]) for i, name in enumerate(METRIC_NAMES)}
        out["tick"] = int(round(out["tick"]))
        out["alive_count"] = int(round(out["alive_count"]))
        out["fallback_count"] = int(round(out["fallback_count"]))
        out["topology_overflow"] = int(round(out["topology_overflow"]))
        out["visual_event_overflow"] = int(round(out["visual_event_overflow"]))
        out["graph_replay_count"] = int(round(out["graph_replay_count"]))
        out["raqic_eligible_count"] = int(round(out["raqic_eligible_count"]))
        out["raqic_processed_count"] = int(round(out["raqic_processed_count"]))
        out["nan_count_health"] = int(round(out["nan_count_health"]))
        out["nan_count_resource"] = int(round(out["nan_count_resource"]))
        out["dead_nonrest_count"] = int(round(out["dead_nonrest_count"]))
        out["all_ow_accounted"] = out["raqic_eligible_count"] == out["raqic_processed_count"]
        out["backend"] = backend
        out["persistent"] = persistent
        return out
