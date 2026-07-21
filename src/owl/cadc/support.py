"""Fold-specific support, OOD, disagreement, and abstention calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from owl.cadc.schema import AbstentionReason, SupportStatus


@dataclass(frozen=True)
class SupportDecision:
    """One calibrated support, OOD, uncertainty, and abstention decision."""

    status: SupportStatus
    abstention_reason: AbstentionReason
    knn_distance: float
    mahalanobis_distance: float
    neighbor_seed_count: int
    action_support: int
    repeat_support: int
    ensemble_disagreement: float
    interval_width: float


class SupportCalibrator:
    """Fit support geometry on training rows and abstain outside frozen bounds."""

    def __init__(
        self,
        *,
        k: int,
        minimum_seeds: int,
        minimum_decisions: int,
        minimum_repeats: int,
        maximum_disagreement: float,
        maximum_interval_width: float,
        shrinkage: float = 0.1,
    ) -> None:
        if k < 1 or minimum_seeds < 1 or minimum_decisions < 1 or minimum_repeats < 1:
            raise ValueError("support counts must be positive")
        if not 0.0 < shrinkage <= 1.0:
            raise ValueError("covariance shrinkage must be in (0,1]")
        self.k = k
        self.minimum_seeds = minimum_seeds
        self.minimum_decisions = minimum_decisions
        self.minimum_repeats = minimum_repeats
        self.maximum_disagreement = maximum_disagreement
        self.maximum_interval_width = maximum_interval_width
        self.shrinkage = shrinkage
        self.embeddings: npt.NDArray[Any] | None = None
        self.seeds: npt.NDArray[Any] | None = None
        self.center: npt.NDArray[Any] | None = None
        self.precision: npt.NDArray[Any] | None = None
        self.knn_threshold = np.inf
        self.mahalanobis_threshold = np.inf
        self._neighbors: Any | None = None
        self.neighbor_backend = "uninitialized"
        self.geometry_backend = "uninitialized"
        self._gpu_center: Any | None = None
        self._gpu_precision: Any | None = None

    def fit(self, embeddings: Any, seeds: Any, *, quantile: float = 0.99) -> SupportCalibrator:
        """Fit fold-local geometry without reading held-out examples."""
        values = np.asarray(embeddings, dtype=np.float64)
        seed_values = np.asarray(seeds, dtype=np.int64)
        if values.ndim != 2 or values.shape[0] != seed_values.size:
            raise ValueError("support embeddings and seeds have incompatible shapes")
        if values.shape[0] <= self.k:
            raise ValueError("support fit needs more rows than k")
        if not np.isfinite(values).all():
            raise ValueError("support embeddings contain nonfinite values")
        self.embeddings = values
        self.seeds = seed_values
        try:
            import cupy as cp
            from cuml.neighbors import NearestNeighbors as GPUNearestNeighbors

            if not cp.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            self._neighbors = GPUNearestNeighbors(
                n_neighbors=min(self.k + 1, values.shape[0]),
                algorithm="brute",
                output_type="cupy",
            ).fit(cp.asarray(values))
            self.neighbor_backend = "cuml_cuda"
            gpu_values = cp.asarray(values, dtype=cp.float64)
            self._gpu_center = gpu_values.mean(axis=0)
            covariance = cp.cov(gpu_values, rowvar=False)
            if covariance.ndim == 0:
                covariance = covariance.reshape(1, 1)
            diagonal = cp.diag(cp.diag(covariance))
            shrunk = (
                (1.0 - self.shrinkage) * covariance
                + self.shrinkage * diagonal
            )
            ridge = cp.maximum(
                cp.asarray(1e-10, dtype=cp.float64),
                cp.trace(shrunk) / max(1, shrunk.shape[0]) * 1e-8,
            )
            self._gpu_precision = cp.linalg.pinv(
                shrunk + ridge * cp.eye(shrunk.shape[0], dtype=cp.float64)
            )
            self.center = cp.asnumpy(self._gpu_center)
            self.precision = cp.asnumpy(self._gpu_precision)
            self.geometry_backend = "cupy_cuda_float64"
        except (ImportError, RuntimeError):
            from sklearn.neighbors import NearestNeighbors

            self._neighbors = NearestNeighbors(
                n_neighbors=min(self.k + 1, values.shape[0]),
                algorithm="auto",
                n_jobs=-1,
            ).fit(values)
            self.neighbor_backend = "sklearn_cpu"
            self.center = values.mean(axis=0)
            covariance = np.cov(values, rowvar=False)
            if covariance.ndim == 0:
                covariance = np.asarray([[float(covariance)]])
            diagonal = np.diag(np.diag(covariance))
            shrunk = (
                (1.0 - self.shrinkage) * covariance
                + self.shrinkage * diagonal
            )
            ridge = max(
                1e-10,
                float(np.trace(shrunk)) / max(1, shrunk.shape[0]) * 1e-8,
            )
            self.precision = np.linalg.pinv(
                shrunk + ridge * np.eye(shrunk.shape[0])
            )
            self.geometry_backend = "numpy_cpu_float64"
        knn, _ = self._knn(values, exclude_self=True)
        mahalanobis = self._mahalanobis(values)
        self.knn_threshold = float(np.quantile(knn, quantile))
        self.mahalanobis_threshold = float(np.quantile(mahalanobis, quantile))
        return self

    def _knn(
        self, values: npt.NDArray[Any], *, exclude_self: bool
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        if self.embeddings is None:
            raise RuntimeError("support calibrator has not been fit")
        if self._neighbors is None:
            raise RuntimeError("support neighbor index has not been fit")
        requested = min(self.k + int(exclude_self), self.embeddings.shape[0])
        query: Any = values
        if self.neighbor_backend == "cuml_cuda":
            import cupy as cp

            query = cp.asarray(values)
        distances, indices = self._neighbors.kneighbors(
            query, n_neighbors=requested, return_distance=True
        )
        if self.neighbor_backend == "cuml_cuda":
            import cupy as cp

            distances = cp.asnumpy(distances)
            indices = cp.asnumpy(indices)
        if exclude_self:
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        return np.sqrt(np.mean(distances**2, axis=1)), indices

    def _mahalanobis(self, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.center is None or self.precision is None:
            raise RuntimeError("support calibrator has not been fit")
        if self.geometry_backend == "cupy_cuda_float64":
            import cupy as cp

            if self._gpu_center is None or self._gpu_precision is None:
                raise RuntimeError("GPU support geometry is incomplete")
            gpu_values = cp.asarray(values, dtype=cp.float64)
            centered = gpu_values - self._gpu_center
            squared = cp.einsum(
                "bi,ij,bj->b",
                centered,
                self._gpu_precision,
                centered,
            )
            return np.asarray(cp.asnumpy(cp.sqrt(cp.maximum(squared, 0.0))))
        centered = values - self.center
        squared = np.einsum("bi,ij,bj->b", centered, self.precision, centered)
        return np.asarray(np.sqrt(np.maximum(squared, 0.0)))

    def decide(
        self,
        embedding: Any,
        *,
        action_support: int,
        repeat_support: int,
        disagreement: float,
        interval_width: float,
    ) -> SupportDecision:
        """Classify one candidate and return all support diagnostics."""
        values = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        knn, indices = self._knn(values, exclude_self=False)
        mahalanobis = self._mahalanobis(values)
        if self.seeds is None:
            raise RuntimeError("support calibrator has not been fit")
        seed_count = int(np.unique(self.seeds[indices[0]]).size)
        reason = AbstentionReason.NONE
        status = SupportStatus.SUPPORTED
        if seed_count < self.minimum_seeds:
            reason, status = AbstentionReason.LOW_SEED_COVERAGE, SupportStatus.INSUFFICIENT
        elif action_support < self.minimum_decisions:
            reason, status = AbstentionReason.LOW_ACTION_SUPPORT, SupportStatus.INSUFFICIENT
        elif repeat_support < self.minimum_repeats:
            reason, status = AbstentionReason.LOW_REPEAT_SUPPORT, SupportStatus.INSUFFICIENT
        elif knn[0] > self.knn_threshold or mahalanobis[0] > self.mahalanobis_threshold:
            reason, status = AbstentionReason.FEATURE_OOD, SupportStatus.OOD
        elif disagreement > self.maximum_disagreement:
            reason, status = AbstentionReason.HIGH_DISAGREEMENT, SupportStatus.LIMITED
        elif interval_width > self.maximum_interval_width:
            reason, status = AbstentionReason.WIDE_INTERVAL, SupportStatus.LIMITED
        return SupportDecision(
            status=status,
            abstention_reason=reason,
            knn_distance=float(knn[0]),
            mahalanobis_distance=float(mahalanobis[0]),
            neighbor_seed_count=seed_count,
            action_support=int(action_support),
            repeat_support=int(repeat_support),
            ensemble_disagreement=float(disagreement),
            interval_width=float(interval_width),
        )

    def decide_batch(
        self,
        embeddings: Any,
        *,
        action_support: Any,
        repeat_support: Any,
        disagreement: Any,
        interval_width: Any,
    ) -> dict[str, npt.NDArray[Any]]:
        """Vectorize support/abstention decisions for a candidate batch."""
        values = np.asarray(embeddings, dtype=np.float64)
        knn, indices = self._knn(values, exclude_self=False)
        mahalanobis = self._mahalanobis(values)
        if self.seeds is None:
            raise RuntimeError("support calibrator has not been fit")
        neighbor_seeds = np.sort(self.seeds[indices], axis=1)
        seed_count = 1 + np.sum(
            neighbor_seeds[:, 1:] != neighbor_seeds[:, :-1], axis=1
        )
        actions = np.asarray(action_support, dtype=np.int64)
        repeats = np.asarray(repeat_support, dtype=np.int64)
        spread = np.asarray(disagreement, dtype=np.float64)
        width = np.asarray(interval_width, dtype=np.float64)
        expected = (values.shape[0],)
        if any(value.shape != expected for value in (actions, repeats, spread, width)):
            raise ValueError("support diagnostic vectors do not align")
        status = np.full(expected, SupportStatus.SUPPORTED.value, dtype=object)
        reason = np.full(expected, AbstentionReason.NONE.value, dtype=object)
        predicates = (
            (
                seed_count < self.minimum_seeds,
                SupportStatus.INSUFFICIENT,
                AbstentionReason.LOW_SEED_COVERAGE,
            ),
            (
                actions < self.minimum_decisions,
                SupportStatus.INSUFFICIENT,
                AbstentionReason.LOW_ACTION_SUPPORT,
            ),
            (
                repeats < self.minimum_repeats,
                SupportStatus.INSUFFICIENT,
                AbstentionReason.LOW_REPEAT_SUPPORT,
            ),
            (
                (knn > self.knn_threshold) | (mahalanobis > self.mahalanobis_threshold),
                SupportStatus.OOD,
                AbstentionReason.FEATURE_OOD,
            ),
            (
                spread > self.maximum_disagreement,
                SupportStatus.LIMITED,
                AbstentionReason.HIGH_DISAGREEMENT,
            ),
            (
                width > self.maximum_interval_width,
                SupportStatus.LIMITED,
                AbstentionReason.WIDE_INTERVAL,
            ),
        )
        undecided = np.ones(expected, dtype=bool)
        for predicate, selected_status, selected_reason in predicates:
            selected = undecided & predicate
            status[selected] = selected_status.value
            reason[selected] = selected_reason.value
            undecided &= ~selected
        return {
            "support_status": status,
            "abstention_reason": reason,
            "knn_distance": knn,
            "mahalanobis_distance": mahalanobis,
            "neighbor_seed_count": seed_count.astype(np.int32),
        }

    def manifest(self) -> dict[str, Any]:
        """Return a canonical manifest of fitted support thresholds."""
        if self.center is None or self.precision is None:
            raise RuntimeError("support calibrator has not been fit")
        payload = {
            "schema_version": "owl.cadc.phase4-support-calibrator.v1",
            "settings": {
                "k": self.k,
                "minimum_seeds": self.minimum_seeds,
                "minimum_decisions": self.minimum_decisions,
                "minimum_repeats": self.minimum_repeats,
                "maximum_disagreement": self.maximum_disagreement,
                "maximum_interval_width": self.maximum_interval_width,
                "shrinkage": self.shrinkage,
            },
            "thresholds": {
                "knn": self.knn_threshold,
                "mahalanobis": self.mahalanobis_threshold,
            },
            "neighbor_backend": self.neighbor_backend,
            "geometry_backend": self.geometry_backend,
            "center": self.center.tolist(),
            "precision": self.precision.tolist(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["digest"] = hashlib.sha256(encoded).hexdigest()
        return payload


def fit_knn_support(embeddings: Any, seeds: Any, *, k: int) -> SupportCalibrator:
    """Convenience fit using permissive decision thresholds for diagnostics."""
    return SupportCalibrator(
        k=k,
        minimum_seeds=1,
        minimum_decisions=1,
        minimum_repeats=1,
        maximum_disagreement=float("inf"),
        maximum_interval_width=float("inf"),
    ).fit(embeddings, seeds)
