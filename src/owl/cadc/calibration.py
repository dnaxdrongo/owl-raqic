"""Independent probability, interval, and conformal calibration layers.

Temperature scaling and split-conformal intervals follow Guo et al. (2017)
and Angelopoulos and Bates (2023), respectively. See
``docs/REFERENCES.md`` [R33, R35].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


class TemperatureCalibrator:
    """One-parameter probability calibrator fit on calibration rows only."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: Any, labels: Any) -> TemperatureCalibrator:
        """Fit a positive scalar temperature on calibration-only rows."""
        values = np.asarray(logits, dtype=np.float64)
        truth = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2 or truth.shape != (values.shape[0],):
            raise ValueError("temperature calibration shapes are incompatible")
        if not np.isfinite(values).all():
            raise ValueError("temperature logits contain nonfinite values")
        candidates = np.exp(np.linspace(np.log(0.05), np.log(20.0), 401))
        losses = np.empty(candidates.size, dtype=np.float64)
        for index, temperature in enumerate(candidates):
            scaled = values / temperature
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            logsum = np.log(np.exp(shifted).sum(axis=1))
            losses[index] = np.mean(logsum - shifted[np.arange(truth.size), truth])
        self.temperature = float(candidates[np.argmin(losses)])
        return self

    def transform(self, logits: Any) -> npt.NDArray[Any]:
        """Apply the fitted temperature without changing class order."""
        return np.asarray(logits, dtype=np.float64) / self.temperature


class ConformalQuantileCalibrator:
    """Split-conformal symmetric intervals with optional Mondrian groups."""

    def __init__(self, *, coverage: float, minimum_group: int) -> None:
        if not 0.5 < coverage < 1.0 or minimum_group < 1:
            raise ValueError("invalid conformal settings")
        self.coverage = coverage
        self.minimum_group = minimum_group
        self.global_radius: float | None = None
        self.group_radius: dict[str, float] = {}

    def fit(
        self, predictions: Any, targets: Any, groups: Any | None = None
    ) -> ConformalQuantileCalibrator:
        """Fit finite-sample global and sufficiently supported group radii."""
        predicted = np.asarray(predictions, dtype=np.float64)
        truth = np.asarray(targets, dtype=np.float64)
        if predicted.shape != truth.shape or predicted.ndim != 1:
            raise ValueError("conformal inputs must be matching vectors")
        score = np.abs(truth - predicted)
        if not np.isfinite(score).all():
            raise ValueError("conformal residuals contain nonfinite values")
        self.global_radius = _finite_sample_quantile(score, self.coverage)
        self.group_radius = {}
        if groups is not None:
            labels = np.asarray(groups).astype(str)
            if labels.shape != score.shape:
                raise ValueError("conformal groups do not align with residuals")
            for group in np.unique(labels):
                selected = score[labels == group]
                if selected.size >= self.minimum_group:
                    self.group_radius[str(group)] = _finite_sample_quantile(
                        selected, self.coverage
                    )
        return self

    def interval(
        self, predictions: Any, groups: Any | None = None
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Return symmetric calibrated intervals for the supplied predictions."""
        if self.global_radius is None:
            raise RuntimeError("conformal calibrator has not been fit")
        values = np.asarray(predictions, dtype=np.float64)
        radius = np.full(values.shape, self.global_radius, dtype=np.float64)
        if groups is not None:
            labels = np.asarray(groups).astype(str)
            for group, value in self.group_radius.items():
                radius[labels == group] = value
        return values - radius, values + radius


class IsotonicValueCalibrator:
    """Monotone scalar-value calibration with an explicit minimum-count gate."""

    def __init__(self, *, minimum_rows: int) -> None:
        if minimum_rows < 2:
            raise ValueError("isotonic minimum rows must be at least two")
        self.minimum_rows = int(minimum_rows)
        self.model: Any | None = None
        self.status = "unfit"

    def fit(self, predictions: Any, targets: Any) -> IsotonicValueCalibrator:
        """Fit a monotone value map or retain a typed identity fallback."""
        predicted = np.asarray(predictions, dtype=np.float64)
        truth = np.asarray(targets, dtype=np.float64)
        if predicted.ndim != 1 or predicted.shape != truth.shape:
            raise ValueError("isotonic inputs must be matching vectors")
        if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
            raise ValueError("isotonic inputs contain nonfinite values")
        if predicted.size < self.minimum_rows or np.unique(predicted).size < 2:
            self.status = "insufficient_support_identity"
            self.model = None
            return self
        from sklearn.isotonic import IsotonicRegression

        self.model = IsotonicRegression(out_of_bounds="clip").fit(predicted, truth)
        self.status = "fit"
        return self

    def transform(self, predictions: Any) -> npt.NDArray[Any]:
        """Apply the fitted monotone map or its registered identity fallback."""
        values = np.asarray(predictions, dtype=np.float64)
        if self.status == "unfit":
            raise RuntimeError("isotonic calibrator has not been fit")
        if self.model is None:
            return values.copy()
        return np.asarray(self.model.predict(values.reshape(-1)), dtype=np.float64).reshape(
            values.shape
        )

    def manifest(self) -> dict[str, Any]:
        """Return the calibration status without serializing estimator internals."""
        return {
            "status": self.status,
            "minimum_rows": self.minimum_rows,
            "fitted": self.model is not None,
        }


def _finite_sample_quantile(values: npt.NDArray[Any], coverage: float) -> float:
    rank = int(np.ceil((values.size + 1) * coverage)) - 1
    rank = min(max(rank, 0), values.size - 1)
    return float(np.sort(values)[rank])


@dataclass
class CalibrationPipeline:
    """Frozen calibration container kept separate from model fitting."""

    temperature: TemperatureCalibrator
    conformal: ConformalQuantileCalibrator

    @classmethod
    def create(cls, *, coverage: float, minimum_group: int) -> CalibrationPipeline:
        """Construct an unfitted temperature-and-conformal pipeline."""
        return cls(
            TemperatureCalibrator(),
            ConformalQuantileCalibrator(coverage=coverage, minimum_group=minimum_group),
        )
