"""Frozen, urgency-aware scalar projections over preserved raw outcomes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


def stabilized_softmax(values: Any, *, axis: int = -1, beta: float = 1.0) -> Any:
    """Compute a backend-native max-shifted softmax."""
    xp = _array_namespace(values)
    scaled = xp.asarray(values) * beta
    shifted = scaled - xp.max(scaled, axis=axis, keepdims=True)
    exponent = xp.exp(shifted)
    denominator = xp.sum(exponent, axis=axis, keepdims=True)
    return exponent / xp.maximum(denominator, xp.asarray(1e-30, dtype=exponent.dtype))


def _array_namespace(value: Any) -> Any:
    if type(value).__module__.split(".", maxsplit=1)[0] == "cupy":
        import cupy as cp

        return cp
    return np


@dataclass(frozen=True)
class HomeostaticDrive:
    """Fold-fitted setpoint/scale/asymmetry definition for urgency weighting."""

    names: tuple[str, ...]
    setpoints: tuple[float, ...]
    scales: tuple[float, ...]
    lower_asymmetry: tuple[float, ...]
    upper_asymmetry: tuple[float, ...]
    urgency_beta: float = 2.0

    def __post_init__(self) -> None:
        lengths = {
            len(self.names),
            len(self.setpoints),
            len(self.scales),
            len(self.lower_asymmetry),
            len(self.upper_asymmetry),
        }
        if len(lengths) != 1 or not self.names:
            raise ValueError("homeostatic drive vectors must be nonempty and equal length")
        if any(value <= 0 for value in self.scales):
            raise ValueError("homeostatic scales must be positive")
        if any(value < 0 for value in (*self.lower_asymmetry, *self.upper_asymmetry)):
            raise ValueError("homeostatic asymmetry weights must be nonnegative")

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 identity of this drive definition."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def urgency(self, state: Any) -> tuple[Any, Any]:
        """Return per-component deviations and normalized urgency weights."""
        xp = _array_namespace(state)
        values = xp.asarray(state)
        setpoints = xp.asarray(self.setpoints, dtype=values.dtype)
        scales = xp.asarray(self.scales, dtype=values.dtype)
        lower = xp.asarray(self.lower_asymmetry, dtype=values.dtype)
        upper = xp.asarray(self.upper_asymmetry, dtype=values.dtype)
        standardized = (values - setpoints) / scales
        penalties = xp.where(standardized < 0, -standardized * lower, standardized * upper)
        return standardized, stabilized_softmax(
            penalties, axis=-1, beta=self.urgency_beta
        )

    def drive(self, state: Any) -> Any:
        """Return the positive urgency-weighted distance from setpoint."""
        xp = _array_namespace(state)
        standardized, weights = self.urgency(state)
        lower = xp.asarray(self.lower_asymmetry, dtype=standardized.dtype)
        upper = xp.asarray(self.upper_asymmetry, dtype=standardized.dtype)
        magnitude = xp.where(standardized < 0, -standardized * lower, standardized * upper)
        return xp.sum(weights * magnitude, axis=-1)

    def improvement(self, source: Any, outcome_delta: Any) -> Any:
        """Return source drive minus post-outcome drive; positive is improvement."""
        return self.drive(source) - self.drive(source + outcome_delta)


@dataclass(frozen=True)
class ScalarizationProfile:
    """Named, frozen weights over the registered vector outcomes."""
    name: str
    weights: tuple[tuple[str, float], ...]
    death_penalty: float = 0.0
    cvar_penalty: float = 0.0
    information_weight: float = 0.0
    externality_weight: float = 0.0

    def weight_map(self) -> dict[str, float]:
        """Return the outcome-name to scalar-weight mapping."""
        return dict(self.weights)


def default_profiles() -> tuple[ScalarizationProfile, ...]:
    """Return the pre-registered agent, oracle, and collective profiles."""
    base = (
        ("homeostatic_improvement", 1.0),
        ("health_delta", 0.5),
        ("resource_delta", 0.35),
        ("boundary_delta", 0.15),
        ("integration_delta", 0.15),
        ("memory_delta", 0.1),
    )
    return (
        ScalarizationProfile("agent_risk_neutral", base, information_weight=0.2),
        ScalarizationProfile(
            "agent_risk_averse",
            base,
            death_penalty=4.0,
            cvar_penalty=1.0,
            information_weight=0.2,
        ),
        ScalarizationProfile(
            "oracle_diagnostic",
            base,
            death_penalty=2.0,
            information_weight=0.5,
        ),
        ScalarizationProfile(
            "collective_balanced",
            base,
            death_penalty=2.0,
            information_weight=0.2,
            externality_weight=0.5,
        ),
    )


class ScalarizationRegistry:
    """Immutable scalarization profiles plus sensitivity-grid identity."""

    def __init__(self, profiles: Sequence[ScalarizationProfile] | None = None) -> None:
        values = tuple(profiles) if profiles is not None else default_profiles()
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ValueError("scalarization profile names must be unique")
        self._profiles = values

    @property
    def profiles(self) -> tuple[ScalarizationProfile, ...]:
        """Return the immutable ordered scalarization profiles."""
        return self._profiles

    @property
    def digest(self) -> str:
        """Return the canonical SHA-256 identity of all profiles."""
        payload = json.dumps(
            [asdict(value) for value in self._profiles],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def get(self, name: str) -> ScalarizationProfile:
        """Resolve one profile by exact registered name."""
        matches = [value for value in self._profiles if value.name == name]
        if len(matches) != 1:
            raise KeyError(f"unknown scalarization profile: {name}")
        return matches[0]


def lower_tail_cvar(values: Any, *, alpha: float, axis: int = -1) -> Any:
    """Return empirical lower-tail CVaR with an explicit alpha contract."""
    if not 0 < alpha <= 0.5:
        raise ValueError("CVaR alpha must be in (0, 0.5]")
    xp = _array_namespace(values)
    ordered = xp.sort(xp.asarray(values), axis=axis)
    count = ordered.shape[axis]
    tail = max(1, int(np.ceil(alpha * count)))
    slices = [slice(None)] * ordered.ndim
    slices[axis] = slice(0, tail)
    return xp.mean(ordered[tuple(slices)], axis=axis)


def quantile_cvar_weights(
    quantile_levels: Sequence[float], *, alpha: float
) -> npt.NDArray[Any]:
    """Return frozen trapezoidal weights for a lower-tail quantile integral.

    The first available lower-tail quantile is extended to probability zero and
    the last one is extended to ``alpha``.  This makes the approximation explicit,
    normalized, translation equivariant, and independent of the compute backend.
    """
    levels = np.asarray(tuple(quantile_levels), dtype=np.float64)
    if levels.ndim != 1 or levels.size < 1:
        raise ValueError("quantile grid must be one-dimensional and nonempty")
    if not 0.0 < alpha <= 0.5:
        raise ValueError("CVaR alpha must be in (0, 0.5]")
    if not np.all(np.diff(levels) > 0.0):
        raise ValueError("quantile grid must be strictly increasing")
    if levels[0] <= 0.0 or levels[-1] >= 1.0:
        raise ValueError("quantile grid must lie inside (0,1)")
    keep = levels <= alpha
    if not keep.any():
        raise ValueError("quantile grid has no point at or below CVaR alpha")
    lower = levels[keep]
    weights = np.zeros(levels.size, dtype=np.float64)
    weights[0] += lower[0]
    for index in range(1, lower.size):
        width = lower[index] - lower[index - 1]
        weights[index - 1] += 0.5 * width
        weights[index] += 0.5 * width
    weights[lower.size - 1] += alpha - lower[-1]
    weights /= alpha
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("CVaR quadrature weights do not sum to one")
    return weights


def quantile_cvar(
    quantiles: Any,
    quantile_levels: Sequence[float],
    *,
    alpha: float,
) -> Any:
    """Approximate lower-tail CVaR from a configured monotone quantile grid."""
    xp = _array_namespace(quantiles)
    values = xp.asarray(quantiles)
    weights = quantile_cvar_weights(quantile_levels, alpha=alpha)
    if values.shape[-1] != weights.size:
        raise ValueError("quantile values and configured grid have different widths")
    return xp.sum(values * xp.asarray(weights, dtype=values.dtype), axis=-1)


def scalarize(
    raw: Mapping[str, Any],
    profile: ScalarizationProfile,
    *,
    cvar: Any | None = None,
) -> Any:
    """Project raw outcomes without mutating or discarding their components."""
    if not raw:
        raise ValueError("raw outcome mapping cannot be empty")
    first = next(iter(raw.values()))
    xp = _array_namespace(first)
    result = xp.zeros_like(xp.asarray(first), dtype=xp.float64)
    for name, weight in profile.weights:
        if name not in raw:
            raise KeyError(f"scalarization input missing: {name}")
        result = result + float(weight) * xp.asarray(raw[name], dtype=xp.float64)
    if "death_by_horizon" in raw:
        result = result - profile.death_penalty * xp.asarray(
            raw["death_by_horizon"], dtype=xp.float64
        )
    if cvar is not None:
        result = result - profile.cvar_penalty * xp.asarray(cvar, dtype=xp.float64)
    if "information_control_value" in raw:
        result = result + profile.information_weight * xp.asarray(
            raw["information_control_value"], dtype=xp.float64
        )
    if "externality_value" in raw:
        result = result + profile.externality_weight * xp.asarray(
            raw["externality_value"], dtype=xp.float64
        )
    return result


def candidate_advantage(value_a: Any, value_b: Any) -> Any:
    """Return the matched-repeat candidate advantage A minus B."""
    xp = _array_namespace(value_a)
    left = xp.asarray(value_a)
    right = xp.asarray(value_b)
    if left.shape != right.shape:
        raise ValueError("paired candidate values must have identical shapes")
    return left - right
