from __future__ import annotations

import numpy as np

from owl.cadc.calibration import (
    ConformalQuantileCalibrator,
    IsotonicValueCalibrator,
    TemperatureCalibrator,
)
from owl.cadc.support import SupportCalibrator


def test_conformal_interval_is_finite_and_has_requested_empirical_coverage() -> None:
    prediction = np.linspace(-1.0, 1.0, 200)
    target = prediction + np.sin(np.arange(200)) * 0.1
    calibrator = ConformalQuantileCalibrator(coverage=0.9, minimum_group=20).fit(
        prediction, target, np.repeat(["a", "b"], 100)
    )
    lower, upper = calibrator.interval(prediction, np.repeat(["a", "b"], 100))
    assert np.isfinite(lower).all() and np.isfinite(upper).all()
    assert np.mean((target >= lower) & (target <= upper)) >= 0.9


def test_temperature_is_positive_and_deterministic() -> None:
    logits = np.asarray([[2.0, -1.0], [-1.0, 2.0], [0.5, 0.2], [0.2, 0.5]])
    labels = np.asarray([0, 1, 0, 1])
    first = TemperatureCalibrator().fit(logits, labels)
    second = TemperatureCalibrator().fit(logits, labels)
    assert first.temperature > 0.0
    assert first.temperature == second.temperature


def test_isotonic_value_calibration_is_monotone_or_fails_to_identity() -> None:
    prediction = np.linspace(-2.0, 2.0, 100)
    target = prediction**3
    calibrator = IsotonicValueCalibrator(minimum_rows=20).fit(prediction, target)
    transformed = calibrator.transform(prediction)
    assert calibrator.status == "fit"
    assert np.all(np.diff(transformed) >= 0.0)
    identity = IsotonicValueCalibrator(minimum_rows=200).fit(prediction, target)
    assert np.array_equal(identity.transform(prediction), prediction)


def test_support_abstains_for_ood_and_low_repeat() -> None:
    rng = np.random.default_rng(8)
    embeddings = rng.normal(size=(80, 4))
    seeds = np.repeat(np.arange(8), 10)
    support = SupportCalibrator(
        k=5,
        minimum_seeds=2,
        minimum_decisions=10,
        minimum_repeats=4,
        maximum_disagreement=0.5,
        maximum_interval_width=2.0,
    ).fit(embeddings, seeds)
    low_repeat = support.decide(
        embeddings[0],
        action_support=80,
        repeat_support=1,
        disagreement=0.0,
        interval_width=0.1,
    )
    ood = support.decide(
        np.full(4, 1000.0),
        action_support=80,
        repeat_support=8,
        disagreement=0.0,
        interval_width=0.1,
    )
    assert low_repeat.abstention_reason.value == "low_repeat_support"
    assert ood.status.value == "out_of_distribution"
