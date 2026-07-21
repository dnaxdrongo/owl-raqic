from __future__ import annotations

import numpy as np

from owl.cadc.evaluation import NegativeControlRunner, evaluate_rankings


def _slow_ranking_reference(
    predicted: np.ndarray, truth: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    negative = np.finfo(np.float64).min
    predicted_best = np.argmax(np.where(valid, predicted, negative), axis=1)
    true_best = np.argmax(np.where(valid, truth, negative), axis=1)
    pair_correct = 0
    pair_total = 0
    pair_log_loss = 0.0
    topk = {1: 0.0, 3: 0.0, 5: 0.0}
    ndcg = {1: 0.0, 3: 0.0, 5: 0.0}
    spearman = []
    kendall = []
    for left in range(22):
        for right in range(left + 1, 22):
            selected = valid[:, left] & valid[:, right]
            predicted_order = predicted[selected, left] > predicted[selected, right]
            true_order = truth[selected, left] > truth[selected, right]
            non_tie = truth[selected, left] != truth[selected, right]
            pair_correct += int(np.sum((predicted_order == true_order) & non_tie))
            pair_total += int(non_tie.sum())
            delta = np.clip(
                predicted[selected, left] - predicted[selected, right], -60.0, 60.0
            )
            probability = 1.0 / (1.0 + np.exp(-delta))
            loss = -(
                true_order * np.log(np.clip(probability, 1e-12, 1.0))
                + (~true_order) * np.log(np.clip(1.0 - probability, 1e-12, 1.0))
            )
            pair_log_loss += float(loss[non_tie].sum())
    for row in range(predicted.shape[0]):
        selected = np.flatnonzero(valid[row])
        predicted_order = selected[
            np.argsort(-predicted[row, selected], kind="stable")
        ]
        truth_order = selected[np.argsort(-truth[row, selected], kind="stable")]
        for cutoff in (1, 3, 5):
            width = min(cutoff, selected.size)
            topk[cutoff] += float(true_best[row] in predicted_order[:width])
            relevance = truth[row, predicted_order[:width]]
            ideal = truth[row, truth_order[:width]]
            floor = float(truth[row, selected].min())
            discount = 1.0 / np.log2(np.arange(width, dtype=np.float64) + 2.0)
            denominator = float(np.sum((ideal - floor) * discount))
            ndcg[cutoff] += (
                float(np.sum((relevance - floor) * discount)) / denominator
                if denominator > 1e-12
                else 1.0
            )
        left_rank = np.argsort(
            np.argsort(predicted[row, selected], kind="stable"), kind="stable"
        )
        right_rank = np.argsort(
            np.argsort(truth[row, selected], kind="stable"), kind="stable"
        )
        spearman.append(float(np.corrcoef(left_rank, right_rank)[0, 1]))
        concordant = 0
        discordant = 0
        for left in range(selected.size):
            for right in range(left + 1, selected.size):
                product = (
                    predicted[row, selected[left]] - predicted[row, selected[right]]
                ) * (truth[row, selected[left]] - truth[row, selected[right]])
                concordant += int(product > 0.0)
                discordant += int(product < 0.0)
        if concordant + discordant:
            kendall.append((concordant - discordant) / (concordant + discordant))
    return {
        "top1_accuracy": float(np.mean(predicted_best == true_best)),
        "mean_regret": float(
            np.mean(
                truth[np.arange(truth.shape[0]), true_best]
                - truth[np.arange(truth.shape[0]), predicted_best]
            )
        ),
        "pairwise_accuracy": pair_correct / pair_total,
        "pairwise_log_loss": pair_log_loss / pair_total,
        "spearman": float(np.mean(spearman)),
        "kendall_tau": float(np.mean(kendall)),
        **{f"top{k}_contains_best": value / predicted.shape[0] for k, value in topk.items()},
        **{f"ndcg_at_{k}": value / predicted.shape[0] for k, value in ndcg.items()},
    }


def test_negative_control_shuffles_are_deterministic_and_stratified() -> None:
    runner = NegativeControlRunner(91)
    values = np.arange(12)
    strata = np.repeat(np.arange(3), 4)
    first = runner.action_shuffle(values, strata)
    second = runner.action_shuffle(values, strata)
    assert np.array_equal(first, second)
    for group in range(3):
        selected = strata == group
        assert set(first[selected]) == set(values[selected])


def test_temporal_break_never_crosses_safe_strata() -> None:
    runner = NegativeControlRunner(91)
    values = np.asarray([[1], [2], [10], [20]])
    order = np.asarray(["0001", "0002", "0001", "0002"])
    strata = np.asarray(["a", "a", "b", "b"])
    shifted = runner.temporal_break(values, order, strata)
    assert shifted[:, 0].tolist() == [2, 1, 20, 10]


def test_ranking_metrics_reject_single_candidate_rows() -> None:
    mask = np.zeros((1, 22), dtype=bool)
    mask[0, 0] = True
    try:
        evaluate_rankings(np.zeros((1, 22)), np.zeros((1, 22)), mask)
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:
        raise AssertionError("single-candidate ranking unexpectedly passed")


def test_ranking_metrics_report_pairwise_and_topk_contracts() -> None:
    target = np.arange(22, dtype=np.float64)[None, :]
    mask = np.ones((1, 22), dtype=bool)
    result = evaluate_rankings(target, target, mask)
    assert result["top1_accuracy"] == 1.0
    assert result["ndcg_at_5"] == 1.0
    assert result["top3_contains_best"] == 1.0
    assert result["kendall_tau"] == 1.0


def test_vectorized_ranking_metrics_match_slow_reference() -> None:
    rng = np.random.default_rng(8127)
    predicted = rng.normal(size=(19, 22))
    truth = rng.normal(size=(19, 22))
    truth[:, 3] = truth[:, 2]
    mask = rng.random((19, 22)) > 0.35
    mask[:, :2] = True
    expected = _slow_ranking_reference(predicted, truth, mask)
    actual = evaluate_rankings(predicted, truth, mask)
    for name, value in expected.items():
        assert np.isclose(actual[name], value, atol=1e-12, rtol=1e-12), name
