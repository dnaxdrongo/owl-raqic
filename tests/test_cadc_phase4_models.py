from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from owl.cadc.models import ActionAgnosticBaseline, CADCMore2Suite  # noqa: E402
from owl.cadc.models.ranker import listwise_loss, pairwise_loss  # noqa: E402
from owl.cadc.models.transition import StructuralTransitionModel  # noqa: E402


def _model() -> CADCMore2Suite:
    return CADCMore2Suite(
        StructuralTransitionModel(
            context_dim=4,
            candidate_dim=3,
            direction_dim=2,
            hidden_dim=16,
            outcome_dim=20,
            quantile_count=7,
            time_bins=5,
            death_causes=4,
        ),
        16,
    )


def test_complete_suite_shapes_and_monotone_quantiles() -> None:
    model = _model()
    output = model(
        torch.zeros(3, 4),
        torch.zeros(3, 22, 3),
        torch.zeros(3, 2, 8, 2),
        torch.ones(3, 2, 8, dtype=torch.bool),
        torch.zeros(3, dtype=torch.long),
    )
    assert output["outcome_mean"].shape == (3, 22, 20)
    assert output["rank_score"].shape == (3, 22)
    assert output["competing_risk_logits"].shape == (3, 22, 5, 5)
    assert len(output["externality_head"]) == 5
    assert torch.all(output["return_quantiles"].diff(dim=-1) >= 0)


def test_rank_losses_are_finite_and_have_gradients() -> None:
    left = torch.tensor([1.0, 0.0], requires_grad=True)
    right = torch.tensor([0.0, 1.0], requires_grad=True)
    loss = pairwise_loss(left, right, torch.tensor([1.0, 0.0]))
    loss.backward()
    assert torch.isfinite(loss)
    scores = torch.zeros(2, 22, requires_grad=True)
    targets = torch.arange(22, dtype=torch.float32).repeat(2, 1)
    mask = torch.ones(2, 22, dtype=torch.bool)
    list_loss = listwise_loss(scores, targets, mask)
    list_loss.backward()
    assert torch.isfinite(list_loss)


def test_action_agnostic_baseline_accepts_context_only() -> None:
    baseline = ActionAgnosticBaseline(4, 1)
    assert baseline(torch.zeros(5, 4)).shape == (5, 1)


def test_complete_suite_loss_contract_is_finite_and_backpropagates() -> None:
    """Exercise the exact positional contract used by the target-GPU trainer."""

    script = Path(__file__).resolve().parents[1] / "scripts/train_cadc_phase4.py"
    spec = importlib.util.spec_from_file_location("cadc_phase4_training_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = _model()
    output = model(
        torch.zeros(2, 4),
        torch.zeros(2, 22, 3),
        torch.zeros(2, 2, 8, 2),
        torch.ones(2, 2, 8, dtype=torch.bool),
        torch.zeros(2, dtype=torch.long),
    )
    mask = torch.ones(2, 22, dtype=torch.bool)
    target = torch.zeros(2, 22, 20)
    target[..., 5] = 1.0
    target[..., 15] = 1.0
    quantile_levels = torch.tensor([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    loss = module._suite_loss(
        output,
        target,
        mask,
        0,
        torch.zeros(2, 22),
        torch.zeros(2, 22, 20),
        torch.zeros(2, 22, 7),
        torch.zeros(2, 22),
        quantile_levels,
        torch.tensor([0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
