from __future__ import annotations

import numpy as np

from owl.counterfactual.scheduler import BranchStatus, CounterfactualScheduler
from owl.gpu.stages.topdown_gpu import _clip_backend_scalar
from tests.counterfactual_phase25_helpers import source_run


class _StrictClipNamespace:
    """Model CuPy's refusal to dispatch ``clip`` from a Python float."""

    @staticmethod
    def asarray(value):
        return np.asarray(value)

    @staticmethod
    def clip(value, lower, upper):
        if isinstance(value, float):
            raise AttributeError("'float' object has no attribute 'clip'")
        return np.clip(value, lower, upper)


def test_backend_scalar_clip_normalizes_python_float() -> None:
    result = _clip_backend_scalar(_StrictClipNamespace, 1.25)
    assert np.asarray(result).shape == ()
    assert float(result) == 1.0


def test_source_and_branch_preserve_deferred_device_metric_mode(tmp_path) -> None:
    cfg, run, source = source_run(tmp_path)
    try:
        assert source.state.metadata["defer_host_metrics"] is True
        scheduler = CounterfactualScheduler(run, cfg)
        branch = scheduler._branch_context(source, int(cfg.world.seed))  # noqa: SLF001
        try:
            assert branch.ds.metadata["defer_host_metrics"] is True
        finally:
            branch.close(checkpoint=False)
    finally:
        run.close(checkpoint=False)


def test_branch_failure_retains_actionable_traceback(tmp_path, monkeypatch) -> None:
    cfg, run, source = source_run(tmp_path)

    def fail_injection(*_args, **_kwargs):
        raise RuntimeError("diagnostic sentinel")

    monkeypatch.setattr("owl.counterfactual.scheduler.inject_forced_actions", fail_injection)
    try:
        scheduler = CounterfactualScheduler(run, cfg)
        decision_id = source.decisions.materialize_ids(run.ds.backend)[0]
        selected = int(source.decisions.selected_action[0])
        result = scheduler._execute_branch(  # noqa: SLF001 - diagnostic contract
            source,
            0,
            decision_id,
            selected,
            -1,
            int(cfg.world.seed),
            anchor=True,
        )
        assert result.status == BranchStatus.FAILED
        assert result.failure == "RuntimeError: diagnostic sentinel"
        assert any("fail_injection" in line for line in result.failure_traceback)
    finally:
        run.close(checkpoint=False)
