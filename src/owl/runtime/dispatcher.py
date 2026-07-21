from __future__ import annotations

from typing import Any

import numpy as np

from owl.core.init import initialize_world
from owl.engine.loop import step
from owl.record.metrics import collect_metrics

from .execution_plan import ExecutionPlan
from .run_result import RunResult


def _run_cpu_or_stage_once(
    cfg: Any, plan: ExecutionPlan, initial_state: Any | None = None
) -> RunResult:
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng) if initial_state is None else initial_state
    if getattr(cfg.raqic, "enabled", False):
        from owl.raqic.state import ensure_raqic_fields

        ensure_raqic_fields(state, cfg)
    metrics: list[dict[str, Any]] = []
    for _ in range(int(cfg.world.max_steps)):
        step(state, cfg, rng)
        metrics.append(collect_metrics(state, cfg))
    return RunResult(
        state=state,
        metrics=metrics,
        execution_plan=plan,
        execution_metadata={
            "simulation_backend": plan.simulation_backend,
            "device_state_instances": (
                int(cfg.world.max_steps) if plan.simulation_backend == "gpu_stage_once" else 0
            ),
            "checkpoint_count": 0,
            "fallback_count": 0,
        },
    )


def _run_persistent(cfg: Any, plan: ExecutionPlan, initial_state: Any | None = None) -> RunResult:
    from owl.gpu.run_context import PersistentOWLDeviceRun

    run = PersistentOWLDeviceRun.from_config(cfg, initial_state=initial_state, plan=plan)
    try:
        run.run(max_steps=int(cfg.world.max_steps), checkpoint_final=False)
        state = run.checkpoint()
        metadata = run.execution_metadata()
        return RunResult(
            state=state,
            metrics=list(run.metrics),
            execution_plan=plan,
            execution_metadata=metadata,
        )
    finally:
        run.close(checkpoint=False)


def _run_multi_gpu(cfg: Any, plan: ExecutionPlan, initial_state: Any | None = None) -> RunResult:
    if initial_state is not None:
        raise ValueError("distributed execution does not accept an in-memory initial_state")
    from owl.gpu.distributed.launch import run_distributed

    result = run_distributed(cfg, plan)
    return result


def dispatch_run(cfg: Any, plan: ExecutionPlan, *, initial_state: Any | None = None) -> RunResult:
    if plan.simulation_backend in {"cpu", "gpu_stage_once"}:
        return _run_cpu_or_stage_once(cfg, plan, initial_state=initial_state)
    if plan.simulation_backend in {"gpu_persistent", "gpu_graph"}:
        return _run_persistent(cfg, plan, initial_state=initial_state)
    if plan.simulation_backend == "gpu_multi":
        return _run_multi_gpu(cfg, plan, initial_state=initial_state)
    raise AssertionError(f"Unhandled execution backend: {plan.simulation_backend}")
