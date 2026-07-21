from __future__ import annotations

from owl.viz.frame_scheduler import VisualFrameScheduler, VisualScheduleMode, fixed_frame_plan
from owl.viz.render_budget import RenderBudget


def test_fixed_schedule_produces_exact_frame_count() -> None:
    plan = fixed_frame_plan(300, 1, 8)
    assert len(plan) == 2400
    assert plan[0].progress == 0.125
    assert plan[-1].scientific_tick == 300
    assert plan[-1].progress == 1.0


def test_fixed_scheduler_never_adapts_cadence() -> None:
    scheduler = VisualFrameScheduler(VisualScheduleMode.RECORD_FIXED, 8, 1, render_every=1)
    before = scheduler.requests_for_tick(1)
    scheduler.observe_live_cost(1.0, 1000.0)
    assert scheduler.render_every == 1
    assert scheduler.requests_for_tick(1) == before


def test_fixed_render_budget_never_changes() -> None:
    budget = RenderBudget(mode="record_fixed")
    budget.observe(simulation_ms=1.0, render_ms=1000.0)
    assert budget.render_every == 1
    assert budget.decorative_particle_fraction == 1.0
