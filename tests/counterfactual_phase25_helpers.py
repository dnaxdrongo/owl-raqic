from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from scripts.run_cadc_phase3_acceptance import synthetic_initial_state

from owl.core.actions import Action
from owl.core.config import load_config
from owl.counterfactual.rng_registry import branch_seed
from owl.counterfactual.scheduler import CounterfactualScheduler
from owl.counterfactual.source import CounterfactualSourceCollector
from owl.gpu.run_context import PersistentOWLDeviceRun


def source_run(tmp_path: Path) -> tuple[Any, Any, Any]:
    cfg = load_config("configs/cadc_phase3_phase25_numpy_smoke.yaml")
    collector = CounterfactualSourceCollector(
        cfg,
        "dd3b83d74a9e00aa1a206e83ccf8d1218e52c7b85014215838ba9f57810802fd",
        run_id="pytest-phase3",
        condition="synthetic",
    )
    run = PersistentOWLDeviceRun.from_config(
        cfg,
        initial_state=copy.deepcopy(synthetic_initial_state(cfg)),
        force_backend="numpy",
        output_root=tmp_path,
        counterfactual_observer=collector,
    )
    run.step()
    assert len(collector.sources) == 1
    return cfg, run, collector.sources[0]


def execute_action(tmp_path: Path, action: Action) -> tuple[Any, Any, Any, Any]:
    cfg, run, source = source_run(tmp_path)
    scheduler = CounterfactualScheduler(run, cfg)
    decision_id = source.decisions.materialize_ids(run.ds.backend)[0]
    seed = branch_seed(int(cfg.world.seed), source.state.source_state_id, 0)
    result = scheduler._execute_branch(  # noqa: SLF001 - exact branch contract test
        source,
        0,
        decision_id,
        int(action),
        0,
        seed,
        anchor=False,
    )
    return cfg, run, source, result
