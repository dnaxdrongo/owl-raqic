from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from owl.gpu.memory_model import build_counterfactual_memory_plan


def test_counterfactual_memory_plan_counts_source_and_branch_evidence() -> None:
    state_array = np.zeros(64, dtype=np.float32)
    evidence_array = np.zeros(32, dtype=np.int16)
    ds = SimpleNamespace(
        arrays={"state": state_array},
        patch_arrays={},
        global_arrays={},
        health=state_array,
        metadata={
            "cadc_device_buffer": SimpleNamespace(arrays={"evidence": evidence_array})
        },
    )
    counterfactual = SimpleNamespace(
        event_capacity_per_branch_tick=1,
        horizons=(1, 3),
        max_pending_bytes=1024,
        max_device_bytes=512 * 1024**2,
        memory_safety_fraction=0.7,
        max_active_branches=8,
    )
    cfg = SimpleNamespace(counterfactual=counterfactual)

    plan = build_counterfactual_memory_plan(ds, cfg, scratch_bytes=2048)

    assert plan.per_branch_evidence_bytes == evidence_array.nbytes
    assert plan.source_snapshot_bytes == state_array.nbytes + evidence_array.nbytes
    assert plan.per_branch_bytes >= state_array.nbytes + evidence_array.nbytes + 2048
