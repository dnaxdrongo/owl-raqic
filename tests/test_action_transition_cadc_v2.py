from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from owl.core.actions import Action
from owl.core.config import SimulationConfig, load_config
from owl.gpu.run_context import PersistentOWLDeviceRun
from owl.record.cadc_schema import (
    CADC_ACTION_TRANSITION_SCHEMA_DIGEST,
    CADC_ACTION_TRANSITION_SCHEMA_VERSION,
    CADC_SCHEMA_DIGEST,
    CADCEventCode,
    ContributionCode,
    ReasonCode,
)
from owl.record.replay_recorder import ReplayRecorder
from owl.viz.visual_snapshot import snapshot_from_arrays
from tests.test_action_transition_execution_contract import _state


def _config(action: Action) -> SimulationConfig:
    base = load_config("configs/gpu_v07_persistent_small.yaml")
    data = base.model_dump(mode="json")
    data["initialization"]["population_density"] = 0.0
    data["debug"]["assert_invariants"] = False
    data["recording"]["cadc"]["enabled"] = True
    data["recording"]["cadc"]["include_dense_context"] = True
    data["recording"]["cadc"]["profile"] = "exact"
    data["actions"]["enabled_actions"] = [action.name]
    data["action_transitions"] = {
        "enabled": True,
        "action_contract_version": "owl.action-transitions.v1",
        "legacy_unsupported_action_recovery": False,
        "active_sense_enabled": True,
        "flee_execution_enabled": True,
        "pursue_execution_enabled": True,
    }
    return SimulationConfig.model_validate(data)


@pytest.mark.parametrize(
    ("action", "compiled"),
    ((Action.FLEE, Action.MOVE_E), (Action.PURSUE, Action.MOVE_NE)),
)
def test_factual_high_level_execution_keeps_selected_and_compiled_separate(
    tmp_path: Path, action: Action, compiled: Action
) -> None:
    cfg = _config(action)
    initial = _state(cfg, pursuit=action == Action.PURSUE)
    initial.tick = 0
    run = PersistentOWLDeviceRun.from_config(
        cfg,
        initial_state=initial,
        force_backend="numpy",
        output_root=tmp_path / action.name.lower(),
    )
    try:
        run.step()
        buffer = run.cadc_buffer
        assert buffer is not None
        y, x = 5, 5
        assert int(buffer.arrays["selected_action"][y, x]) == int(action)
        assert int(buffer.arrays["attempted_action"][y, x]) == int(action)
        assert int(buffer.arrays["realized_action"][y, x]) == int(action)
        assert int(buffer.arrays["compiled_execution_action"][y, x]) == int(compiled)
        assert bool(buffer.arrays["execution_success"][y, x])
        assert int(buffer.arrays["execution_reason_code"][y, x]) != int(
            ReasonCode.NO_EXECUTION_CONTRACT
        )
        target_slot = buffer.event_codes.index(int(CADCEventCode.ACTION_TARGET_ACQUIRED))
        movement_slot = buffer.event_codes.index(int(CADCEventCode.MOVEMENT_SUCCESS))
        flat = y * cfg.world.width + x
        assert bool(buffer.arrays["event_active"][target_slot, flat])
        assert bool(buffer.arrays["event_active"][movement_slot, flat])
        movement_contribution = buffer.contribution_codes.index(
            int(ContributionCode.MOVEMENT)
        )
        resource_delta = buffer.arrays["contribution_delta"][
            movement_contribution, y, x, 1
        ]
        assert resource_delta == pytest.approx(-cfg.resources.movement_cost, abs=2e-8)
    finally:
        run.close(checkpoint=False)


def test_v2_columnar_writer_records_sense_information_and_direction_rows(
    tmp_path: Path,
) -> None:
    cfg = _config(Action.SENSE)
    initial = _state(cfg, pursuit=False)
    initial.tick = 0
    run = PersistentOWLDeviceRun.from_config(
        cfg, initial_state=initial, force_backend="numpy", output_root=tmp_path / "science"
    )
    try:
        run.step()
        buffer = run.cadc_buffer
        assert buffer is not None
        assert buffer.schema_version == CADC_ACTION_TRANSITION_SCHEMA_VERSION
        assert buffer.schema_digest == CADC_ACTION_TRANSITION_SCHEMA_DIGEST
        assert buffer.schema_digest != CADC_SCHEMA_DIGEST
        arrays = {
            name: np.asarray(value)
            for name, value in run.ds.arrays.items()
            if not name.startswith("_") and getattr(value, "ndim", 0) >= 2
        }
        snapshot = snapshot_from_arrays(
            tick=int(run.ds.tick),
            boundary_mode=str(cfg.world.boundary_mode),
            arrays=arrays,
            events=(),
            metadata={"source": "cadc-action-v2-test"},
        )
        root = tmp_path / "bundle"
        recorder = ReplayRecorder(
            root,
            run_id="cadc-action-v2",
            condition="sense",
            seed=int(cfg.world.seed),
            requested_ticks=1,
            recording_tier="analysis_full",
            action_names=[action.name for action in Action],
            cadc_config=cfg.recording.cadc,
        )
        recorder.record_device(run.ds, snapshot, diagnostics={})
        recorder.close()
        cadc = root / "analysis" / "cadc_v2"
        decisions = pq.read_table(cadc / "decisions.parquet")
        candidates = pq.read_table(cadc / "candidates.parquet")
        directions = pq.read_table(cadc / "action_directions.parquet")
        execution = pq.read_table(cadc / "execution.parquet")
        information = pq.read_table(cadc / "information.parquet")
        assert candidates.num_rows == decisions.num_rows * 22
        assert directions.num_rows == decisions.num_rows * 16
        assert execution.num_rows == decisions.num_rows
        assert information.num_rows > 0
        assert execution.schema.metadata[b"owl.cadc.schema_digest"].decode() == (
            CADC_ACTION_TRANSITION_SCHEMA_DIGEST
        )
        assert np.all(information.column("information_execution_success").to_numpy())
        assert np.all(information.column("new_cell_count").to_numpy() >= 0)
        sense = candidates.column("action_index").to_numpy() == int(Action.SENSE)
        assert np.all(candidates.column("prechoice_executable").to_numpy()[sense])
        assert not np.any(
            candidates.column("prechoice_reason_code").to_numpy()[sense]
            == int(ReasonCode.NO_EXECUTION_CONTRACT)
        )
    finally:
        run.close(checkpoint=False)
