from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from owl.core.actions import Action
from owl.core.config import load_config
from owl.gpu.run_context import PersistentOWLDeviceRun
from owl.record.cadc_schema import CADCEventCode, ReasonCode
from owl.record.replay_recorder import ReplayRecorder
from owl.viz.visual_snapshot import snapshot_from_arrays


def _run_and_record(tmp_path: Path) -> tuple[PersistentOWLDeviceRun, Path, object]:
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    cfg.debug.assert_invariants = False
    cfg.recording.cadc.enabled = True
    cfg.recording.cadc.include_dense_context = True
    cfg.recording.cadc.profile = "exact"
    run = PersistentOWLDeviceRun.from_config(
        cfg, force_backend="numpy", output_root=tmp_path / "scientific"
    )
    run.step()
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
        metadata={"source": "cadc-test"},
    )
    root = tmp_path / "bundle"
    recorder = ReplayRecorder(
        root,
        run_id="cadc-test",
        condition="all_on",
        seed=int(cfg.world.seed),
        requested_ticks=1,
        recording_tier="analysis_full",
        action_names=[action.name for action in Action],
        cadc_config=cfg.recording.cadc,
    )
    recorder.record_device(run.ds, snapshot, diagnostics={})
    manifest = recorder.close()
    return run, root, manifest


def test_cadc_replay_tables_have_exact_cardinality_and_join_keys(tmp_path: Path) -> None:
    run, root, manifest = _run_and_record(tmp_path)
    try:
        assert manifest.completed_ticks == 1
        cadc = root / "analysis" / "cadc_v1"
        decisions = pq.read_table(cadc / "decisions.parquet")
        candidates = pq.read_table(cadc / "candidates.parquet")
        execution = pq.read_table(cadc / "execution.parquet")
        contributions = pq.read_table(cadc / "contributions.parquet")
        dense = pq.read_table(cadc / "dense_context.parquet")

        assert decisions.num_rows > 0
        assert candidates.num_rows == decisions.num_rows * 22
        assert execution.num_rows == decisions.num_rows
        assert contributions.num_rows == decisions.num_rows * len(
            run.cadc_buffer.contribution_codes
        )
        count = len(run.cadc_buffer.contribution_codes)
        for name in run.cadc_buffer.contribution_fields:
            delta = contributions.column(f"delta_{name}").to_numpy().reshape(-1, count)
            start = contributions.column(f"start_{name}").to_numpy().reshape(-1, count)[:, 0]
            end = contributions.column(f"end_{name}").to_numpy().reshape(-1, count)[:, 0]
            np.testing.assert_allclose(delta.sum(axis=1), end - start, atol=1e-6, rtol=0)
        assert dense.num_rows == decisions.num_rows
        np.testing.assert_array_equal(
            decisions.column("dense_context_ref").to_numpy(),
            dense.column("dense_context_id").to_numpy(),
        )
        assert np.all(dense.column("radius").to_numpy() == 1)
        decision_ids = decisions.column("decision_sequence").to_numpy()
        assert np.unique(decision_ids).size == decisions.num_rows
        candidate_ids = candidates.column("candidate_sequence").to_numpy()
        assert np.unique(candidate_ids).size == candidates.num_rows
        actions = candidates.column("action_index").to_numpy().reshape(-1, 22)
        np.testing.assert_array_equal(actions[0], np.arange(22, dtype=np.int16))
        assert np.all(execution.column("execution_reason_code").to_numpy() >= 0)
        sense = candidates.column("action_index").to_numpy() == int(Action.SENSE)
        assert not np.any(candidates.column("prechoice_executable").to_numpy()[sense])
        assert np.all(
            candidates.column("prechoice_reason_code").to_numpy()[sense]
            == int(ReasonCode.NO_EXECUTION_CONTRACT)
        )
    finally:
        run.close(checkpoint=False)


def test_cadc_events_have_deterministic_unique_sequences(tmp_path: Path) -> None:
    run, root, _ = _run_and_record(tmp_path)
    try:
        event_root = root / "analysis" / "cadc_v1" / "events.parquet"
        parts = sorted(event_root.glob("part-*.parquet"))
        if not parts:
            return
        events = pq.read_table(event_root)
        sequence = events.column("event_sequence").to_numpy()
        assert np.unique(sequence).size == events.num_rows
        assert np.all(np.diff(sequence) > 0)
        assert np.all(events.column("event_code").to_numpy() > 0)
    finally:
        run.close(checkpoint=False)


def test_cadc_hot_path_has_no_candidate_or_ow_row_construction() -> None:
    repo = Path(__file__).resolve().parents[1]
    sources = [
        repo / "src/owl/record/cadc_capture.py",
        repo / "src/owl/record/cadc_writer.py",
        repo / "src/owl/record/gpu_replay_staging.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "Table.from_pylist" not in combined
    assert ".item()" not in combined
    assert ".tolist()" not in combined
    tree = ast.parse((repo / "src/owl/record/cadc_writer.py").read_text(encoding="utf-8"))
    record = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "record"
    )
    rendered_loops = [
        ast.unparse(node.target) for node in ast.walk(record) if isinstance(node, ast.For)
    ]
    assert all(target not in {"ow", "candidate", "row"} for target in rendered_loops)


def test_cadc_records_exact_toxin_damage_and_preclear_death_identity(tmp_path: Path) -> None:
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    cfg.debug.assert_invariants = False
    cfg.recording.cadc.enabled = True
    cfg.actions.enabled_actions = ["REST"]
    run = PersistentOWLDeviceRun.from_config(
        cfg, force_backend="numpy", output_root=tmp_path / "death-scientific"
    )
    try:
        y, x = np.argwhere(run.ds.health > 0)[0]
        ow_id = int(run.ds.occupancy[y, x])
        run.ds.health[y, x] = np.float32(0.001)
        run.ds.toxin[y, x] = np.float32(1.0)
        run.ds.toxin_resistance[y, x] = np.float32(0.0)
        run.ds.boundary[y, x] = np.float32(0.5)
        run.ds.integration[y, x] = np.float32(0.5)
        run.step()
        buffer = run.cadc_buffer
        toxin_slot = buffer.event_codes.index(int(CADCEventCode.TOXIN_DAMAGE_EVIDENCE))
        death_slot = buffer.event_codes.index(int(CADCEventCode.DEATH))
        assert np.any(buffer.arrays["event_active"][toxin_slot])
        assert np.any(buffer.arrays["event_active"][death_slot])
        death_index = np.flatnonzero(buffer.arrays["event_active"][death_slot])[0]
        assert int(buffer.arrays["pre_ow_id"].reshape(-1)[death_index]) == ow_id
        assert int(buffer.arrays["event_target_ow_id"][death_slot, death_index]) == ow_id
        assert float(buffer.arrays["event_payload"][toxin_slot, death_index, 0]) > 0
    finally:
        run.close(checkpoint=False)
