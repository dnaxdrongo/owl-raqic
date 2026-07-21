from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from owl.core.actions import Action
from owl.core.config import load_config
from owl.gpu.run_context import PersistentOWLDeviceRun
from owl.record.replay_recorder import ReplayRecorder
from owl.replay.zarr_source import ZarrReplayDataSource
from owl.viz.visual_snapshot import snapshot_from_arrays


def _snapshot(run: PersistentOWLDeviceRun, boundary_mode: str) -> object:
    arrays = {
        name: np.asarray(value)
        for name, value in run.ds.arrays.items()
        if not name.startswith("_") and getattr(value, "ndim", 0) >= 2
    }
    return snapshot_from_arrays(
        tick=int(run.ds.tick),
        boundary_mode=boundary_mode,
        arrays=arrays,
        events=(),
        metadata={"source": "cadc-information-test"},
    )


def _make(tmp_path: Path, *, requested_ticks: int) -> tuple[object, object, ReplayRecorder]:
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    cfg.debug.assert_invariants = False
    cfg.recording.cadc.enabled = True
    cfg.recording.cadc.include_dense_context = True
    cfg.recording.cadc.profile = "exact"
    run = PersistentOWLDeviceRun.from_config(
        cfg, force_backend="numpy", output_root=tmp_path / "scientific"
    )
    recorder = ReplayRecorder(
        tmp_path / "bundle",
        run_id="cadc-information",
        condition="all_on",
        seed=int(cfg.world.seed),
        requested_ticks=requested_ticks,
        recording_tier="analysis_full",
        action_names=[action.name for action in Action],
        cadc_config=cfg.recording.cadc,
    )
    return cfg, run, recorder


def _step_and_record(cfg: object, run: object, recorder: ReplayRecorder) -> None:
    run.step()
    recorder.record_device(
        run.ds,
        _snapshot(run, str(cfg.world.boundary_mode)),
        diagnostics={},
    )


def test_information_records_link_to_next_decision_or_terminal(tmp_path: Path) -> None:
    cfg, run, recorder = _make(tmp_path, requested_ticks=2)
    try:
        _step_and_record(cfg, run, recorder)
        _step_and_record(cfg, run, recorder)
        recorder.close()
        root = tmp_path / "bundle" / "analysis" / "cadc_v1"
        information = pq.read_table(root / "information.parquet")
        followups = pq.read_table(root / "information_followups.parquet")

        tick = information.column("tick").to_numpy()
        first = tick == 1
        assert np.count_nonzero(first) > 0
        sequence = information.column("decision_sequence").to_numpy()
        selected = information.column("information_kind").to_numpy()
        np.testing.assert_array_equal(
            information.column("pre_observation_ref").to_numpy(), sequence
        )
        np.testing.assert_array_equal(
            information.column("post_memory_ref").to_numpy(), sequence
        )
        assert set(np.unique(selected)).issubset({int(Action.SENSE), int(Action.COMMUNICATE)})
        assert np.all(information.column("timing_code").to_numpy() == 1)
        receiver_count = information.column("receiver_count").to_numpy()
        receiver_status = information.column("receiver_link_status").to_numpy()
        assert np.all(receiver_count[selected == int(Action.SENSE)] == 0)
        assert np.all(receiver_status[selected == int(Action.SENSE)] == 1)
        assert np.all(receiver_count[selected == int(Action.COMMUNICATE)] == -1)
        assert np.all(receiver_status[selected == int(Action.COMMUNICATE)] == 2)
        for field in (
            "observation_before",
            "memory_before",
            "memory_after",
            "emitted_channels",
            "received_channels",
        ):
            values = information.column(field).combine_chunks().values
            assert len(values) == information.num_rows * run.cadc_buffer.channel_count

        source = followups.column("source_decision_sequence").to_numpy()
        linked = np.isin(source, sequence[first])
        assert np.count_nonzero(linked) == np.count_nonzero(first)
        followup_decision = followups.column("followup_decision_sequence").to_numpy()[linked]
        followup_observation = followups.column("followup_observation_ref").to_numpy()[linked]
        np.testing.assert_array_equal(followup_observation, followup_decision)
        assert set(followups.column("followup_status").to_numpy()[linked]).issubset({1, 2})
    finally:
        run.close(checkpoint=False)


def test_information_pending_links_survive_replay_resume(tmp_path: Path) -> None:
    cfg, run, recorder = _make(tmp_path, requested_ticks=2)
    try:
        _step_and_record(cfg, run, recorder)
        first_path = tmp_path / "bundle" / "analysis" / "cadc_v1" / "information.parquet"
        first_sequences = pq.read_table(first_path).column("decision_sequence").to_numpy()
        recorder.close(state="INTERRUPTED_RESUMABLE", failure="test boundary")

        resumed = ReplayRecorder.resume(tmp_path / "bundle")
        _step_and_record(cfg, run, resumed)
        manifest = resumed.close()
        assert manifest.completed_ticks == 2
        followups = pq.read_table(
            tmp_path
            / "bundle"
            / "analysis"
            / "cadc_v1"
            / "information_followups.parquet"
        )
        source = followups.column("source_decision_sequence").to_numpy()
        linked = np.isin(source, first_sequences)
        assert np.count_nonzero(linked) == first_sequences.size
        assert set(followups.column("followup_status").to_numpy()[linked]).issubset({1, 2})
    finally:
        run.close(checkpoint=False)


def test_replay_discovers_cadc_tables_without_breaking_optional_schema(tmp_path: Path) -> None:
    cfg, run, recorder = _make(tmp_path, requested_ticks=1)
    try:
        _step_and_record(cfg, run, recorder)
        recorder.close()
        source = ZarrReplayDataSource(tmp_path / "bundle")
        assert source.cadc_manifest is not None
        assert "candidates" in source.available_cadc_tables()
        candidates = source.load_cadc_table(
            "candidates", tick=1, columns=("tick", "ow_id", "action_index")
        )
        assert candidates is not None
        assert candidates.num_rows > 0
        assert source.load_cadc_table("not_a_table") is None
    finally:
        run.close(checkpoint=False)
