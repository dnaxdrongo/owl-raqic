from __future__ import annotations

from pathlib import Path

import numpy as np

from owl.record.replay_recorder import ReplayRecorder
from owl.replay.zarr_source import ZarrReplayDataSource
from owl.viz.event_bus import VisualEvent, VisualEventType
from owl.viz.visual_snapshot import snapshot_from_arrays


def _snapshot(tick: int) -> object:
    shape = (8, 8)
    health: np.ndarray = np.zeros(shape, dtype=np.float32)
    resource: np.ndarray = np.zeros(shape, dtype=np.float32)
    occupancy: np.ndarray = np.full(shape, -1, dtype=np.int64)
    y, x = 2, 1 + tick
    health[y, x] = 1.0
    resource[y, x] = 0.75
    occupancy[y, x] = 101
    arrays = {
        "health": health,
        "resource": resource,
        "occupancy": occupancy,
        "obstacle": np.zeros(shape, dtype=bool),
        "readout": np.zeros(shape, dtype=np.int16),
        "raqic_readout": np.zeros(shape, dtype=np.int16),
        "integration": health * 0.8,
        "lineage_id": np.where(health > 0, 7, -1).astype(np.int64),
        "parent_id": np.where(health > 0, 55, -1).astype(np.int64),
        "age": health * tick,
        "development_stage": health * 0.5,
        "raqic_record_confidence": health * 0.9,
        "raqic_probabilities": np.eye(22, dtype=np.float32)[np.zeros(shape, dtype=int)],
        "possibility": np.eye(22, dtype=np.float32)[np.zeros(shape, dtype=int)],
        "last_utilities": np.broadcast_to(np.linspace(-1, 1, 22, dtype=np.float32), (*shape, 22)),
        "pre_utilities": np.broadcast_to(
            np.linspace(-0.5, 0.5, 22, dtype=np.float32), (*shape, 22)
        ),
        "raqic_score": np.broadcast_to(np.linspace(0, 2, 22, dtype=np.float64), (*shape, 22)),
        "raqic_phase": np.broadcast_to(np.linspace(-3.0, 3.0, 22, dtype=np.float64), (*shape, 22)),
        "raqic_pre_mixer_probabilities": np.eye(22, dtype=np.float64)[np.zeros(shape, dtype=int)],
        "raqic_utility_innovation": np.zeros((*shape, 22), dtype=np.float64),
        "raqic_resonant_parent_intention": np.zeros((*shape, 22), dtype=np.float64),
        "_authority_bool": np.ones((*shape, 22), dtype=bool),
        "raqic_utility_innovation_norm": np.zeros(shape, dtype=np.float64),
        "raqic_interference_delta_l1": np.zeros(shape, dtype=np.float64),
    }
    event = VisualEvent(
        tick=tick,
        event_type=VisualEventType.MOVE,
        y=y,
        x=x,
        action=1,
        source_id=101,
    )
    return snapshot_from_arrays(
        tick=tick,
        boundary_mode="toroidal",
        arrays=arrays,
        events=(event,),
        metadata={"source": "test"},
    )


def test_replay_bundle_round_trip(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    recorder = ReplayRecorder(
        bundle,
        run_id="synthetic",
        condition="all_on",
        seed=9303,
        requested_ticks=3,
        recording_tier="analysis_full",
        action_names=[f"ACTION_{index}" for index in range(22)],
    )
    for tick in range(1, 4):
        recorder.record(_snapshot(tick), diagnostics={})
    manifest = recorder.close()
    assert manifest.completed_ticks == 3

    source = ZarrReplayDataSource(bundle)
    assert source.available_ticks() == (1, 2, 3)
    snapshot = source.load_snapshot(2)
    assert snapshot.position_of(101) == (2, 3)
    details = source.load_ow_details(2, 101)
    assert details is not None
    assert details.values["lineage_id"] == 7
    assert len(details.action_math) == 22
    assert details.action_math[0]["action_name"] == "ACTION_0"
    assert float(details.action_math[0]["last_utilities"]) == -1.0
    assert (bundle / "analysis" / "ow_action_math.parquet").exists()
    assert (bundle / "viewer" / "README.md").exists()
    assert (bundle / "viewer" / "launch_viewer_windows.bat").exists()
    assert source.verify(metadata_only=False)["passed"] is True


def test_replay_bundle_append_resume(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    recorder = ReplayRecorder(
        bundle,
        run_id="resume",
        condition="all_on",
        seed=9303,
        requested_ticks=3,
        recording_tier="analysis_full",
        action_names=[f"ACTION_{index}" for index in range(22)],
    )
    recorder.record(_snapshot(1), diagnostics={})
    recorder.close(state="INTERRUPTED_RESUMABLE", failure="test")
    resumed = ReplayRecorder.resume(bundle)
    resumed.record(_snapshot(2), diagnostics={})
    resumed.record(_snapshot(3), diagnostics={})
    manifest = resumed.close(state="SUCCEEDED")
    assert manifest.completed_ticks == 3
    source = ZarrReplayDataSource(bundle)
    assert source.available_ticks() == (1, 2, 3)
    assert len(source.load_action_math(3, 101)) == 22


def test_viewer_headless_smoke(tmp_path: Path, monkeypatch: object) -> None:
    bundle = tmp_path / "bundle"
    recorder = ReplayRecorder(
        bundle,
        run_id="synthetic",
        condition="all_on",
        seed=9303,
        requested_ticks=2,
        recording_tier="analysis_full",
        action_names=[f"ACTION_{index}" for index in range(22)],
    )
    recorder.record(_snapshot(1), diagnostics={})
    recorder.record(_snapshot(2), diagnostics={})
    recorder.close()

    from owl.viz.replay_app import ReplayApplication

    app = ReplayApplication(bundle, window_size=(900, 600), headless=True)
    assert app.run(max_frames=2) == 0


def test_viewer_exports_outside_read_only_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    recorder = ReplayRecorder(
        bundle,
        run_id="synthetic",
        condition="all_on",
        seed=9303,
        requested_ticks=1,
        recording_tier="analysis_full",
        action_names=[f"ACTION_{index}" for index in range(22)],
    )
    recorder.record(_snapshot(1), diagnostics={})
    recorder.close()
    before = {
        path.relative_to(bundle): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in bundle.rglob("*")
        if path.is_file()
    }
    from owl.viz.replay_app import ReplayApplication

    output = tmp_path / "exports"
    app = ReplayApplication(bundle, window_size=(900, 600), headless=True, output_dir=output)
    assert app.run(max_frames=1) == 0
    after = {
        path.relative_to(bundle): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in bundle.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert output.exists()


def test_standalone_entry_opens_packaged_zip(tmp_path: Path) -> None:
    import zipfile

    from owl.viz.standalone_replay_entry import _safe_extract

    bundle = tmp_path / "bundle"
    recorder = ReplayRecorder(
        bundle,
        run_id="zip",
        condition="all_on",
        seed=9303,
        requested_ticks=1,
        recording_tier="replay_standard",
        action_names=[f"ACTION_{index}" for index in range(22)],
    )
    recorder.record(_snapshot(1), diagnostics={})
    recorder.close()
    archive = tmp_path / "experiment.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in bundle.rglob("*"):
            if path.is_file():
                handle.write(path, Path("run") / "bundle" / path.relative_to(bundle))
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    assert _safe_extract(archive, extracted) == extracted / "run" / "bundle"
