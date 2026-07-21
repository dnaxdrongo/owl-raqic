from __future__ import annotations

from pathlib import Path

from owl.experiments.controller import _package, _status
from owl.experiments.progress import atomic_write_json
from owl.viz.replay_timeline import PLAYBACK_SPEEDS, PlaybackClock


def test_status_is_read_only(tmp_path: Path, capsys: object) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    atomic_write_json(run_root / "run_progress.json", {"state": "PLANNED", "phase": "none"})
    before = {path.relative_to(run_root): path.stat().st_mtime_ns for path in run_root.rglob("*")}
    assert _status(run_root) == 0
    after = {path.relative_to(run_root): path.stat().st_mtime_ns for path in run_root.rglob("*")}
    assert before == after


def test_failure_package_is_named_as_evidence(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    atomic_write_json(
        run_root / "run_progress.json",
        {"state": "FAILED_PARTIAL", "phase": "simulate", "error": "example"},
    )
    (run_root / "error.log").write_text("failed\n", encoding="utf-8")
    assert _package(run_root) == 0
    files = list((run_root / "packages").glob("*.zip"))
    assert len(files) == 1
    assert "FAILED_PARTIAL_evidence" in files[0].name


def test_playback_clock_supports_requested_speeds_and_reverse() -> None:
    assert 1.2 in PLAYBACK_SPEEDS
    clock = PlaybackClock(ticks=tuple(range(1, 21)), tick_rate=10.0)
    clock.set_speed(1.2)
    clock.playing = True
    assert clock.update(0.1) == 2
    clock.direction = -1
    assert clock.update(0.1) == 1
    clock.step(10)
    assert clock.current_tick == 11


def test_playback_clock_exposes_read_only_subtick_animation_progress() -> None:
    clock = PlaybackClock(ticks=(1, 2, 3), tick_rate=10.0)
    assert clock.interpolation_progress() == 1.0
    clock.playing = True
    assert clock.update(0.025) == 1
    assert clock.interpolation_progress() == 0.25
    assert clock.update(0.075) == 2
    assert clock.interpolation_progress() == 0.0


def test_registered_manifest_requires_exactly_one_full_replay(tmp_path: Path) -> None:
    import yaml

    from owl.experiments.manifest import ExperimentManifest

    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "owl.experiment.v1",
                "name": "bad",
                "ticks": 1,
                "seed": 1,
                "conditions": [
                    {"name": "a", "config": "a.yaml", "full_replay": False},
                    {"name": "b", "config": "b.yaml", "full_replay": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="exactly one"):
        ExperimentManifest.load(path)


def test_registered_manifest_bounds_condition_concurrency(tmp_path: Path) -> None:
    import yaml

    from owl.experiments.manifest import ExperimentManifest

    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "owl.experiment.v1",
                "name": "bad-concurrency",
                "ticks": 1,
                "seed": 1,
                "max_concurrent_conditions": 3,
                "conditions": [
                    {"name": "a", "config": "a.yaml", "full_replay": True},
                    {"name": "b", "config": "b.yaml", "full_replay": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="max_concurrent_conditions"):
        ExperimentManifest.load(path)


def test_registered_manifest_supports_unique_matched_seed_blocks(tmp_path: Path) -> None:
    import yaml

    from owl.experiments.manifest import ExperimentManifest

    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "owl.experiment.v1",
                "name": "matched",
                "ticks": 300,
                "seeds": [43001, 43002, 43003],
                "max_concurrent_conditions": 4,
                "conditions": [
                    {"name": "a", "config": "a.yaml", "full_replay": True},
                    {"name": "b", "config": "b.yaml", "full_replay": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = ExperimentManifest.load(path)
    assert manifest.seeds == (43001, 43002, 43003)
    assert manifest.seed == 43001
    assert manifest.max_concurrent_conditions == 4


def test_phase4_budget_manifest_assigns_full_replay_only_to_c3() -> None:
    from owl.experiments.manifest import ExperimentManifest

    root = Path(__file__).resolve().parents[1]
    manifest = ExperimentManifest.load(
        root / "experiments" / "phase4_h100_budget_development.yaml"
    )
    full = [condition.name for condition in manifest.conditions if condition.full_replay]
    assert full == ["phase_interference"]
    assert manifest.recording_tier == "analysis_full"


def test_duplicate_run_lock_is_rejected(tmp_path: Path) -> None:
    import pytest

    from owl.experiments.process_control import RunLock

    lock_path = tmp_path / ".experiment.lock"
    with RunLock(lock_path), pytest.raises(RuntimeError, match="already exists"):
        RunLock(lock_path).acquire()


def test_success_package_requires_verified_bundle(tmp_path: Path) -> None:
    import numpy as np

    from owl.record.replay_recorder import ReplayRecorder
    from owl.viz.visual_snapshot import snapshot_from_arrays

    run_root = tmp_path / "run"
    run_root.mkdir()
    bundle = run_root / "bundle"
    recorder = ReplayRecorder(
        bundle,
        run_id="package",
        condition="all_on",
        seed=1,
        requested_ticks=1,
        recording_tier="replay_standard",
        action_names=[f"ACTION_{index}" for index in range(22)],
    )
    recorder.record(
        snapshot_from_arrays(
            tick=1,
            boundary_mode="toroidal",
            arrays={
                "health": np.ones((2, 2), dtype=np.float32),
                "resource": np.ones((2, 2), dtype=np.float32),
                "occupancy": np.arange(4, dtype=np.int64).reshape(2, 2),
                "readout": np.zeros((2, 2), dtype=np.int16),
            },
        ),
        diagnostics={},
    )
    recorder.close()
    atomic_write_json(run_root / "run_progress.json", {"state": "SUCCEEDED", "phase": "done"})
    assert _package(run_root) == 0
    packages = list((run_root / "packages").glob("*_results.zip"))
    assert len(packages) == 1
