"""Recording, metrics, Zarr fallback, and snapshot tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.core.state import EventRecord, OWRecord
from owl.engine.loop import step
from owl.record.metrics import collect_metrics, save_metrics, summarize_metrics
from owl.record.snapshots import load_snapshot, save_snapshot
from owl.record.zarr_recorder import ZarrRecorder, create_recorder


def make_recording_cfg(height: int = 20, width: int = 20) -> SimulationConfig:
    """Return a small deterministic recording config."""
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = height
    data["world"]["width"] = width
    data["world"]["patch_size"] = 5
    data["world"]["max_steps"] = 20
    data["initialization"]["population_density"] = 0.55
    data["initialization"]["food_patch_count"] = 2
    data["recording"]["enabled"] = True
    data["recording"]["record_every"] = 2
    data["visualization"]["enabled"] = False
    data["debug"]["assert_invariants"] = True
    return SimulationConfig.model_validate(data)


def make_state(seed: int = 123):
    cfg = make_recording_cfg()
    rng = np.random.default_rng(seed)
    state = initialize_world(cfg, rng)
    return cfg, state, rng


def test_collect_metrics_is_scalar_bounded_and_pure() -> None:
    cfg, state, _ = make_state()
    before_health = state.health.copy()
    metrics = collect_metrics(state, cfg)

    expected = {
        "tick",
        "alive_count",
        "alive_fraction",
        "mean_integration",
        "mean_resource",
        "mean_health",
        "global_integration",
        "food_total",
        "signal_total",
        "carnivore_fraction",
        "mean_possibility_entropy",
    }
    assert expected <= set(metrics)
    assert metrics["tick"] == state.tick
    assert 0 <= metrics["alive_count"] <= state.health.size
    assert 0.0 <= metrics["alive_fraction"] <= 1.0
    assert 0.0 <= metrics["mean_health"] <= 1.0
    assert 0.0 <= metrics["mean_possibility_entropy"] <= 1.0
    assert np.array_equal(state.health, before_health)


def test_save_metrics_json_csv_and_summary(tmp_path: Path) -> None:
    cfg, state, rng = make_state()
    rows = [collect_metrics(state, cfg)]
    step(state, cfg, rng)
    rows.append(collect_metrics(state, cfg))

    json_path = tmp_path / "metrics.json"
    csv_path = tmp_path / "metrics.csv"
    parquet_path = tmp_path / "metrics.parquet"

    save_metrics(rows, str(json_path))
    save_metrics(rows, str(csv_path))
    save_metrics(rows, str(parquet_path))

    assert json_path.exists()
    assert csv_path.exists()
    assert parquet_path.exists()
    assert json.loads(json_path.read_text())[0]["tick"] == rows[0]["tick"]
    assert "tick" in csv_path.read_text().splitlines()[0]

    summary = summarize_metrics(rows)
    assert summary["num_records"] == 2
    assert summary["first_tick"] == rows[0]["tick"]
    assert summary["last_tick"] == rows[-1]["tick"]
    assert summary["max_alive"] >= summary["min_alive"]
    assert summary["final_alive"] == rows[-1]["alive_count"]

    empty = summarize_metrics([])
    assert empty["num_records"] == 0


def test_save_metrics_rejects_nonscalar_values_and_bad_extension(tmp_path: Path) -> None:
    try:
        save_metrics(
            [{"tick": 0, "bad": np.zeros((2, 2), dtype=np.float32)}], str(tmp_path / "bad.json")
        )
    except TypeError as exc:
        assert "must be scalar" in str(exc)
    else:
        raise AssertionError("nonscalar metrics should fail")

    try:
        save_metrics([{"tick": 0}], str(tmp_path / "bad.ext"))
    except ValueError as exc:
        assert "unsupported metrics file extension" in str(exc)
    else:
        raise AssertionError("bad metric extension should fail")


def test_zarr_recorder_records_with_numpy_fallback_or_real_zarr(tmp_path: Path) -> None:
    cfg, state, rng = make_state()
    path = tmp_path / "run.zarr"
    recorder = ZarrRecorder(str(path), state, max_steps=5, record_every=1)

    recorder.maybe_record(state)
    step(state, cfg, rng)
    recorder.maybe_record(state)
    recorder.close()

    assert recorder.index == 2
    assert recorder.closed
    assert path.exists()

    # In this runtime zarr may be absent; the fallback writes.npy arrays and
    # metadata. If zarr is installed, the store directory should still exist.
    metadata = path / "metadata.json"
    if metadata.exists():
        meta = json.loads(metadata.read_text())
        assert meta["recorded_count"] == 2
        arr = np.load(path / "state__health.npy")
        assert arr.shape[:3] == (2, cfg.world.height, cfg.world.width)
        ticks = np.load(path / "tick.npy")
        assert ticks.tolist() == [0, 1]


def test_create_recorder_obeys_config_enabled_flag(tmp_path: Path) -> None:
    cfg, state, _ = make_state()
    cfg.recording.zarr_path = str(tmp_path / "enabled.zarr")
    recorder = create_recorder(cfg, state, max_steps=3)
    assert recorder is not None
    recorder.close()

    cfg.recording.enabled = False
    assert create_recorder(cfg, state, max_steps=3) is None


def test_recorder_schedule_and_capacity(tmp_path: Path) -> None:
    cfg, state, rng = make_state()
    recorder = ZarrRecorder(str(tmp_path / "scheduled.zarr"), state, max_steps=2, record_every=2)

    recorder.maybe_record(state)  # tick 0
    step(state, cfg, rng)  # tick 1, not recorded
    recorder.maybe_record(state)
    step(state, cfg, rng)  # tick 2, recorded
    recorder.maybe_record(state)

    assert recorder.index == 2
    recorder.close()

    try:
        ZarrRecorder(str(tmp_path / "bad.zarr"), state, max_steps=1, record_every=0)
    except ValueError as exc:
        assert "record_every must be positive" in str(exc)
    else:
        raise AssertionError("record_every=0 should fail")


def test_snapshot_roundtrip_restores_dense_sparse_and_nested_state(tmp_path: Path) -> None:
    cfg, state, _ = make_state()
    state.tick = 7
    state.event_queue.append(
        EventRecord(kind="collision", tick=7, source=(1, 2), target=(3, 4), payload={"p": 0.5})
    )
    state.mobile_ows[42] = OWRecord(
        id=42,
        type_id=2,
        pos_y=1,
        pos_x=2,
        occupied_cells=[(1, 2), (1, 3)],
        parent_id=None,
        children=[43],
        traits=np.array([0.1, 0.2], dtype=np.float32),
        alive=True,
    )

    snapshot_path = tmp_path / "snapshot.npz"
    save_snapshot(state, str(snapshot_path))
    loaded = load_snapshot(str(snapshot_path))

    assert loaded.tick == state.tick
    assert np.array_equal(loaded.health, state.health)
    assert np.array_equal(loaded.possibility, state.possibility)
    assert np.array_equal(loaded.signal, state.signal)
    assert np.array_equal(loaded.patches.integration, state.patches.integration)
    assert loaded.global_state.integration == state.global_state.integration
    assert loaded.event_queue[0].payload == {"p": 0.5}
    assert 42 in loaded.mobile_ows
    assert np.allclose(loaded.mobile_ows[42].traits, np.array([0.1, 0.2], dtype=np.float32))
    assert loaded.mobile_ows[42].children == [43]


def test_snapshot_rejects_bad_paths(tmp_path: Path) -> None:
    cfg, state, _ = make_state()

    try:
        save_snapshot(state, str(tmp_path))
    except ValueError as exc:
        assert "points to a directory" in str(exc)
    else:
        raise AssertionError("directory snapshot path should fail")

    try:
        load_snapshot(str(tmp_path / "missing.npz"))
    except FileNotFoundError as exc:
        assert "snapshot not found" in str(exc)
    else:
        raise AssertionError("missing snapshot should fail")


def test_zarr_recorder_uses_nested_real_zarr_api_when_available(
    monkeypatch, tmp_path: Path
) -> None:
    """Exercise the installed-zarr path with a minimal fake zarr module.

    This guards the code path that is skipped in minimal runtimes without zarr:
    nested array names must create subgroups and metadata must be written.
    """

    class FakeArray:
        def __init__(self, shape, dtype):
            self.data = np.zeros(shape, dtype=np.dtype(dtype))
            self.shape = self.data.shape

        def __setitem__(self, key, value):
            self.data[key] = value

        def __getitem__(self, key):
            return self.data[key]

    class FakeGroup:
        def __init__(self, path: Path):
            self.path = path
            self.path.mkdir(parents=True, exist_ok=True)
            self.attrs = {}
            self.groups = {}
            self.arrays = {}
            self.store = self

        def require_group(self, name):
            if name not in self.groups:
                self.groups[name] = FakeGroup(self.path / name)
            return self.groups[name]

        def create_array(self, name, shape, chunks, dtype):
            del chunks
            arr = FakeArray(shape, dtype)
            self.arrays[name] = arr
            return arr

        def close(self):
            return None

    roots = {}

    class FakeZarr:
        @staticmethod
        def open_group(path, mode="r"):
            del mode
            root = FakeGroup(Path(path))
            roots[str(path)] = root
            return root

    monkeypatch.setitem(sys.modules, "zarr", FakeZarr)

    cfg, state, _rng = make_state()
    recorder = ZarrRecorder(str(tmp_path / "fake_real.zarr"), state, max_steps=2, record_every=1)
    recorder.maybe_record(state)
    recorder.close()

    root = roots[str(tmp_path / "fake_real.zarr")]
    assert root.attrs["recorded_count"] == 1
    assert "state" in root.groups
    assert "integration" in root.groups["state"].arrays
    assert (tmp_path / "fake_real.zarr" / "metrics.json").exists()
