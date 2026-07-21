from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from owl.record.action_math_batch import NumPyReplayBatchBuilder, build_living_index
from owl.record.parquet_sink import PartitionedParquetSink
from owl.record.replay_recorder import ReplayRecorder
from owl.record.replay_schema import compile_replay_schema
from owl.viz.visual_snapshot import snapshot_from_arrays


def _arrays() -> dict[str, np.ndarray]:
    h, w, actions = 2, 4, 3
    health = np.zeros((h, w), dtype=np.float32)
    occupancy = np.full((h, w), -1, dtype=np.int64)
    health.reshape(-1)[[1, 6]] = 1.0
    occupancy.reshape(-1)[[1, 6]] = [101, 202]
    readout = np.full((h, w), -1, dtype=np.int16)
    readout.reshape(-1)[[1, 6]] = [2, 1]

    world_action = np.arange(h * w * actions, dtype=np.float64).reshape(h, w, actions)
    authority = np.ones((h, w, actions), dtype=bool)
    authority[0, 1, 1] = False
    patch = np.asarray(
        [
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ]
        ],
        dtype=np.float64,
    )
    global_action = np.asarray([30.0, 31.0, 32.0], dtype=np.float64)
    scalar = np.arange(h * w, dtype=np.float32).reshape(h, w)
    return {
        "health": health,
        "occupancy": occupancy,
        "resource": scalar,
        "readout": readout,
        "raqic_readout": readout,
        "lineage_id": np.where(health > 0, 7, -1).astype(np.int64),
        "parent_id": np.where(health > 0, 55, -1).astype(np.int64),
        "raqic_probabilities": world_action,
        "last_utilities": world_action.astype(np.float32),
        "pre_utilities": (world_action + 100).astype(np.float32),
        "raqic_score": world_action + 200,
        "raqic_phase": world_action + 300,
        "raqic_parent_action_phase": world_action + 400,
        "raqic_parent_action_coherence": world_action + 500,
        "raqic_patch_action_phase": patch,
        "raqic_patch_action_coherence": patch + 100,
        "raqic_global_action_phase": global_action,
        "raqic_global_action_coherence": global_action + 100,
        "_authority_bool": authority,
        "raqic_policy_kl": scalar.astype(np.float64),
    }


def _compiled_and_builder(
    *, max_batch_rows: int = 6
) -> tuple[Any, NumPyReplayBatchBuilder, dict[str, np.ndarray]]:
    arrays = _arrays()
    names = ("REST", "MOVE", "FEED")
    compiled = compile_replay_schema(
        arrays,
        world_shape=(2, 4),
        action_names=names,
        recording_tier="analysis_full",
    )
    builder = NumPyReplayBatchBuilder(
        compiled,
        condition="all_on",
        seed=9303,
        action_names=names,
        max_batch_rows=max_batch_rows,
        max_batch_bytes=4 * 1024 * 1024,
        full_validation=True,
    )
    return compiled, builder, arrays


def _concat_action_table(
    builder: NumPyReplayBatchBuilder, arrays: dict[str, np.ndarray]
) -> pa.Table:
    living = build_living_index(arrays, world_shape=(2, 4))
    batches = [
        item.to_record_batch(full_validation=True)
        for item in builder.iter_action_math_batches(arrays, tick=9, living=living)
    ]
    return pa.Table.from_batches(batches)


def test_vectorized_action_math_preserves_order_dtype_and_context_projection() -> None:
    compiled, builder, arrays = _compiled_and_builder(max_batch_rows=3)
    table = _concat_action_table(builder, arrays)

    assert table.num_rows == 6
    assert table.column("ow_id").to_pylist() == [101, 101, 101, 202, 202, 202]
    assert table.column("action_index").to_pylist() == [0, 1, 2, 0, 1, 2]
    assert table.column("selected").to_pylist() == [False, False, True, False, True, False]
    assert table.column("legal").to_pylist() == [True, False, True, True, True, True]

    expected_world = arrays["raqic_probabilities"].reshape(8, 3)[[1, 6], :].reshape(-1)
    np.testing.assert_array_equal(table.column("raqic_probabilities").to_numpy(), expected_world)
    assert table.schema.field("raqic_probabilities").type == pa.float64()
    assert table.schema.field("last_utilities").type == pa.float32()

    assert table.column("raqic_patch_action_phase").to_pylist() == [
        10.0,
        11.0,
        12.0,
        20.0,
        21.0,
        22.0,
    ]
    assert table.column("raqic_global_action_phase").to_pylist() == [
        30.0,
        31.0,
        32.0,
        30.0,
        31.0,
        32.0,
    ]
    expected_parent = arrays["raqic_parent_action_phase"].reshape(8, 3)[[1, 6], :].reshape(-1)
    np.testing.assert_array_equal(
        table.column("raqic_parent_action_phase").to_numpy(), expected_parent
    )
    assert compiled.patch_shape == (1, 2)


def test_state_decision_and_action_builders_share_canonical_living_order() -> None:
    _compiled, builder, arrays = _compiled_and_builder(max_batch_rows=3)
    living = build_living_index(arrays, world_shape=(2, 4))
    state = pa.Table.from_batches(
        [
            item.to_record_batch(full_validation=True)
            for item in builder.iter_state_batches(arrays, tick=2, living=living)
        ]
    )
    decision = pa.Table.from_batches(
        [
            item.to_record_batch(full_validation=True)
            for item in builder.iter_decision_batches(arrays, tick=2, living=living)
        ]
    )
    action = _concat_action_table(builder, arrays)

    assert state.column("ow_id").to_pylist() == [101, 202]
    assert decision.column("ow_id").to_pylist() == [101, 202]
    assert action.column("ow_id").to_pylist()[::3] == [101, 202]
    assert decision.column("selected_action").to_pylist() == [2, 1]
    expected_selected = [
        float(arrays["raqic_score"][0, 1, 2]),
        float(arrays["raqic_score"][1, 2, 1]),
    ]
    assert decision.column("selected_raqic_score").to_pylist() == expected_selected


def test_partitioned_sinks_have_independent_part_counters_and_resume(tmp_path: Path) -> None:
    compiled, builder, arrays = _compiled_and_builder(max_batch_rows=3)
    living = build_living_index(arrays, world_shape=(2, 4))
    state_batch = next(builder.iter_state_batches(arrays, tick=1, living=living)).to_record_batch()
    action_batches = [
        item.to_record_batch()
        for item in builder.iter_action_math_batches(arrays, tick=1, living=living)
    ]

    state_root = tmp_path / "ow_state.parquet"
    action_root = tmp_path / "ow_action_math.parquet"
    state_sink = PartitionedParquetSink(
        state_root,
        compiled.state_schema,
        table_name="ow_state",
        schema_digest=compiled.schema_digest,
    )
    action_sink = PartitionedParquetSink(
        action_root,
        compiled.action_math_schema,
        table_name="ow_action_math",
        schema_digest=compiled.schema_digest,
    )
    state_sink.write_batch(state_batch, tick=1)
    for batch in action_batches:
        action_sink.write_batch(batch, tick=1)
    state_sink.close()
    action_sink.close()

    assert [path.name for path in state_root.glob("part-*.parquet")] == ["part-000000.parquet"]
    assert sorted(path.name for path in action_root.glob("part-*.parquet")) == [
        "part-000000.parquet",
        "part-000001.parquet",
    ]

    resumed = PartitionedParquetSink(
        action_root,
        compiled.action_math_schema,
        table_name="ow_action_math",
        schema_digest=compiled.schema_digest,
        resume=True,
        max_committed_tick=1,
    )
    resumed.write_batch(action_batches[0], tick=2)
    resumed.close()
    assert (action_root / "part-000002.parquet").exists()
    assert pq.read_table(action_root).num_rows == 9


def test_recorder_writes_primitive_action_columns_without_json_amplification(
    tmp_path: Path,
) -> None:
    arrays = _arrays()
    snapshot = snapshot_from_arrays(
        tick=1,
        boundary_mode="toroidal",
        arrays=arrays,
        events=(),
        metadata={"source": "v099-test"},
    )
    root = tmp_path / "bundle"
    recorder = ReplayRecorder(
        root,
        run_id="v099",
        condition="all_on",
        seed=9303,
        requested_ticks=1,
        recording_tier="analysis_full",
        action_names=("REST", "MOVE", "FEED"),
        max_batch_rows=3,
    )
    recorder.record(snapshot, diagnostics={})
    manifest = recorder.close()
    assert manifest.completed_ticks == 1

    table = pq.read_table(root / "analysis" / "ow_action_math.parquet")
    assert table.num_rows == 6
    for name in (
        "raqic_patch_action_phase",
        "raqic_global_action_phase",
        "raqic_parent_action_phase",
    ):
        assert pa.types.is_floating(table.schema.field(name).type)
    assert table.column("raqic_patch_action_phase").to_pylist() == [
        10.0,
        11.0,
        12.0,
        20.0,
        21.0,
        22.0,
    ]


def test_large_table_path_contains_no_row_dictionary_or_from_pylist_builder() -> None:
    repo = Path(__file__).resolve().parents[1]
    sources = [
        repo / "src/owl/record/action_math_batch.py",
        repo / "src/owl/record/gpu_replay_staging.py",
        repo / "src/owl/record/replay_recorder.py",
        repo / "src/owl/record/parquet_sink.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "Table.from_pylist" not in combined
    assert "_action_math_rows.append" not in combined

    tree = ast.parse((repo / "src/owl/record/action_math_batch.py").read_text(encoding="utf-8"))
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "iter_action_math_batches"
    )
    for loop in (node for node in ast.walk(target) if isinstance(node, ast.For)):
        # Batch and field-plan loops are allowed; scalar cell/action loops are not.
        rendered = ast.unparse(loop.target)
        assert rendered not in {"y", "x", "action", "action_index", "cell"}


def test_invalid_patch_shape_fails_before_first_tick() -> None:
    arrays = _arrays()
    arrays["raqic_patch_action_phase"] = np.zeros((3, 2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="must divide world shape"):
        compile_replay_schema(
            arrays,
            world_shape=(2, 4),
            action_names=("REST", "MOVE", "FEED"),
            recording_tier="analysis_full",
        )


def test_cupy_builder_is_optional_without_importing_cupy() -> None:
    from owl.record.gpu_replay_staging import cupy_available

    assert isinstance(cupy_available(), bool)


def test_vectorized_event_candidate_filter_preserves_cell_and_type_order() -> None:
    from types import SimpleNamespace

    from owl.core.actions import Action
    from owl.viz.event_bus import VisualEventType, events_from_state

    health = np.ones((2, 3), dtype=np.float32)
    readout = np.asarray(
        [
            [int(Action.REST), int(Action.FEED), int(Action.COMMUNICATE)],
            [int(Action.REST), int(Action.MOVE_E), int(Action.REST)],
        ],
        dtype=np.int16,
    )
    probabilities = np.zeros((2, 3, 22), dtype=np.float32)
    probabilities[..., 0] = 1.0
    probabilities[0, 1, :] = 1.0 / 22.0  # action event followed by uncertainty.
    signal = np.zeros((2, 3, 2), dtype=np.float32)
    signal[0, 2, 1] = 1.0
    state = SimpleNamespace(
        tick=5,
        health=health,
        readout=readout,
        raqic_readout=readout,
        raqic_probabilities=probabilities,
        occupancy=np.arange(6, dtype=np.int64).reshape(2, 3),
        signal_emission=signal,
    )
    events = events_from_state(state, entropy_threshold=1.5).events
    assert [(item.y, item.x, item.event_type) for item in events] == [
        (0, 2, VisualEventType.COMMUNICATE),
        (0, 1, VisualEventType.FEED),
        (1, 1, VisualEventType.MOVE),
        (0, 1, VisualEventType.RAQIC_UNCERTAINTY),
    ]
    communication = next(item for item in events if item.event_type == VisualEventType.COMMUNICATE)
    assert communication.channel == 1


def test_tick_transaction_rolls_back_zarr_and_all_table_parts_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = _arrays()
    snapshot = snapshot_from_arrays(
        tick=1,
        boundary_mode="toroidal",
        arrays=arrays,
        events=(),
        metadata={},
    )
    root = tmp_path / "bundle"
    recorder = ReplayRecorder(
        root,
        run_id="rollback",
        condition="all_on",
        seed=1,
        requested_ticks=1,
        recording_tier="analysis_full",
        action_names=("REST", "MOVE", "FEED"),
        max_batch_rows=3,
    )
    recorder._initialize(snapshot)
    recorder._initialize_columnar(snapshot)
    sink = recorder._sinks["ow_action_math"]
    original = sink.write_batch
    calls = 0

    def fail_after_first(batch: Any, *, tick: int) -> Any:
        nonlocal calls
        calls += 1
        receipt = original(batch, tick=tick)
        if calls == 1:
            raise RuntimeError("injected action sink failure")
        return receipt

    monkeypatch.setattr(sink, "write_batch", fail_after_first)
    with pytest.raises(RuntimeError, match="injected action sink failure"):
        recorder.record(snapshot, diagnostics={})

    assert recorder._arrays["tick"].shape[0] == 0
    assert not list((root / "replay" / "commits").glob("tick_*.json"))
    for table_root in (
        root / "analysis" / "ow_state.parquet",
        root / "analysis" / "ow_decisions.parquet",
        root / "analysis" / "ow_action_math.parquet",
    ):
        assert not list(table_root.glob("part-*.parquet"))

    monkeypatch.setattr(sink, "write_batch", original)
    recorder.record(snapshot, diagnostics={})
    assert recorder.close().completed_ticks == 1


def test_snapshot_capture_includes_dedicated_patch_and_global_device_maps() -> None:
    from types import SimpleNamespace

    from owl.viz.visual_snapshot import snapshot_from_device_state

    class Backend:
        name = "numpy"

        @staticmethod
        def asnumpy(value: Any) -> np.ndarray:
            return np.asarray(value)

    ds = SimpleNamespace(
        arrays={
            "health": np.ones((2, 2), dtype=np.float32),
            "occupancy": np.arange(4, dtype=np.int64).reshape(2, 2),
        },
        patch_arrays={"raqic_patch_action_phase": np.arange(6, dtype=np.float64).reshape(1, 2, 3)},
        global_arrays={"raqic_global_action_phase": np.arange(3, dtype=np.float64)},
        backend=Backend(),
        xp=np,
        is_gpu=False,
        tick=4,
        metadata={},
    )
    snapshot = snapshot_from_device_state(
        ds,
        field_names=(
            "health",
            "occupancy",
            "raqic_patch_action_phase",
            "raqic_global_action_phase",
        ),
    )
    assert snapshot.arrays["raqic_patch_action_phase"].shape == (1, 2, 3)
    assert snapshot.arrays["raqic_global_action_phase"].tolist() == [0.0, 1.0, 2.0]
    assert not snapshot.arrays["raqic_patch_action_phase"].flags.writeable


def _record_bundle(root: Path, *, materialization_mode: str) -> None:
    arrays = _arrays()
    snapshot = snapshot_from_arrays(
        tick=1,
        boundary_mode="toroidal",
        arrays=arrays,
        events=(),
        metadata={"source": "v099-deferred-test"},
    )
    recorder = ReplayRecorder(
        root,
        run_id=f"v099-{materialization_mode}",
        condition="all_on",
        seed=9303,
        requested_ticks=1,
        recording_tier="analysis_full",
        action_names=("REST", "MOVE", "FEED"),
        max_batch_rows=3,
        materialization_mode=materialization_mode,
    )
    recorder.record(snapshot, diagnostics={})
    recorder.close()


def test_v099_deferred_equals_inline_and_exports_stream(tmp_path: Path) -> None:
    from owl.record.action_math_materializer import materialize_action_math
    from owl.replay.zarr_source import ZarrReplayDataSource

    inline = tmp_path / "inline"
    deferred = tmp_path / "deferred"
    _record_bundle(inline, materialization_mode="inline")
    _record_bundle(deferred, materialization_mode="deferred")

    pending_status = __import__("json").loads(
        (deferred / "run_status.json").read_text(encoding="utf-8")
    )
    assert pending_status["state"] == "SCIENTIFIC_ARRAYS_COMPLETE_MATERIALIZATION_PENDING"
    assert not (deferred / "analysis" / "ow_action_math.parquet").exists()
    pending_source = ZarrReplayDataSource(deferred, verify_metadata=True)
    with pytest.raises(RuntimeError, match="materialization is incomplete"):
        pending_source.load_action_math(1, 101)
    with pytest.raises(RuntimeError, match="materialization is incomplete"):
        pending_source.export_action_math_csv(
            str(tmp_path / "pending.csv"), ow_id=101, start_tick=1, end_tick=1
        )

    result = materialize_action_math(deferred, max_batch_rows=3)
    assert result["state"] == "SUCCEEDED"

    inline_table = pq.read_table(inline / "analysis" / "ow_action_math.parquet")
    deferred_table = pq.read_table(deferred / "analysis" / "ow_action_math.parquet")
    assert inline_table.schema.equals(deferred_table.schema, check_metadata=True)
    assert inline_table.equals(deferred_table)

    source = ZarrReplayDataSource(deferred, verify_metadata=True)
    action_csv = tmp_path / "action.csv"
    selection_csv = tmp_path / "selection.csv"
    source.export_action_math_csv(str(action_csv), ow_id=101, start_tick=1, end_tick=1)
    source.export_selection_csv(str(selection_csv), ow_id=101, start_tick=1, end_tick=1)
    assert len(action_csv.read_text(encoding="utf-8").splitlines()) == 4
    assert len(selection_csv.read_text(encoding="utf-8").splitlines()) == 2
    assert source.verify(metadata_only=False)["passed"] is True


def test_v099_deferred_materialization_resume_is_idempotent(tmp_path: Path) -> None:
    from owl.record.action_math_materializer import materialize_action_math

    deferred = tmp_path / "deferred"
    _record_bundle(deferred, materialization_mode="deferred")
    first = materialize_action_math(deferred, max_batch_rows=3)
    second = materialize_action_math(deferred, max_batch_rows=3)
    assert first["rows_written"] == 6
    assert second["already_complete"] is True
    assert pq.read_table(deferred / "analysis" / "ow_action_math.parquet").num_rows == 6


def test_v099_batch_boundaries_and_adaptive_policy_preserve_table_semantics() -> None:
    from owl.record.replay_telemetry import AdaptiveBatchPolicy

    compiled, small_builder, arrays = _compiled_and_builder(max_batch_rows=3)
    large_builder = NumPyReplayBatchBuilder(
        compiled,
        condition="all_on",
        seed=9303,
        action_names=("REST", "MOVE", "FEED"),
        max_batch_rows=6,
        max_batch_bytes=4 * 1024 * 1024,
    )
    living = build_living_index(arrays, world_shape=(2, 4))
    small = pa.Table.from_batches(
        [
            item.to_record_batch()
            for item in small_builder.iter_action_math_batches(arrays, tick=4, living=living)
        ]
    )
    large = pa.Table.from_batches(
        [
            item.to_record_batch()
            for item in large_builder.iter_action_math_batches(arrays, tick=4, living=living)
        ]
    )
    assert small.schema.equals(large.schema, check_metadata=True)
    assert small.equals(large)

    policy = AdaptiveBatchPolicy(
        action_count=3,
        initial_rows=6,
        min_rows=3,
        max_rows=30,
        target_batch_bytes=3_000,
    )
    next_rows = policy.observe(rows=6, arrow_bytes=600, elapsed_seconds=0.1)
    assert 3 <= next_rows <= 30
    assert next_rows % 3 == 0
    assert policy.telemetry.observations == 1


def test_v099_package_fails_closed_while_materialization_is_pending(tmp_path: Path) -> None:
    import json

    from owl.experiments.controller import _package

    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run_progress.json").write_text(
        json.dumps(
            {
                "state": "SCIENTIFIC_ARRAYS_COMPLETE_MATERIALIZATION_PENDING",
                "phase": "materialization_pending",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="pending action-table materialization"):
        _package(run_root)


def test_gpu_replay_staging_world_arrays_take_precedence_over_patch_collisions() -> None:
    from types import SimpleNamespace

    import numpy as np

    from owl.record.gpu_replay_staging import _authoritative_device_array_map

    world_health = np.ones((200, 200), dtype=np.float32)
    patch_health = np.zeros((40, 40), dtype=np.float32)

    ds = SimpleNamespace(
        arrays={
            "health": world_health,
            "occupancy": np.zeros((200, 200), dtype=np.int64),
            "selected_action": np.zeros((200, 200), dtype=np.int16),
        },
        patch_arrays={
            "health": patch_health,
            "raqic_patch_action_phase": np.zeros((40, 40), dtype=np.float32),
        },
        global_arrays={
            "health": np.zeros((1,), dtype=np.float32),
            "raqic_global_action_phase": np.zeros((22,), dtype=np.float32),
        },
    )

    arrays = _authoritative_device_array_map(ds)

    assert arrays["health"] is world_health
    assert arrays["health"] is not patch_health
    assert arrays["raqic_patch_action_phase"].shape == (40, 40)
    assert arrays["raqic_global_action_phase"].shape == (22,)
