from __future__ import annotations

import ast
from pathlib import Path

import pyarrow.parquet as pq

from owl.core.config import load_config
from owl.gpu.run_context import PersistentOWLDeviceRun
from owl.record.cadc_writer import CADCFactualRecorder
from owl.record.gpu_replay_staging import collect_cadc_host_packet


def test_cadc_packet_is_bounded_immutable_and_writer_batches_rows(tmp_path: Path) -> None:
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    cfg.debug.assert_invariants = False
    cfg.recording.cadc.enabled = True
    cfg.recording.cadc.max_batch_rows = 22
    run = PersistentOWLDeviceRun.from_config(
        cfg, force_backend="numpy", output_root=tmp_path / "scientific"
    )
    try:
        run.step()
        packet = collect_cadc_host_packet(run.ds)
        assert packet.transfer_bytes == run.cadc_buffer.nbytes
        assert packet.transfer_bytes <= cfg.recording.cadc.max_pending_bytes
        assert packet.transfer_count == 0
        assert all(not value.flags.writeable for value in packet.arrays.values())

        writer = CADCFactualRecorder(
            tmp_path / "bundle",
            packet,
            run_id="bounded",
            condition="all_on",
            seed=int(cfg.world.seed),
            config=cfg.recording.cadc.model_dump(mode="json"),
            compression="zstd",
            row_group_rows=22,
        )
        receipt = writer.record(packet)
        writer.close()
        assert receipt.row_counts["candidates"] == receipt.row_counts["decisions"] * 22
        parts = sorted(
            (tmp_path / "bundle" / "analysis" / "cadc_v1" / "candidates.parquet").glob(
                "part-*.parquet"
            )
        )
        assert len(parts) > 1
        assert max(pq.ParquetFile(path).metadata.num_rows for path in parts) <= 22
    finally:
        run.close(checkpoint=False)


def test_cadc_transfer_function_has_one_packet_sync_and_no_row_conversion() -> None:
    source = Path("src/owl/record/gpu_replay_staging.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "collect_cadc_host_packet"
    )
    rendered = ast.unparse(function)
    assert ".item(" not in rendered
    assert "asnumpy" not in rendered
    assert rendered.count("ready.synchronize()") == 1
