import json

from owl.record.gpu_async_writer import AsyncGPUWriter


def test_async_gpu_writer(tmp_path):
    path = tmp_path / "metrics.jsonl"
    w = AsyncGPUWriter(path).start()
    w.write({"tick": 1, "x": 2})
    w.close()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["tick"] == 1
