import json
from pathlib import Path

from owl.core.config import load_config
from owl.gpu.numerical_ledger import NumericalLedger
from owl.gpu.run_manifest import build_run_manifest


def test_run_manifest_has_hashes(tmp_path):
    cfg_path = Path("configs/gpu_v07_persistent_small.yaml")
    cfg = load_config(cfg_path)
    manifest = build_run_manifest(
        ".",
        cfg_path,
        seed=cfg.world.seed,
        precision=cfg.raqic.full_gpu_precision,
        run_class=cfg.raqic.full_gpu_run_class,
        all_cell_semantics=True,
        fallback_count=0,
    )
    path = manifest.write(tmp_path / "manifest.json")
    payload = json.loads(path.read_text())
    assert len(payload["repo_sha256"]) == 64
    assert len(payload["config_sha256"]) == 64


def test_numerical_ledger_rejects_probability_error():
    cfg = load_config("configs/gpu_v07_persistent_small.yaml")
    ledger = NumericalLedger.from_config(cfg)
    ledger.update_metrics({"raqic_max_row_error": 1.0})
    assert not ledger.validate()["passed"]
