from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts/run_cadc_phase4_corpus.py"
    spec = importlib.util.spec_from_file_location("phase4_corpus_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inventory(root: Path, *, unit_id: str, passed: bool) -> None:
    root.mkdir(parents=True)
    payload: dict[str, object] = {"unit_id": unit_id, "passed": passed}
    if passed:
        factual = root / "factual_bundle"
        counterfactual = root / "counterfactual"
        factual.mkdir()
        counterfactual.mkdir()
        (factual / "manifest.json").write_text("{}\n", encoding="utf-8")
        (counterfactual / "manifest.json").write_text("{}\n", encoding="utf-8")
        payload.update(
            {
                "factual_root": str(factual),
                "counterfactual_root": str(counterfactual),
                "source_decisions": 1,
                "branch_horizons": 1,
                "candidate_pairs": 1,
            }
        )
    (root / "corpus_unit_inventory.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )


def test_failed_unit_is_quarantined_without_deletion(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "runs" / "unit-a"
    _write_inventory(output, unit_id="unit-a", passed=False)
    partial = output / "factual_bundle" / "partial.parquet"
    partial.parent.mkdir()
    partial.write_bytes(b"partial evidence")

    receipt = module._quarantine_failed_output(output, "unit-a")

    assert receipt is not None
    archived = Path(receipt["path"])
    assert not output.exists()
    assert (
        archived / "factual_bundle" / "partial.parquet"
    ).read_bytes() == b"partial evidence"
    assert receipt["inventory_sha256"]


def test_retry_attempt_paths_are_monotonic(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "runs" / "unit-a"
    _write_inventory(output, unit_id="unit-a", passed=False)
    first = module._quarantine_failed_output(output, "unit-a")
    _write_inventory(output, unit_id="unit-a", passed=False)
    second = module._quarantine_failed_output(output, "unit-a")

    assert first is not None and first["path"].endswith("attempt-0001")
    assert second is not None and second["path"].endswith("attempt-0002")


def test_successful_unit_is_never_quarantined(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "runs" / "unit-a"
    _write_inventory(output, unit_id="unit-a", passed=True)

    with pytest.raises(RuntimeError, match="refusing to quarantine a successful"):
        module._quarantine_failed_output(output, "unit-a")

    assert output.is_dir()


def test_success_inventory_identity_must_match(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "runs" / "unit-a"
    _write_inventory(output, unit_id="unit-b", passed=True)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        module._validated_success(output, "unit-a")


def test_success_inventory_requires_nonempty_registered_roots(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "runs" / "unit-a"
    _write_inventory(output, unit_id="unit-a", passed=True)
    (output / "counterfactual" / "manifest.json").unlink()

    with pytest.raises(RuntimeError, match="empty counterfactual_root"):
        module._validated_success(output, "unit-a")


def test_parallel_unit_command_freezes_transfer_mode(tmp_path: Path) -> None:
    module = _module()
    args = SimpleNamespace(
        engine_root=tmp_path / "engine",
        phase25_certificate=tmp_path / "phase25.json",
        hardening_receipt=tmp_path / "hardening.json",
        backend="cupy",
        branch_transfer_mode="deferred_bounded",
        aggregate_device_budget_bytes=80_000,
        max_concurrent_units=4,
    )
    unit = {
        "unit_id": "unit-a",
        "derived_config_path": str(tmp_path / "unit.yaml"),
        "context_family": "balanced",
        "source_tick": 2,
    }
    command = module._unit_command(args, tmp_path / "helper.py", unit, tmp_path / "out")
    assert command[-4:] == [
        "--branch-transfer-mode",
        "deferred_bounded",
        "--worker-device-budget-bytes",
        "20000",
    ]


def test_worker_environment_prevents_cpu_thread_oversubscription() -> None:
    module = _module()
    environment = module._worker_environment()
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
