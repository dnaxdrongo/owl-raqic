"""Repository-integrity, every-agent, and report contract tests."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

import owl_raqic
from owl.core.config import load_config
from owl_raqic.qiskit_backend.per_ow_executor import validate_processed_ow_ids
from owl_raqic.reports.audit_report import write_audit_json
from owl_raqic.reports.benchmark_report import write_benchmark_csv
from owl_raqic.reports.markdown import write_markdown_report

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_VERSION = "0.9.5.1"


def _active_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def test_active_version_declarations_are_consistent() -> None:
    version = _active_version()
    assert owl_raqic.__version__ == version
    assert version == "0.9.9"
    assert f"v{version}" in (ROOT / "V099_RECORDING_ACCELERATION_PACKAGE_README.md").read_text(
        encoding="utf-8"
    )
    assert f"v{version}" in (
        ROOT / "OWL_RAQIC_V099_GPU_COLUMNAR_REPLAY_IMPLEMENTATION_REPORT.md"
    ).read_text(encoding="utf-8")


def test_v0951_release_metadata_remains_historical() -> None:
    release_candidate = json.loads((ROOT / "RELEASE_CANDIDATE.json").read_text(encoding="utf-8"))
    release_manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert release_candidate["version"] == HISTORICAL_VERSION
    assert release_manifest["release"] == f"v{HISTORICAL_VERSION}"
    assert f"v{HISTORICAL_VERSION}" in (ROOT / "README_V0_9_5_1_RELEASE_INTEGRITY.md").read_text(
        encoding="utf-8"
    )


def test_report_writers_preserve_supported_values_and_order(tmp_path: Path) -> None:
    audit = write_audit_json(
        tmp_path / "nested" / "audit.json",
        {
            "array": np.asarray([1, 2, 3], dtype=np.int64),
            "complex": 2.0 + 3.0j,
        },
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["array"] == [1, 2, 3]
    assert payload["complex"] == {"real": 2.0, "imag": 3.0}

    benchmark = write_benchmark_csv(
        tmp_path / "benchmark.csv",
        ({"name": name, "value": value} for name, value in (("a", 1), ("b", 2))),
    )
    with benchmark.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"name": "a", "value": "1"},
        {"name": "b", "value": "2"},
    ]

    markdown = write_markdown_report(
        tmp_path / "report.md",
        "Title",
        {"First": "One", "Second": "Two"},
    )
    text = markdown.read_text(encoding="utf-8")
    assert text.index("## First") < text.index("## Second")
    assert all(isinstance(path, Path) for path in (audit, benchmark, markdown))


def test_production_config_uses_current_release_paths() -> None:
    text = (ROOT / "configs/gpu_v095_production_persistent.yaml").read_text(encoding="utf-8")
    assert "gpu_v0951_production_persistent" in text
    assert "gpu_v093_production_persistent" not in text
    cfg = load_config(ROOT / "configs/gpu_v095_production_persistent.yaml")
    assert cfg.raqic.gpu_all_cells_required is True
    assert cfg.raqic.max_cells_per_tick is None


def test_per_ow_identity_validation_rejects_missing_duplicate_and_reordering() -> None:
    expected = np.asarray([101, 103, 107, 109], dtype=np.int64)
    np.testing.assert_array_equal(validate_processed_ow_ids(expected, expected), expected)
    with pytest.raises(RuntimeError, match="row-count mismatch"):
        validate_processed_ow_ids(expected, expected[:-1])
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_processed_ow_ids(expected, np.asarray([101, 103, 103, 109]))
    with pytest.raises(RuntimeError, match="reordered"):
        validate_processed_ow_ids(expected, expected[::-1])


def test_source_science_certificate_proves_every_ow_contract(tmp_path: Path) -> None:
    output = tmp_path / "science.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/certify_v0951_science.py",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    every_ow = payload["every_ow"]
    assert every_ow["eligible_cells"] == every_ow["processed_cells"]
    assert every_ow["unique_ow_ids"] == every_ow["ow_id_count"]
    assert every_ow["all_cell_cap_rejected"] is True
    assert payload["adelic"]["product_formula_passed"] is True
    assert payload["quantum_instrument"]["kraus_completeness"]["complete"] is True


def test_release_verifier_detects_mutation(tmp_path: Path) -> None:
    if _active_version() != HISTORICAL_VERSION:
        pytest.skip("v0.9.5.1 release verifier applies only to the frozen v0.9.5.1 tree")
    copied = tmp_path / "repo"
    shutil.copytree(
        ROOT,
        copied,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv*",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "build",
            "dist",
            "*.pyc",
        ),
    )
    shutil.rmtree(copied / "reports", ignore_errors=True)
    shutil.rmtree(copied / "runs", ignore_errors=True)
    clean = subprocess.run(
        [
            sys.executable,
            "scripts/certify_v0951_release_integrity.py",
            "--verify",
        ],
        cwd=copied,
        text=True,
        capture_output=True,
        check=False,
    )
    assert clean.returncode == 0, clean.stdout + clean.stderr

    readme = copied / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nmutation\n", encoding="utf-8")
    mutated = subprocess.run(
        [
            sys.executable,
            "scripts/certify_v0951_release_integrity.py",
            "--verify",
        ],
        cwd=copied,
        text=True,
        capture_output=True,
        check=False,
    )
    assert mutated.returncode != 0
    assert (
        "mismatch" in (mutated.stdout + mutated.stderr).lower()
        or "stale" in (mutated.stdout + mutated.stderr).lower()
    )


def test_per_ow_parallel_chunks_restore_source_order(monkeypatch: pytest.MonkeyPatch) -> None:
    import time
    from types import MethodType
    from typing import Any

    from owl_raqic.qiskit_backend.per_ow_executor import PerOWQiskitExecutor
    from owl_raqic.qiskit_backend.qiskit_policy import (
        QiskitDecisionMode,
        QiskitExecutionPolicy,
        QiskitReadoutPolicy,
    )

    policy = QiskitExecutionPolicy(
        mode=QiskitDecisionMode.EVERY_OW_STATIC_EXACT,
        circuit_families=("static",),
        authoritative_family="static",
        method="statevector",
        device="CPU",
        strict_gpu=False,
        chunk_size=2,
        job_queue_depth=3,
        cache_templates=False,
        readout_policy=QiskitReadoutPolicy.ARGMAX,
        confirm_expensive=True,
    )
    executor = PerOWQiskitExecutor(policy, seed=11)

    def fake_exact(
        self: PerOWQiskitExecutor,
        family: str,
        probabilities: np.ndarray,
        phases: np.ndarray,
        masks: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        del self, family, phases, masks
        # Earlier source chunks take longer so worker completion order differs
        # from source order. The executor must still reconstruct source order.
        time.sleep(float(0.004 * (1.0 - probabilities[0, 0])))
        return probabilities.copy(), [{} for _ in range(probabilities.shape[0])]

    monkeypatch.setattr(
        executor,
        "_execute_exact_chunk",
        MethodType(fake_exact, executor),
    )
    probabilities = np.asarray(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
        ],
        dtype=np.float64,
    )
    ids = np.asarray([41, 43, 47, 53, 59, 61], dtype=np.int64)
    result = executor.execute(
        probabilities,
        np.zeros_like(probabilities),
        np.ones_like(probabilities, dtype=bool),
        ids,
        tick=5,
    )
    np.testing.assert_array_equal(result.authoritative.processed_ow_ids, ids)
    np.testing.assert_allclose(result.authoritative.probabilities, probabilities)
    assert result.metadata["all_ow_accounted"] is True
