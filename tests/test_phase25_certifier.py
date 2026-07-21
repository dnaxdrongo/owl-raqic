from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _certifier() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "certify_phase25_action_contracts.py"
    spec = importlib.util.spec_from_file_location("phase25_certifier_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_table(root: Path, table: pa.Table) -> None:
    destination = root / "sample.parquet"
    destination.mkdir(parents=True)
    pq.write_table(table, destination / "part-000000.parquet")


def test_float_mapping_and_comparison_do_not_require_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _certifier()
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert module._numpy_float_dtype(pa.float16()) == np.dtype(np.float16)
    assert module._numpy_float_dtype(pa.float32()) == np.dtype(np.float32)
    assert module._numpy_float_dtype(pa.float64()) == np.dtype(np.float64)
    values = np.asarray([0.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    result = module._compare_floating_arrays(
        values, values.copy(), pa.float32(), label="probe"
    )
    assert result["negative_infinity_count"] == 1
    assert result["positive_infinity_count"] == 1
    assert result["nan_count"] == 1


def test_scalar_and_fixed_list_columns_are_compared_without_pandas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _certifier()
    cpu = tmp_path / "cpu"
    gpu = tmp_path / "gpu"
    table = pa.table(
        {
            "category": pa.array([1, 2], type=pa.int16()),
            "scalar": pa.array([1.0, 2.0], type=pa.float32()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float32()), 2
            ),
        }
    )
    _write_table(cpu, table)
    _write_table(gpu, table)
    monkeypatch.setattr(module, "TABLES", ("sample",))
    monkeypatch.setitem(sys.modules, "pandas", None)
    evidence = module._compare_tables(cpu, gpu)
    assert evidence["sample"]["rows"] == 2
    assert set(evidence["sample"]["floating_columns"]) == {"scalar", "vector"}


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([np.inf], [-np.inf], "positive_infinity"),
        ([np.nan], [0.0], "nan"),
        ([0.0], [2.0], "float tolerance exceeded"),
    ],
)
def test_float_mismatches_fail_closed(
    left: list[float], right: list[float], message: str
) -> None:
    module = _certifier()
    with pytest.raises(AssertionError, match=message):
        module._compare_floating_arrays(
            np.asarray(left, dtype=np.float32),
            np.asarray(right, dtype=np.float32),
            pa.float32(),
            label="probe",
        )


def test_categorical_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _certifier()
    cpu = tmp_path / "cpu"
    gpu = tmp_path / "gpu"
    _write_table(cpu, pa.table({"category": pa.array([1], type=pa.int16())}))
    _write_table(gpu, pa.table({"category": pa.array([2], type=pa.int16())}))
    monkeypatch.setattr(module, "TABLES", ("sample",))
    with pytest.raises(AssertionError, match="categorical evidence differs"):
        module._compare_tables(cpu, gpu)


def test_missing_inputs_always_materialize_failed_closed_certificate(tmp_path: Path) -> None:
    module = _certifier()
    output = tmp_path / "certificate.json"
    args = argparse.Namespace(
        cpu_acceptance=str(tmp_path / "missing_cpu.json"),
        gpu_acceptance=str(tmp_path / "missing_gpu.json"),
        command_status=str(tmp_path / "missing_status.json"),
        output=str(output),
        allowed_device_regex=r"H100|H200|B200",
        repository_root=None,
    )
    certificate = module.certify(args)
    assert output.is_file()
    assert certificate["passed"] is False
    assert certificate["classification"] == "FAILED_CLOSED"
    assert certificate["phase3_unlocked"] is False
    assert len(certificate["failures"]) >= 3


def test_certifier_self_test_passes_without_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _certifier()
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert module._self_test()["passed"] is True
