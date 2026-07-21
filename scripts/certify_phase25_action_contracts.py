#!/usr/bin/env python3
"""Validate action-transition contracts on the target GPU and fail on incomplete evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

TABLES = (
    "decisions",
    "agent_context",
    "oracle_context",
    "dense_context",
    "candidates",
    "action_directions",
    "execution",
    "events",
    "contributions",
    "information",
    "information_followups",
)
EXPECTED_ACTION_ORDER = (
    "REST",
    "SENSE",
    "MOVE_N",
    "MOVE_S",
    "MOVE_E",
    "MOVE_W",
    "MOVE_NE",
    "MOVE_NW",
    "MOVE_SE",
    "MOVE_SW",
    "FEED",
    "COMMUNICATE",
    "INHIBIT",
    "INTEGRATE",
    "REPAIR",
    "REPRODUCE",
    "INGEST",
    "EXPEL",
    "SPLIT",
    "MERGE",
    "FLEE",
    "PURSUE",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _column_array(column: pa.ChunkedArray) -> np.ndarray[Any, np.dtype[Any]]:
    value = column.combine_chunks()
    if pa.types.is_fixed_size_list(value.type):
        child = value.values.to_numpy(zero_copy_only=False)
        return np.asarray(child).reshape(len(value), int(value.type.list_size))
    return np.asarray(value.to_numpy(zero_copy_only=False))


def _numpy_float_dtype(scalar_type: pa.DataType) -> np.dtype[Any]:
    """Map Arrow floats without importing pandas or its extension dtypes."""

    if pa.types.is_float16(scalar_type):
        return np.dtype(np.float16)
    if pa.types.is_float32(scalar_type):
        return np.dtype(np.float32)
    if pa.types.is_float64(scalar_type):
        return np.dtype(np.float64)
    raise TypeError(f"unsupported Arrow floating type: {scalar_type}")


def _compare_floating_arrays(
    left: np.ndarray[Any, np.dtype[Any]],
    right: np.ndarray[Any, np.dtype[Any]],
    scalar_type: pa.DataType,
    *,
    label: str,
) -> dict[str, Any]:
    """Compare floats with explicit NaN/infinity classes and finite tolerances."""

    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: floating array shapes differ")

    masks = {
        "nan": (np.isnan(a), np.isnan(b)),
        "positive_infinity": (np.isposinf(a), np.isposinf(b)),
        "negative_infinity": (np.isneginf(a), np.isneginf(b)),
    }
    for kind, (a_mask, b_mask) in masks.items():
        if not np.array_equal(a_mask, b_mask):
            raise AssertionError(f"{label}: {kind} classifications differ")

    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    if not np.array_equal(finite_a, finite_b):
        raise AssertionError(f"{label}: finite classifications differ")

    dtype = _numpy_float_dtype(scalar_type)
    finfo = np.finfo(dtype)
    atol = float(8.0 * finfo.eps) if finfo.bits <= 32 else float(64.0 * finfo.eps)
    a_finite = a[finite_a]
    b_finite = b[finite_b]
    difference = np.abs(a_finite.astype(np.float64) - b_finite.astype(np.float64))
    maximum = float(np.max(difference, initial=0.0))
    if not np.allclose(a_finite, b_finite, atol=atol, rtol=0.0, equal_nan=False):
        raise AssertionError(
            f"{label}: float tolerance exceeded (max_abs={maximum}, atol={atol})"
        )
    return {
        "max_abs": maximum,
        "atol": atol,
        "rtol": 0.0,
        "nan_count": int(masks["nan"][0].sum()),
        "positive_infinity_count": int(masks["positive_infinity"][0].sum()),
        "negative_infinity_count": int(masks["negative_infinity"][0].sum()),
    }


def _compare_tables(cpu_root: Path, gpu_root: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name in TABLES:
        left = pads.dataset(cpu_root / f"{name}.parquet", format="parquet").to_table()
        right = pads.dataset(gpu_root / f"{name}.parquet", format="parquet").to_table()
        if left.schema != right.schema:
            raise AssertionError(f"{name}: CPU/GPU Arrow schemas differ")
        if left.num_rows != right.num_rows:
            raise AssertionError(f"{name}: CPU/GPU row counts differ")
        table_result: dict[str, Any] = {
            "rows": left.num_rows,
            "exact_columns": 0,
            "floating_columns": {},
        }
        for field in left.schema:
            left_column = left.column(field.name)
            right_column = right.column(field.name)
            scalar_type = (
                field.type.value_type
                if pa.types.is_fixed_size_list(field.type)
                else field.type
            )
            if pa.types.is_floating(scalar_type):
                a = _column_array(left_column)
                b = _column_array(right_column)
                table_result["floating_columns"][field.name] = _compare_floating_arrays(
                    a,
                    b,
                    scalar_type,
                    label=f"{name}.{field.name}",
                )
            else:
                if left_column.to_pylist() != right_column.to_pylist():
                    raise AssertionError(f"{name}.{field.name}: categorical evidence differs")
                table_result["exact_columns"] += 1
        evidence[name] = table_result
    return evidence


def _require_acceptance(payload: dict[str, Any], *, backend: str) -> None:
    if payload.get("passed") is not True:
        raise AssertionError(f"{backend} acceptance did not pass")
    if payload.get("backend") != backend:
        raise AssertionError(f"acceptance backend is not {backend}")
    if payload.get("ticks") != 25 or payload.get("replay_completed_ticks") != 25:
        raise AssertionError(f"{backend} acceptance is not a complete 25-tick run")
    if payload.get("factual_schema_version") != "owl.cadc.factual.v2":
        raise AssertionError(f"{backend} acceptance did not emit CADC factual v2")
    if tuple(payload.get("action_order", ())) != EXPECTED_ACTION_ORDER:
        raise AssertionError("immutable 22-action ordering changed")
    if payload.get("recorder_on_off_exact") is not True:
        raise AssertionError(f"{backend} recorder-on/off scientific recovery failed")
    rows = payload.get("rows", {})
    decisions = int(rows.get("decisions", -1))
    if int(rows.get("candidates", -1)) != decisions * 22:
        raise AssertionError(f"{backend} candidate cardinality is not 22 per decision")
    if int(rows.get("action_directions", -1)) != decisions * 16:
        raise AssertionError(f"{backend} directional cardinality is not 16 per decision")
    if float(rows.get("max_contribution_reconciliation_error_abs", 1.0)) != 0.0:
        raise AssertionError(f"{backend} contribution reconciliation is not exact")
    selected = rows.get("selected_action_contracts", {})
    successful = rows.get("successful_action_contracts", {})
    for action in ("SENSE", "FLEE", "PURSUE"):
        if int(selected.get(action, 0)) <= 0:
            raise AssertionError(f"{backend} acceptance did not select {action}")
        if int(successful.get(action, 0)) <= 0:
            raise AssertionError(f"{backend} acceptance did not successfully execute {action}")
    telemetry = payload.get("packet_transfer_telemetry", {})
    if int(telemetry.get("packet_count", -1)) != 25:
        raise AssertionError(f"{backend} packet telemetry is incomplete")
    if int(telemetry.get("event_overflow_total", -1)) != 0:
        raise AssertionError(f"{backend} packet telemetry reports overflow")


def certify(args: argparse.Namespace) -> dict[str, Any]:
    cpu_acceptance = Path(args.cpu_acceptance).resolve()
    gpu_acceptance = Path(args.gpu_acceptance).resolve()
    status_path = Path(args.command_status).resolve()
    output = Path(args.output).resolve()
    repository_root = (
        Path(args.repository_root).resolve()
        if getattr(args, "repository_root", None)
        else None
    )
    failures: list[str] = []
    parity: dict[str, Any] = {}
    current_source_sha256: str | None = None
    inputs: dict[str, dict[str, Any]] = {}
    for name, path in (
        ("cpu_acceptance", cpu_acceptance),
        ("gpu_acceptance", gpu_acceptance),
        ("command_status", status_path),
    ):
        try:
            inputs[name] = _json(path)
        except Exception as exc:
            inputs[name] = {}
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    cpu = inputs["cpu_acceptance"]
    gpu = inputs["gpu_acceptance"]
    status = inputs["command_status"]
    try:
        if failures:
            raise AssertionError("one or more required certificate inputs are unavailable")
        _require_acceptance(cpu, backend="numpy")
        _require_acceptance(gpu, backend="cupy")
        if cpu.get("source_sha256") != gpu.get("source_sha256"):
            raise AssertionError("CPU and GPU candidates used different source scopes")
        if repository_root is not None:
            source_root = repository_root / "src"
            if str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))
            from owl.experiments.controller import _release_hash

            current_source_sha256 = _release_hash(repository_root)
            if cpu.get("source_sha256") != current_source_sha256:
                raise AssertionError(
                    "acceptance source scope does not match the current repository"
                )
        if cpu.get("config_sha256") != gpu.get("config_sha256"):
            raise AssertionError("CPU and GPU candidates used different configuration")
        if cpu.get("schema_digest") != gpu.get("schema_digest"):
            raise AssertionError("CPU and GPU CADC schema digests differ")
        device = gpu.get("cuda_device")
        if not isinstance(device, dict) or not str(device.get("name", "")):
            raise AssertionError("positive CUDA device metadata is absent")
        if not re.search(args.allowed_device_regex, str(device["name"]), flags=re.IGNORECASE):
            raise AssertionError(f"CUDA device is outside target class: {device['name']}")
        gpu_transfer = gpu["packet_transfer_telemetry"]
        if int(gpu_transfer.get("transfer_count_total", 0)) <= 0:
            raise AssertionError("GPU packet telemetry has no device-to-host transfers")
        if int(gpu_transfer.get("packet_bytes_max", 0)) > int(
            gpu["configured_device_buffer_limit"]
        ):
            raise AssertionError("GPU packet exceeded the configured device buffer limit")
        required_status = (
            "environment_preflight",
            "pip_check",
            "certifier_smoke",
            "acceptance_numpy",
            "acceptance_cupy",
            "pytest",
            "ruff",
        )
        failed_commands = {
            name: status.get(name)
            for name in required_status
            if status.get(name) != 0
        }
        if failed_commands:
            raise AssertionError(f"required command gates failed: {failed_commands}")
        cpu_root = cpu_acceptance.parent / "bundle" / "analysis" / "cadc_v2"
        gpu_root = gpu_acceptance.parent / "bundle" / "analysis" / "cadc_v2"
        parity = _compare_tables(cpu_root, gpu_root)
    except Exception as exc:  # fail-closed certificate is always materialized
        failure = f"{type(exc).__name__}: {exc}"
        if failure not in failures:
            failures.append(failure)

    passed = not failures
    certificate = {
        "schema_version": "owl.phase2.5.target-gpu-certificate.v1",
        "passed": passed,
        "classification": (
            "PHASE2_5_TARGET_GPU_CERTIFIED" if passed else "FAILED_CLOSED"
        ),
        "phase3_unlocked": passed,
        "phase4_unlocked": False,
        "failures": failures,
        "source_sha256": gpu.get("source_sha256"),
        "current_source_sha256": (
            current_source_sha256 if repository_root is not None else None
        ),
        "config_sha256": gpu.get("config_sha256"),
        "cadc_schema_digest": gpu.get("schema_digest"),
        "cuda_device": gpu.get("cuda_device"),
        "command_status": status,
        "cpu_gpu_parity": parity,
        "declared_float_tolerance": {
            "float32_atol": float(8.0 * np.finfo(np.float32).eps),
            "float64_atol": float(64.0 * np.finfo(np.float64).eps),
            "rtol": 0.0,
        },
        "packet_transfer_telemetry": gpu.get("packet_transfer_telemetry"),
        "qiskit_aer_gpu": "not_exercised_by_phase2_5_acceptance_config",
        "input_checksums": {
            "cpu_acceptance": _sha256(cpu_acceptance) if cpu_acceptance.is_file() else None,
            "gpu_acceptance": _sha256(gpu_acceptance) if gpu_acceptance.is_file() else None,
            "command_status": _sha256(status_path) if status_path.is_file() else None,
        },
        "gpu_evidence_files": (
            _artifact_hashes(gpu_acceptance.parent)
            if gpu_acceptance.parent.is_dir()
            else []
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return certificate


def _self_test() -> dict[str, Any]:
    for scalar_type in (pa.float16(), pa.float32(), pa.float64()):
        dtype = _numpy_float_dtype(scalar_type)
        values = np.asarray([0.0, 1.0, np.nan, np.inf, -np.inf], dtype=dtype)
        _compare_floating_arrays(values, values.copy(), scalar_type, label=str(scalar_type))
    try:
        _compare_floating_arrays(
            np.asarray([np.inf], dtype=np.float32),
            np.asarray([-np.inf], dtype=np.float32),
            pa.float32(),
            label="nonfinite_probe",
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("non-finite mismatch probe did not fail closed")
    return {
        "passed": True,
        "pandas_required": False,
        "float_types": ["float16", "float32", "float64"],
        "nonfinite_classification_checked": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-acceptance")
    parser.add_argument("--gpu-acceptance")
    parser.add_argument("--command-status")
    parser.add_argument("--output")
    parser.add_argument("--repository-root")
    parser.add_argument("--allowed-device-regex", default=r"H100|H200|B200")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), indent=2, sort_keys=True))
        return 0
    missing = [
        name
        for name in ("cpu_acceptance", "gpu_acceptance", "command_status", "output")
        if not getattr(args, name)
    ]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    certificate = certify(args)
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0 if certificate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
