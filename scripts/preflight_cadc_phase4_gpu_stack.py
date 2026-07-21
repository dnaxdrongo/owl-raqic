#!/usr/bin/env python3
"""Exercise the exact CUDA dataframe/model APIs used after the paid corpus."""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402


def _device_name(properties: dict[str, Any]) -> str:
    value = properties["name"]
    return value.decode() if isinstance(value, bytes) else str(value)


def _gpu_smoke(config_path: Path) -> dict[str, Any]:
    import cudf
    import cupy as cp
    import torch
    import xgboost as xgb
    from cuml.neighbors import NearestNeighbors

    config = load_phase4_config(config_path)
    if not cp.cuda.is_available() or not torch.cuda.is_available():
        raise RuntimeError("CuPy and Torch must both report CUDA availability")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    device = _device_name(properties)
    expected = config.runtime.target.value.upper()
    if expected not in device.upper():
        raise RuntimeError(f"target/device mismatch: expected {expected}, found {device}")

    # Mirror the post-corpus dataframe operations before paid work starts.
    decision = cudf.DataFrame(
        {
            "source_decision_id": ["d0", "d1"],
            "seed": cp.asarray([101, 102], dtype=cp.int64),
            "selected_action": cp.asarray([1, 2], dtype=cp.int16),
        }
    )
    candidate = cudf.DataFrame(
        {
            "source_decision_id": ["d0", "d0", "d1", "d1"],
            "action_index": cp.asarray([0, 1, 0, 1], dtype=cp.int16),
            "policy_legal": cp.asarray([True, True, True, False]),
            "prechoice_executable": cp.asarray([True, True, True, False]),
        }
    )
    target = cudf.DataFrame(
        {
            "source_decision_id": ["d0", "d0", "d1", "d1"],
            "forced_action": cp.asarray([0, 1, 0, 1], dtype=cp.int16),
            "repeat_index": cp.asarray([0, 0, 0, 0], dtype=cp.int16),
            "horizon": cp.asarray([1, 1, 1, 1], dtype=cp.int16),
            "value": cp.asarray([1.0, 2.0, 3.0, 4.0], dtype=cp.float32),
        }
    )
    if not bool(decision["source_decision_id"].is_unique):
        raise AssertionError("cuDF many-to-one right-key uniqueness contract failed")
    joined = candidate.merge(
        decision[["source_decision_id", "seed"]],
        on="source_decision_id",
        how="inner",
    ).sort_values(["source_decision_id", "action_index"])
    if len(joined) != len(candidate):
        raise AssertionError("cuDF many-to-one join cardinality contract failed")
    keys = ["source_decision_id", "forced_action", "horizon"]
    grouped = target.groupby(keys)
    count = grouped.size().reset_index(name="repeat_count")
    unique = grouped["repeat_index"].nunique().reset_index(name="repeat_unique")
    extrema = grouped["value"].agg(["min", "max"]).reset_index()
    if len(count) != 4 or len(unique) != 4 or len(extrema) != 4:
        raise AssertionError("cuDF grouped validation contract failed")

    list_frame = cudf.DataFrame(
        {"vector": cudf.Series([[1.0, 2.0], [3.0, 4.0]])}
    )
    leaves = list_frame["vector"].list.leaves.values.reshape(2, 2)
    if not bool(cp.array_equal(leaves, cp.asarray([[1.0, 2.0], [3.0, 4.0]]))):
        raise AssertionError("cuDF fixed-list extraction contract failed")
    arrow_rows = decision[["source_decision_id"]].to_arrow().to_pylist()
    if arrow_rows != [{"source_decision_id": "d0"}, {"source_decision_id": "d1"}]:
        raise AssertionError("cuDF compact Arrow metadata transfer failed")

    with tempfile.TemporaryDirectory(prefix="owl_phase4_gpu_smoke_") as temporary:
        parquet = Path(temporary) / "smoke.parquet"
        target.to_parquet(parquet, index=False)
        restored = cudf.read_parquet(parquet).sort_values(
            ["source_decision_id", "forced_action"]
        )
        if len(restored) != len(target):
            raise AssertionError("cuDF Parquet round-trip row count failed")

    # Verify zero-copy CuPy/Torch interchange and BF16 tensor-core execution.
    cupy_tensor = cp.arange(256, dtype=cp.float32).reshape(16, 16)
    torch_tensor = torch.utils.dlpack.from_dlpack(cupy_tensor)
    if torch_tensor.data_ptr() != int(cupy_tensor.data.ptr):
        raise AssertionError("CuPy-to-Torch DLPack was not zero-copy")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        product = torch_tensor @ torch_tensor.T
    if not bool(torch.isfinite(product).all()):
        raise FloatingPointError("Torch BF16 smoke produced nonfinite output")

    # Exercise the two auxiliary CUDA libraries used by training/calibration.
    design = cp.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=cp.float32
    )
    labels = cp.asarray([0.0, 1.0, 1.0, 2.0], dtype=cp.float32)
    matrix = xgb.QuantileDMatrix(design, label=labels)
    booster = xgb.train(
        {"objective": "reg:squarederror", "tree_method": "hist", "device": "cuda"},
        matrix,
        num_boost_round=2,
        verbose_eval=False,
    )
    prediction = booster.inplace_predict(design)
    if prediction.shape != (4,):
        raise AssertionError("XGBoost CUDA prediction shape failed")
    neighbors = NearestNeighbors(n_neighbors=2, algorithm="brute", output_type="cupy")
    neighbors.fit(design)
    distances, indices = neighbors.kneighbors(design[:2])
    if distances.shape != (2, 2) or indices.shape != (2, 2):
        raise AssertionError("cuML CUDA nearest-neighbor contract failed")
    covariance = cp.cov(design.astype(cp.float64), rowvar=False)
    precision = cp.linalg.pinv(
        covariance + cp.eye(covariance.shape[0], dtype=cp.float64) * 1e-8
    )
    centered = design.astype(cp.float64) - design.mean(axis=0, dtype=cp.float64)
    mahalanobis = cp.sqrt(
        cp.maximum(cp.einsum("bi,ij,bj->b", centered, precision, centered), 0.0)
    )
    if not bool(cp.isfinite(mahalanobis).all()):
        raise FloatingPointError("CuPy float64 support geometry is nonfinite")

    cp.cuda.get_current_stream().synchronize()
    free, total = cp.cuda.runtime.memGetInfo()
    aggregate_budget = int(config.runtime.max_device_bytes)
    if aggregate_budget > int(free):
        raise MemoryError(
            "configured aggregate device budget exceeds post-stack free memory: "
            f"{aggregate_budget} > {int(free)}"
        )
    per_worker_budget = aggregate_budget // int(config.runtime.corpus_workers)
    if per_worker_budget <= 0:
        raise MemoryError("configured corpus workers receive no device-memory budget")
    return {
        "device": device,
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        "cupy_version": cp.__version__,
        "cudf_version": cudf.__version__,
        "torch_version": torch.__version__,
        "xgboost_version": xgb.__version__,
        "dataframe_join_rows": len(joined),
        "dataframe_group_rows": len(count),
        "fixed_list_width": int(leaves.shape[1]),
        "parquet_roundtrip_rows": len(target),
        "dlpack_zero_copy": True,
        "torch_bf16": True,
        "xgboost_cuda": True,
        "cuml_cuda": True,
        "support_geometry_cuda_float64": True,
        "device_free_bytes": int(free),
        "device_total_bytes": int(total),
        "aggregate_device_budget_bytes": aggregate_budget,
        "corpus_workers": int(config.runtime.corpus_workers),
        "per_worker_device_budget_bytes": per_worker_budget,
        "physical_headroom_after_aggregate_budget_bytes": int(total)
        - aggregate_budget,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    config = load_phase4_config(config_path)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema_version": "owl.cadc.phase4-gpu-stack-smoke.v1",
        "target": config.runtime.target.value,
        "backend": config.runtime.backend,
        "precision": config.runtime.precision,
        "python": platform.python_version(),
        "phase5_locked": True,
    }
    try:
        if config.runtime.target.value == "cpu":
            payload.update({"passed": True, "skipped": True, "reason": "cpu_reference"})
        else:
            payload.update(_gpu_smoke(config_path))
            payload.update({"passed": True, "skipped": False})
    except Exception as exc:
        payload.update(
            {
                "passed": False,
                "skipped": False,
                "exception_type": type(exc).__name__,
                "failure": str(exc),
            }
        )
    payload["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
