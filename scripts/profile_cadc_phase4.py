#!/usr/bin/env python3
"""Profile canonical ETL and a fixed-shape CADC-MORE 2 forward/backward batch."""

from __future__ import annotations

import argparse
import copy
import platform
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.gpu_io import device_memory_snapshot, to_torch_dlpack  # noqa: E402
from owl.cadc.models import CADCMore2Suite  # noqa: E402
from owl.cadc.models.transition import StructuralTransitionModel  # noqa: E402
from owl.cadc.pipeline import load_phase4_tensors  # noqa: E402


def _precision_context(precision: str) -> object:
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        import torch

        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise RuntimeError(
        "B200 FP8 profiling is locked pending an independently certified parity path"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    started = time.perf_counter()
    batch = load_phase4_tensors(
        Path(args.dataset) / "canonical_data",
        backend=config.runtime.backend,
        history_length=config.features.history_length,
        quantile_levels=config.scalarization.quantiles,
        cvar_alpha=config.scalarization.cvar_alpha,
    )
    etl_seconds = time.perf_counter() - started
    payload = {
        "schema_version": "owl.cadc.phase4-performance.v1",
        "target": config.runtime.target.value,
        "backend": config.runtime.backend,
        "precision": config.runtime.precision,
        "python": platform.python_version(),
        "decision_count": len(batch.decision_ids),
        "etl_seconds": etl_seconds,
        "etl_decisions_per_second": len(batch.decision_ids) / max(etl_seconds, 1e-12),
        "memory_before_model": device_memory_snapshot(),
        "phase5_locked": True,
    }
    if config.runtime.target.value != "cpu":
        import cupy as cp
        import torch

        if not torch.cuda.is_available() or not cp.cuda.is_available():
            raise RuntimeError("target profile requires positive CuPy and Torch CUDA")
        if config.runtime.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cuda.matmul.allow_tf32 = False
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode()
        expected_token = {
            "h100": "H100",
            "h200": "H200",
            "b200": "B200",
        }[config.runtime.target.value]
        if expected_token not in str(name).upper():
            raise RuntimeError(
                f"target/device mismatch: requested {config.runtime.target.value}, found {name}"
            )
        context = to_torch_dlpack(batch.context)
        candidates = to_torch_dlpack(batch.candidates)
        candidate_mask = to_torch_dlpack(batch.candidate_mask).bool()
        directions = to_torch_dlpack(batch.directions)
        direction_mask = to_torch_dlpack(batch.direction_mask).bool()
        dlpack_zero_copy = {
            "context": context.data_ptr() == int(batch.context.data.ptr),
            "candidates": candidates.data_ptr() == int(batch.candidates.data.ptr),
            "directions": directions.data_ptr() == int(batch.directions.data.ptr),
        }
        if not all(dlpack_zero_copy.values()):
            raise AssertionError("CuPy-to-Torch DLPack transfer was not zero-copy")
        size = min(config.runtime.batch_size, context.shape[0])
        eager = CADCMore2Suite(
            StructuralTransitionModel(
                context_dim=context.shape[-1],
                candidate_dim=candidates.shape[-1],
                direction_dim=directions.shape[-1],
                hidden_dim=config.models.hidden_width,
                outcome_dim=batch.outcomes.shape[-1],
                quantile_count=len(config.scalarization.quantiles),
                time_bins=batch.outcomes.shape[1],
                death_causes=4,
                depth=config.models.depth,
                dropout=config.models.dropout,
            ),
            config.models.hidden_width,
        ).cuda()
        eager.eval()
        horizon = torch.zeros(size, device="cuda", dtype=torch.long)
        inputs = (
            context[:size],
            candidates[:size],
            directions[:size],
            direction_mask[:size],
            horizon,
        )
        with torch.no_grad():
            fp32_gpu = eager(*inputs)
            repeated_gpu = eager(*inputs)
            single_gpu = eager(
                context[:1],
                candidates[:1],
                directions[:1],
                direction_mask[:1],
                horizon[:1],
            )
        deterministic_error = float(
            (fp32_gpu["rank_score"] - repeated_gpu["rank_score"])
            .abs()
            .max()
            .cpu()
        )
        batch_single_error = float(
            (fp32_gpu["rank_score"][:1] - single_gpu["rank_score"])
            .abs()
            .max()
            .cpu()
        )
        cpu_model = copy.deepcopy(eager).cpu().eval()
        with torch.no_grad():
            cpu_output = cpu_model(
                torch.as_tensor(cp.asnumpy(batch.context[:size])),
                torch.as_tensor(cp.asnumpy(batch.candidates[:size])),
                torch.as_tensor(cp.asnumpy(batch.directions[:size])),
                torch.as_tensor(cp.asnumpy(batch.direction_mask[:size])).bool(),
                torch.zeros(size, dtype=torch.long),
            )
        cpu_gpu_error = float(
            (
                fp32_gpu["rank_score"].detach().cpu()
                - cpu_output["rank_score"].detach().cpu()
            )
            .abs()
            .max()
        )
        if deterministic_error > 1e-7 or batch_single_error > 1e-5:
            raise AssertionError("deterministic or batch/single inference parity failed")
        if cpu_gpu_error > 2e-4:
            raise AssertionError("FP32 CPU/GPU inference parity failed")
        active_model = eager
        compile_error = 0.0
        if config.runtime.compile:
            active_model = torch.compile(copy.deepcopy(eager), dynamic=False)
            active_model.eval()
            with torch.no_grad(), _precision_context(config.runtime.precision):
                eager_output = eager(*inputs)
                compiled_output = active_model(*inputs)
            compile_error = float(
                (eager_output["rank_score"] - compiled_output["rank_score"])
                .abs()
                .max()
                .cpu()
            )
            tolerance = 1e-5 if config.runtime.precision == "fp32" else 5e-3
            if compile_error > tolerance:
                raise AssertionError("eager/compiled inference parity failed")
        active_model.train()
        optimizer = torch.optim.AdamW(
            active_model.parameters(), lr=config.training.learning_rate
        )
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(config.runtime.precision):
                output = active_model(
                    context[:size],
                    candidates[:size],
                    directions[:size],
                    direction_mask[:size],
                    torch.zeros(size, device="cuda", dtype=torch.long),
                )
                loss = output["outcome_mean"].square().mean()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        iterations = 20
        started = time.perf_counter()
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(config.runtime.precision):
                output = active_model(
                    context[:size],
                    candidates[:size],
                    directions[:size],
                    direction_mask[:size],
                    torch.zeros(size, device="cuda", dtype=torch.long),
                )
                loss = output["outcome_mean"].square().mean()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        payload.update(
            {
                "device": str(name),
                "compute_capability": f"{properties['major']}.{properties['minor']}",
                "multiprocessor_count": int(properties["multiProcessorCount"]),
                "total_global_memory_bytes": int(properties["totalGlobalMem"]),
                "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
                "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
                "cupy_version": cp.__version__,
                "torch_version": torch.__version__,
                "batch_size": size,
                "training_steps": iterations,
                "training_seconds": elapsed,
                "examples_per_second": iterations * size / elapsed,
                "peak_torch_device_bytes": int(torch.cuda.max_memory_allocated()),
                "max_device_bytes": config.runtime.max_device_bytes,
                "within_device_memory_bound": int(torch.cuda.max_memory_allocated())
                <= config.runtime.max_device_bytes,
                "candidate_mask_true": int(candidate_mask[:size].sum().cpu()),
                "dlpack_zero_copy": dlpack_zero_copy,
                "deterministic_repeat_max_abs_error": deterministic_error,
                "batch_single_max_abs_error": batch_single_error,
                "fp32_cpu_gpu_max_abs_error": cpu_gpu_error,
                "eager_compiled_max_abs_error": compile_error,
                "compiled_path_exercised": bool(config.runtime.compile),
                "memory_after_model": device_memory_snapshot(),
            }
        )
        if not payload["within_device_memory_bound"]:
            raise MemoryError("Phase 4 training profile exceeded device-memory bound")
    payload["passed"] = True
    atomic_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
