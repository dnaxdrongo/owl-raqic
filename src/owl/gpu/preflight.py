from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from owl_raqic.gpu.backend import detect_cupy
from owl_raqic.qiskit_backend.gpu_execution import aer_gpu_available_runtime


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command(command: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        cp = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "available": True,
            "returncode": cp.returncode,
            "stdout": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


@dataclass
class GPUPreflightReport:
    python: str
    platform: str
    cupy: dict[str, Any]
    qiskit_aer_gpu: dict[str, Any]
    nvidia_smi: dict[str, Any]
    nsight_systems: bool
    nsight_compute: bool
    vispy_installed: bool
    pygame_installed: bool
    versions: dict[str, str | None]
    graph_available: bool
    strict_ready: bool
    warnings: list[str]
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        jp = directory / "gpu_v08_preflight.json"
        mp = directory / "GPU_V0_8_PREFLIGHT.md"
        jp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = [
            "# OWL + RAQIC GPU v0.8 Preflight",
            "",
            f"- Strict ready: **{self.strict_ready}**",
            f"- CuPy/CUDA: **{self.cupy.get('available', False)}**",
            f"- Qiskit-Aer-GPU: **{self.qiskit_aer_gpu.get('available', False)}**",
            f"- CUDA Graph support: **{self.graph_available}**",
            f"- Nsight Systems: **{self.nsight_systems}**",
            f"- Nsight Compute: **{self.nsight_compute}**",
            "",
            "## Warnings",
            *[f"- {x}" for x in self.warnings],
            "",
            "## Failures",
            *[f"- {x}" for x in self.failures],
            "",
            "## Environment",
            "```json",
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            "```",
        ]
        mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return jp, mp


def run_preflight(*, require_qiskit_gpu: bool = False) -> GPUPreflightReport:
    cupy_info = detect_cupy()
    qiskit_info = aer_gpu_available_runtime()
    graph_available = False
    if cupy_info.available:
        try:
            import cupy as cp

            graph_available = all(
                hasattr(cp.cuda.Stream, name) for name in ("begin_capture", "end_capture")
            )
        except Exception:
            graph_available = False
    warnings: list[str] = []
    failures: list[str] = []
    if not cupy_info.available:
        failures.append(f"CuPy/CUDA unavailable: {cupy_info.error}")
    if require_qiskit_gpu and not qiskit_info.available:
        failures.append(f"Qiskit-Aer-GPU unavailable: {qiskit_info.error}")
    if cupy_info.available and not graph_available:
        warnings.append("CUDA graph capture is unavailable; graph benchmark is invalid.")
    if not shutil.which("nsys"):
        warnings.append("Nsight Systems CLI not found; CUDA-event profiling remains available.")
    if not shutil.which("ncu"):
        warnings.append("Nsight Compute CLI not found; kernel microprofile remains unavailable.")
    try:
        import vispy  # noqa: F401

        vispy_installed = True
    except Exception:
        vispy_installed = False
        warnings.append("VisPy not installed; interactive GPU viewer unavailable.")
    try:
        import pygame  # noqa: F401

        pygame_installed = True
    except Exception:
        pygame_installed = False
    strict_ready = cupy_info.available and (qiskit_info.available or not require_qiskit_gpu)
    return GPUPreflightReport(
        python=sys.version,
        platform=platform.platform(),
        cupy=cupy_info.to_dict(),
        qiskit_aer_gpu=qiskit_info.to_dict(),
        nvidia_smi=_command(["nvidia-smi", "-L"])
        if shutil.which("nvidia-smi")
        else {"available": False},
        nsight_systems=bool(shutil.which("nsys")),
        nsight_compute=bool(shutil.which("ncu")),
        vispy_installed=vispy_installed,
        pygame_installed=pygame_installed,
        versions={
            name: _version(name)
            for name in (
                "cupy-cuda12x",
                "qiskit",
                "qiskit-aer",
                "qiskit-aer-gpu",
                "vispy",
                "pygame",
            )
        },
        graph_available=graph_available,
        strict_ready=bool(strict_ready),
        warnings=warnings,
        failures=failures,
    )
