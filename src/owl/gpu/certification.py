from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
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


def sha256_path(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    if p.is_file():
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    else:
        for file_path in sorted(x for x in p.rglob("*") if x.is_file() and ".git" not in x.parts):
            h.update(str(file_path.relative_to(p)).encode())
            h.update(file_path.read_bytes())
    return h.hexdigest()


def _nvidia_smi() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
        return {"available": True, "rows": [row.strip() for row in out.splitlines() if row.strip()]}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


@dataclass
class CertificationReport:
    strict_gpu_requested: bool
    cupy_gpu_available: bool
    qiskit_gpu_available: bool
    graph_captured_segments: list[str]
    fallback_count: int
    all_cells_satisfied: bool
    environment: dict[str, Any]
    versions: dict[str, Any]
    device: dict[str, Any]
    graph_status: dict[str, Any]
    failures: list[str]
    repo_sha256: str | None = None

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["passed"] = self.passed
        return out

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "gpu_runtime_certification.json"
        md_path = directory / "GPU_RUNTIME_CERTIFICATION.md"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lines = [
            "# GPU Runtime Certification",
            "",
            f"**Passed:** {self.passed}",
            f"**CuPy GPU:** {self.cupy_gpu_available}",
            f"**Qiskit Aer GPU:** {self.qiskit_gpu_available}",
            f"**Fallback count:** {self.fallback_count}",
            f"**All cells satisfied:** {self.all_cells_satisfied}",
            f"**Captured graph segments:** {', '.join(self.graph_captured_segments) or 'none'}",
            "",
            "## Failures",
            *(f"- {x}" for x in self.failures),
            "",
            "## Device",
            "```json",
            json.dumps(self.device, indent=2, sort_keys=True),
            "```",
        ]
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return json_path, md_path


def certify_runtime(
    *,
    run_context: Any | None = None,
    strict_gpu: bool = True,
    require_qiskit_gpu: bool = False,
    repo_path: str | Path | None = None,
) -> CertificationReport:
    cupy = detect_cupy()
    qiskit = aer_gpu_available_runtime()
    graph_status = (
        run_context.graph_manager.graph_status()
        if run_context and run_context.graph_manager
        else {}
    )
    fallback_count = int(getattr(run_context, "fallback_count", 0))
    last = getattr(run_context, "last_diagnostics", {}) if run_context else {}
    raqic = last.get("raqic", {}) if isinstance(last, dict) else {}
    metric = (
        getattr(run_context, "metrics", [])[-1]
        if run_context is not None and getattr(run_context, "metrics", [])
        else {}
    )
    if metric and "all_ow_accounted" in metric:
        all_cells = bool(metric["all_ow_accounted"])
    elif raqic and raqic.get("eligible_cells") is not None:
        all_cells = int(raqic.get("eligible_cells", 0)) == int(raqic.get("processed_cells", 0))
    else:
        all_cells = True
    failures: list[str] = []
    if strict_gpu and not cupy.available:
        failures.append(f"CuPy/CUDA unavailable: {cupy.error}")
    if strict_gpu and fallback_count:
        failures.append(f"strict GPU run recorded {fallback_count} fallback(s)")
    if require_qiskit_gpu and not qiskit.available:
        failures.append(f"Qiskit-Aer-GPU unavailable: {qiskit.error}")
    if not all_cells:
        failures.append("processed cell count differs from eligible cell count")
    return CertificationReport(
        strict_gpu_requested=bool(strict_gpu),
        cupy_gpu_available=bool(cupy.available),
        qiskit_gpu_available=bool(qiskit.available),
        graph_captured_segments=list(graph_status.get("captured_segments", [])),
        fallback_count=fallback_count,
        all_cells_satisfied=all_cells,
        environment={
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": os.getcwd(),
        },
        versions={
            name: _version(name)
            for name in ("numpy", "cupy-cuda12x", "qiskit", "qiskit-aer", "vispy", "pygame")
        },
        device={"cupy": cupy.to_dict(), "nvidia_smi": _nvidia_smi(), "qiskit": asdict(qiskit)},
        graph_status=graph_status,
        failures=failures,
        repo_sha256=sha256_path(repo_path) if repo_path else None,
    )
