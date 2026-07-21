#!/usr/bin/env python3
"""Capture a complete target-specific software and device manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.experiments.controller import _release_hash  # noqa: E402


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    payload: dict[str, object] = {
        "schema_version": "owl.cadc.phase4-environment.v1",
        "target": config.runtime.target.value,
        "precision": config.runtime.precision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "packages": dict(sorted(packages.items(), key=lambda value: value[0].lower())),
        "required_versions": {
            name: _version(name)
            for name in (
                "numpy",
                "scipy",
                "pyarrow",
                "polars",
                "scikit-learn",
                "torch",
                "xgboost",
                "cupy-cuda12x",
                "cudf-cu12",
                "cuml-cu12",
                "pandas",
                "pydantic",
                "sympy",
            )
        },
        "corpus_contract_sha256": config.corpus_digest(),
        "model_spec_sha256": config.model_spec_digest(),
        "phase4_source_sha256": _release_hash(ROOT),
        "phase5_locked": True,
    }
    if config.runtime.target.value != "cpu":
        import cupy as cp
        import torch

        if not cp.cuda.is_available() or not torch.cuda.is_available():
            raise RuntimeError("target environment manifest requires positive CuPy/Torch CUDA")
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode()
        expected = config.runtime.target.value.upper()
        if expected not in str(name).upper():
            raise RuntimeError(f"target/device mismatch: expected {expected}, found {name}")
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        payload["cuda"] = {
            "device": str(name),
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nvidia_smi": query.stdout.strip().splitlines(),
        }
    payload["passed"] = True
    atomic_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
