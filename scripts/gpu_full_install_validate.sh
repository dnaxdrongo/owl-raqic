#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="gpu_full_validation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
python3.11 -m venv .venv-gpu-full
source .venv-gpu-full/bin/activate
python -m pip install --upgrade pip setuptools wheel | tee "$LOG_DIR/01_install.log"
pip install -e ".[dev,raqic,gpu-full]" | tee -a "$LOG_DIR/01_install.log"
(nvidia-smi || true) | tee "$LOG_DIR/02_nvidia_smi.log"
python - <<'PY' | tee "$LOG_DIR/03_gpu_smoke.log"
from owl.gpu.backend import get_array_backend
b = get_array_backend(strict=True, allow_fallback=False)
print(b.info)
PY
python -m pytest -ra -vv --tb=short | tee "$LOG_DIR/04_pytest_all.log"
python -m pytest tests/test_gpu_full_loop_optional.py -ra -vv --tb=short | tee "$LOG_DIR/05_pytest_gpu_full.log"
tar -czf "${LOG_DIR}.tar.gz" "$LOG_DIR"
echo "Wrote ${LOG_DIR}.tar.gz"
