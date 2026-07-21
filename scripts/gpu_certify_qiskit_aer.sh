#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="reports/qiskit_gpu_certification_${STAMP}"
mkdir -p "$OUT"
python scripts/gpu_v08_preflight.py --strict --require-qiskit-gpu --out "$OUT/preflight"
python -m pytest -q tests/test_gpu_v08_qiskit_strict.py tests/test_raqic_qiskit_gpu_optional.py | tee "$OUT/pytest_qiskit_gpu.log"
bash scripts/gpu_v08_qiskit_validation_benchmark.sh | tee "$OUT/benchmark.log"
echo "$OUT"
