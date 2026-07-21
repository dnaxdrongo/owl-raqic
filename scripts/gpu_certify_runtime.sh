#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="reports/gpu_certification_${STAMP}"
mkdir -p "$OUT"
python scripts/gpu_v08_preflight.py --strict --out "$OUT/preflight"
python -m pytest -q tests/test_gpu_full_loop_optional.py tests/test_gpu_v08_slabs.py | tee "$OUT/pytest_gpu_runtime.log"
python -m owl.gpu.run_context configs/gpu_v08_validation_audit64.yaml --steps 10 | tee "$OUT/runtime.log"
python scripts/write_run_manifest.py configs/gpu_v08_validation_audit64.yaml --out "$OUT/run_manifest.json"
python scripts/hash_run_environment.py --out "$OUT/environment.json"
echo "$OUT"
