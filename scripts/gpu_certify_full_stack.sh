#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash scripts/gpu_certify_runtime.sh
bash scripts/gpu_certify_qiskit_aer.sh
python scripts/audit_gpu_hotspots.py --mode report --strict --out reports/gpu_v08_hotspot_strict.md
python scripts/check_docs_against_repo.py --strict
python scripts/safe_collect_artifacts.py --out owl_v08_certification_artifacts.zip
echo "$ROOT/owl_v08_certification_artifacts.zip"
