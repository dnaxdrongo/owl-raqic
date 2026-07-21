#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/gpu_v09_certify_graph.py   configs/gpu_v09_full_graph_small.yaml   --steps 5   --out reports/v09_graph_certification
