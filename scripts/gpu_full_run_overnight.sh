#!/usr/bin/env bash
set -euo pipefail
source .venv-gpu-full/bin/activate
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="gpu_full_overnight_${RUN_ID}"
mkdir -p "$LOG_DIR"
python -m owl.experiments.run_single configs/gpu_full_overnight_allcell.yaml --condition integrated 2>&1 | tee "$LOG_DIR/run.log"
tar -czf "${LOG_DIR}.tar.gz" "$LOG_DIR" results/gpu_full_overnight_allcell runs/gpu_full_overnight_allcell configs/gpu_full_overnight_allcell.yaml
echo "Wrote ${LOG_DIR}.tar.gz"
