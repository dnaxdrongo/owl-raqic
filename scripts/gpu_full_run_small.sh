#!/usr/bin/env bash
set -euo pipefail
source .venv-gpu-full/bin/activate
python -m owl.experiments.run_single configs/gpu_full_small.yaml --condition integrated
