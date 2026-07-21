#!/usr/bin/env bash
set -euo pipefail
source .venv-gpu-full/bin/activate
python benchmarks/benchmark_gpu_full_stack.py
