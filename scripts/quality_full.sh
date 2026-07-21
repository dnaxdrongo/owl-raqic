#!/usr/bin/env bash
set -euo pipefail
python scripts/check_quality_toolchain.py
ruff format --check .
ruff check . --output-format concise
bash scripts/typecheck_full.sh
