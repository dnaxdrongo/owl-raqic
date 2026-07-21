#!/usr/bin/env bash
set -euo pipefail
ruff check "$@"
ruff format --check "$@"
bash scripts/typecheck_fast.sh
