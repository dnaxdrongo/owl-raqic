#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m mypy --explicit-package-bases \
  src/owl/types src/owl/core src/owl/science src/owl/runtime \
  src/owl/gpu/backend.py src/owl/gpu/device_state.py src/owl/gpu/transfer_ledger.py
