#!/usr/bin/env python3
"""Materialize a checksum-stable scientific challenge receipt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.synthetic import verify_synthetic_contracts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    result = verify_synthetic_contracts(
        tie_tolerance=config.evaluation.tie_tolerance
    )
    result["model_spec_sha256"] = config.model_spec_digest()
    atomic_json(args.output, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
