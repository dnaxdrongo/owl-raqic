#!/usr/bin/env python3
"""Record the user's explicit response to an advisory runtime estimate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", required=True)
    parser.add_argument(
        "--choice",
        required=True,
        choices=("proceed", "reduce", "stop"),
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    estimate_path = Path(args.estimate).resolve()
    estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    if (
        estimate.get("classification") != "USER_RUNTIME_DECISION_REQUIRED"
        or estimate.get("decision_required") is not True
        or estimate.get("automatic_failure") is not False
    ):
        raise ValueError("runtime estimate does not satisfy the decision contract")
    atomic_json(
        args.output,
        {
            "schema_version": "owl.cadc.phase4-runtime-decision.v1",
            "recorded_at": datetime.now(UTC).isoformat(),
            "plan_id": estimate["plan_id"],
            "estimate_path": str(estimate_path),
            "estimate_sha256": sha256_file(estimate_path),
            "choice": args.choice,
            "continue_selected_profile": args.choice == "proceed",
            "note": args.note,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
