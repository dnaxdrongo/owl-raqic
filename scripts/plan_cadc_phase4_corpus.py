#!/usr/bin/env python3
"""Create an immutable multi-seed modeling-corpus plan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.corpus import build_corpus_plan, write_corpus_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = build_corpus_plan(config, output_root=output)
    write_corpus_plan(plan, output / "corpus_plan.json")
    print(output / "corpus_plan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

