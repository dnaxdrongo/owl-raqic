#!/usr/bin/env python3
"""Independently certify a completed multi-seed modeling corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402
from owl.cadc.corpus import CorpusPlan, CorpusUnit, certify_corpus_inventory  # noqa: E402
from owl.cadc.schema import SplitRole  # noqa: E402


def _load_plan(path: Path) -> CorpusPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    units = tuple(
        CorpusUnit(
            unit_id=value["unit_id"],
            seed=int(value["seed"]),
            split_role=SplitRole(value["split_role"]),
            context_family=value["context_family"],
            source_tick=int(value["source_tick"]),
            repeats=int(value["repeats"]),
            horizons=tuple(value["horizons"]),
            maximum_source_decisions=int(value["maximum_source_decisions"]),
            derived_config_path=value["derived_config_path"],
            output_path=value["output_path"],
        )
        for value in payload["units"]
    )
    return CorpusPlan(
        payload["plan_id"],
        payload["phase3_source_sha256"],
        payload["base_config_sha256"],
        payload["config_sha256"],
        units,
        tuple(payload["sealed_phase5_seeds"]),
        tuple(payload["sealed_phase6_seeds"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = _load_plan(Path(args.plan))
    config = load_phase4_config(args.config)
    if plan.config_sha256 != config.corpus_digest():
        raise RuntimeError("corpus plan does not match the scientific corpus contract")
    inventories = []
    for unit in plan.units:
        path = Path(unit.output_path) / "corpus_unit_inventory.json"
        if path.is_file():
            inventories.append(json.loads(path.read_text(encoding="utf-8")))
    certificate = certify_corpus_inventory(
        plan,
        inventories,
        minimum_seeds=config.corpus.minimum_independent_seeds,
        minimum_source_decisions=config.corpus.minimum_source_decisions,
    )
    atomic_json(args.output, certificate)
    return 0 if certificate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
