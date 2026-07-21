#!/usr/bin/env python3
# ruff: noqa: E402 -- approved source-tree bootstrap or optional import gate
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _repo_bootstrap import bootstrap_repo

repo = bootstrap_repo()
from owl.core.config import load_config
from owl.gpu.run_manifest import build_run_manifest


def main() -> Any:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default="reports/run_manifest.json")
    ap.add_argument("--fallback-count", type=int, default=0)
    ap.add_argument("--certification")
    args = ap.parse_args()
    cfg = load_config(args.config)
    manifest = build_run_manifest(
        repo or Path.cwd(),
        args.config,
        seed=cfg.world.seed,
        precision=cfg.raqic.full_gpu_precision,
        run_class=cfg.raqic.full_gpu_run_class,
        all_cell_semantics=bool(getattr(cfg.raqic, "gpu_all_cells_required", True)),
        fallback_count=args.fallback_count,
        certification_path=args.certification,
    )
    print(manifest.write(args.out))


if __name__ == "__main__":
    main()
