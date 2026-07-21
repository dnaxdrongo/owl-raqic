#!/usr/bin/env python3
"""Score every outer fold and assemble checksum-bound out-of-fold tables."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json, sha256_file  # noqa: E402
from owl.cadc.config import load_phase4_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_phase4_config(config_path)
    calibration = Path(args.calibration).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipts = []
    candidate_parts = []
    decision_parts = []
    for fold_root in sorted(calibration.glob("outer-*")):
        outer = int(fold_root.name.split("-", 1)[1])
        fold_output = output / fold_root.name
        command = [
            sys.executable,
            str(ROOT / "scripts" / "score_cadc_phase4.py"),
            "--config",
            str(config_path),
            "--dataset",
            str(Path(args.dataset).resolve()),
            "--predictions",
            str(fold_root / "calibrated_predictions.npz"),
            "--calibration-manifest",
            str(fold_root / "calibration_manifest.json"),
            "--output",
            str(fold_output),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"outer fold {outer} scoring failed")
        receipt = json.loads(
            (fold_output / "score_receipt.json").read_text(encoding="utf-8")
        )
        if receipt.get("passed") is not True or receipt.get("outer_fold") != outer:
            raise RuntimeError(f"outer fold {outer} score receipt failed")
        receipts.append(receipt)
        candidate_parts.append(fold_output / "candidate_scores_compact.parquet")
        decision_parts.append(fold_output / "decision_scores_compact.parquet")
    if len(receipts) != config.splits.outer_folds:
        raise RuntimeError("scored artifact fold count does not match configuration")
    candidate_path = output / "candidate_scores_compact.parquet"
    decision_path = output / "decision_scores_compact.parquet"
    if config.runtime.backend == "cupy":
        try:
            import cudf
        except ImportError as exc:
            raise RuntimeError("GPU score aggregation requires cuDF") from exc
        candidate = cudf.read_parquet([str(path) for path in candidate_parts])
        decision = cudf.read_parquet([str(path) for path in decision_parts])
        candidate.to_parquet(candidate_path, compression="zstd", index=False)
        decision.to_parquet(decision_path, compression="zstd", index=False)
        candidate_rows = len(candidate)
        decision_rows = len(decision)
        aggregation_backend = "cudf_cuda"
    else:
        try:
            import pyarrow.dataset as ds
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("score aggregation requires PyArrow") from exc
        candidate = ds.dataset([str(path) for path in candidate_parts]).to_table()
        decision = ds.dataset([str(path) for path in decision_parts]).to_table()
        pq.write_table(candidate, candidate_path, compression="zstd", row_group_size=65536)
        pq.write_table(decision, decision_path, compression="zstd", row_group_size=65536)
        candidate_rows = candidate.num_rows
        decision_rows = decision.num_rows
        aggregation_backend = "pyarrow_cpu"
    if candidate_rows != decision_rows * 22:
        raise ValueError("aggregate score tables violate the 22-candidate contract")
    atomic_json(
        output / "scored_artifacts_receipt.json",
        {
            "schema_version": "owl.cadc.phase4-scored-artifacts.v1",
            "passed": True,
            "outer_folds": len(receipts),
            "candidate_rows": candidate_rows,
            "decision_rows": decision_rows,
            "aggregation_backend": aggregation_backend,
            "candidate_sha256": sha256_file(candidate_path),
            "decision_sha256": sha256_file(decision_path),
            "fold_receipts": receipts,
            "model_spec_sha256": config.model_spec_digest(),
            "phase5_locked": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
