#!/usr/bin/env python3
"""Package the certified hardware-neutral corpus/dataset for H200/B200 reuse."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
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
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_phase4_config(args.config)
    run = Path(args.run).resolve()
    output = Path(args.output).resolve()
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise ValueError("reusable-data package must end in .tar.gz")
    certificate = json.loads(
        (run / "corpus" / "corpus_certificate.json").read_text(encoding="utf-8")
    )
    dataset_receipt = json.loads(
        (run / "dataset" / "dataset_build_receipt.json").read_text(encoding="utf-8")
    )
    repeat = json.loads((run / "repeat_pilot.json").read_text(encoding="utf-8"))
    if not all(
        (
            certificate.get("passed") is True,
            dataset_receipt.get("passed") is True,
            repeat.get("passed") is True,
            certificate.get("corpus_contract_sha256") == config.corpus_digest(),
            dataset_receipt.get("model_spec_sha256") == config.model_spec_digest(),
            repeat.get("model_spec_sha256") == config.model_spec_digest(),
        )
    ):
        raise RuntimeError("reusable-data source run failed scope validation")
    files = [
        run / "corpus" / "corpus_plan.json",
        run / "corpus" / "corpus_certificate.json",
        run / "dataset" / "dataset_build_receipt.json",
        run / "repeat_pilot.json",
        *sorted((run / "dataset" / "manifests").glob("*.json")),
        *sorted((run / "dataset" / "canonical_data").glob("**/*.json")),
        *sorted((run / "dataset" / "canonical_data").glob("**/*.parquet")),
    ]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("reusable-data input is incomplete")
    registered_parts = {str(value["name"]): value for value in dataset_receipt["parts"]}
    for name, receipt in registered_parts.items():
        part = run / "dataset" / "canonical_data" / name / "part-000000.parquet"
        if sha256_file(part) != receipt["sha256"]:
            raise RuntimeError(f"canonical part checksum mismatch before export: {name}")
    with tempfile.TemporaryDirectory(prefix="phase4-reuse-") as temporary_name:
        temporary = Path(temporary_name)
        manifest_path = temporary / "REUSABLE_DATA_MANIFEST.json"
        manifest = {
            "schema_version": "owl.cadc.phase4-reusable-data.v1",
            "passed": True,
            "phase3_source_sha256": certificate["phase3_source_sha256"],
            "corpus_contract_sha256": config.corpus_digest(),
            "model_spec_sha256": config.model_spec_digest(),
            "compatible_targets": ["h100", "h200", "b200"],
            "precision_policy": {
                "h100": "bf16",
                "h200": "bf16",
                "b200": "bf16",
                "b200_fp8": "separate_parity_certificate_required",
            },
            "files": [
                {
                    "path": str(path.relative_to(run)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
            "phase5_locked": True,
        }
        atomic_json(manifest_path, manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                archive.add(path, arcname=str(path.relative_to(run)), recursive=False)
            archive.add(manifest_path, arcname=manifest_path.name, recursive=False)
    sidecar = output.with_name(f"{output.name}.sha256")
    sidecar.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(output)
    print(sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
