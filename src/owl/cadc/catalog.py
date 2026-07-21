"""Schema-aware discovery of factual v2 and counterfactual v1 evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from owl.cadc.artifacts import atomic_json, sha256_file
from owl.cadc.contracts import DatasetProvenance, TableContractValidator
from owl.cadc.schema import (
    EXPECTED_COUNTERFACTUAL_DIGEST,
    EXPECTED_FACTUAL_V2_DIGEST,
    EXPECTED_PHASE3_SOURCE_SHA256,
    EXPECTED_RNG_REGISTRY_DIGEST,
    stable_id,
)

COUNTERFACTUAL_TABLES = (
    "source_states",
    "source_decisions",
    "branch_attempts",
    "counterfactual_micro_rollouts",
    "branch_events",
    "branch_event_summaries",
    "branch_contributions",
    "candidate_pairs",
    "nonexecutable_candidates",
)

FACTUAL_TABLES = (
    "decisions",
    "agent_context",
    "oracle_context",
    "candidates",
    "action_directions",
    "execution",
    "events",
    "contributions",
    "information",
    "information_followups",
)


@dataclass(frozen=True)
class PartReference:
    """One checksum-verified counterfactual Parquet part."""
    table_name: str
    path: str
    sha256: str
    rows: int
    metadata: dict[str, str]


@dataclass(frozen=True)
class FactualV2Catalog:
    """Discovered factual-v2 schema and table paths for one run."""
    run_root: Path
    table_paths: dict[str, Path]
    schema_path: Path
    schema_digest: str
    source_sha256: str

    @classmethod
    def discover(cls, root: str | Path) -> FactualV2Catalog:
        """Discover exactly one factual-v2 bundle and validate required tables."""
        run_root = Path(root).resolve()
        schemas = sorted(run_root.rglob("schema/cadc_factual_v2.json"))
        if len(schemas) != 1:
            raise RuntimeError(
                f"expected one factual v2 schema below {run_root}, found {len(schemas)}"
            )
        schema_path = schemas[0]
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
        if payload.get("schema_digest") != EXPECTED_FACTUAL_V2_DIGEST:
            raise RuntimeError(f"factual v2 digest mismatch: {schema_path}")
        analysis = schema_path.parent.parent / "analysis" / "cadc_v2"
        table_paths = {
            name: analysis / f"{name}.parquet"
            for name in FACTUAL_TABLES
            if (analysis / f"{name}.parquet").exists()
        }
        required = set(FACTUAL_TABLES[:8])
        missing = required.difference(table_paths)
        if missing:
            raise FileNotFoundError(f"factual v2 tables missing: {sorted(missing)}")
        return cls(
            run_root=run_root,
            table_paths=table_paths,
            schema_path=schema_path,
            schema_digest=str(payload["schema_digest"]),
            source_sha256=str(payload.get("source_sha256", "unknown")),
        )


@dataclass(frozen=True)
class CounterfactualV1Catalog:
    """Discovered counterfactual-v1 manifest with verified part receipts."""
    root: Path
    manifest_path: Path
    parts: tuple[PartReference, ...]
    source_sha256: str
    phase25_certificate_sha256: str
    factual_v2_digest: str
    rng_registry_digest: str

    @classmethod
    def discover(cls, root: str | Path) -> CounterfactualV1Catalog:
        """Discover and checksum-validate one complete counterfactual bundle."""
        value = Path(root).resolve()
        manifests = (
            [value / "counterfactual_manifest.json"]
            if (value / "counterfactual_manifest.json").is_file()
            else sorted(value.rglob("counterfactual_manifest.json"))
        )
        if len(manifests) != 1:
            raise RuntimeError(
                f"expected one counterfactual manifest below {value}, found {len(manifests)}"
            )
        manifest_path = manifests[0]
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_digest": EXPECTED_COUNTERFACTUAL_DIGEST,
            "source_sha256": EXPECTED_PHASE3_SOURCE_SHA256,
            "factual_v2_digest": EXPECTED_FACTUAL_V2_DIGEST,
        }
        failures = [
            key
            for key, expected_value in expected.items()
            if payload.get(key) != expected_value
        ]
        rng = payload.get("rng_registry", {}).get("registry_digest")
        if rng != EXPECTED_RNG_REGISTRY_DIGEST:
            failures.append("rng_registry_digest")
        if failures:
            raise RuntimeError(f"counterfactual manifest identity failed: {failures}")
        parts: list[PartReference] = []
        for raw in payload.get("parts", []):
            table_name = str(raw["table_name"])
            part_path = manifest_path.parent / table_name / str(raw["path"])
            if not part_path.is_file():
                raise FileNotFoundError(part_path)
            if sha256_file(part_path) != raw["sha256"]:
                raise RuntimeError(f"counterfactual part checksum mismatch: {part_path}")
            parts.append(
                PartReference(
                    table_name=table_name,
                    path=str(part_path),
                    sha256=str(raw["sha256"]),
                    rows=int(raw["rows"]),
                    metadata={
                        "schema_digest": str(raw["schema_digest"]),
                        "source_sha256": str(raw["source_sha256"]),
                        "phase25_certificate_sha256": str(
                            raw["phase25_certificate_sha256"]
                        ),
                        "factual_v2_digest": str(raw["factual_v2_digest"]),
                        "rng_registry_digest": str(raw["rng_registry_digest"]),
                    },
                )
            )
        found = {part.table_name for part in parts}
        if found != set(COUNTERFACTUAL_TABLES):
            missing = sorted(set(COUNTERFACTUAL_TABLES) - found)
            raise RuntimeError(
                f"counterfactual table set mismatch: missing={missing}"
            )
        return cls(
            root=manifest_path.parent,
            manifest_path=manifest_path,
            parts=tuple(parts),
            source_sha256=str(payload["source_sha256"]),
            phase25_certificate_sha256=str(payload["phase25_certificate_sha256"]),
            factual_v2_digest=str(payload["factual_v2_digest"]),
            rng_registry_digest=str(rng),
        )

    def parts_for(self, table_name: str) -> tuple[PartReference, ...]:
        """Return all registered parts for one exact table name."""
        return tuple(part for part in self.parts if part.table_name == table_name)


@dataclass(frozen=True)
class Phase4EvidenceCatalog:
    """Paired factual/counterfactual runs bound to one certified provenance."""
    factual: tuple[FactualV2Catalog, ...]
    counterfactual: tuple[CounterfactualV1Catalog, ...]
    provenance: DatasetProvenance
    catalog_id: str

    @classmethod
    def build(
        cls,
        factual_roots: tuple[str, ...],
        counterfactual_roots: tuple[str, ...],
        provenance: DatasetProvenance,
    ) -> Phase4EvidenceCatalog:
        """Build a paired catalog and reject mixed source/schema/RNG identities."""
        if not factual_roots or not counterfactual_roots:
            raise ValueError("factual and counterfactual roots are required")
        factual = tuple(FactualV2Catalog.discover(root) for root in factual_roots)
        counterfactual = tuple(
            CounterfactualV1Catalog.discover(root) for root in counterfactual_roots
        )
        sources = {item.source_sha256 for item in counterfactual}
        factual_digests = {item.factual_v2_digest for item in counterfactual}
        rng_digests = {item.rng_registry_digest for item in counterfactual}
        factual_sources = {item.source_sha256 for item in factual}
        if sources != {provenance.phase3_source_sha256}:
            raise RuntimeError(f"mixed Phase 3 source scopes: {sorted(sources)}")
        if factual_digests != {provenance.factual_v2_digest}:
            raise RuntimeError("mixed factual v2 digests")
        if rng_digests != {provenance.rng_registry_digest}:
            raise RuntimeError("mixed RNG registry digests")
        if factual_sources.difference(
            {provenance.phase3_source_sha256, "unknown"}
        ):
            raise RuntimeError(f"mixed factual source scopes: {sorted(factual_sources)}")
        catalog_id = stable_id(
            "evidence_catalog",
            provenance.to_dict(),
            [sha256_file(item.schema_path) for item in factual],
            [sha256_file(item.manifest_path) for item in counterfactual],
        )
        return cls(factual, counterfactual, provenance, catalog_id)

    def with_phase4_digests(
        self,
        *,
        dataset: str,
        features: str,
        outcomes: str,
        splits: str,
    ) -> Phase4EvidenceCatalog:
        """Return a copy bound to concrete CADC-MORE 2 registry digests."""
        return replace(
            self,
            provenance=replace(
                self.provenance,
                phase4_dataset_schema_digest=dataset,
                feature_schema_digest=features,
                outcome_registry_digest=outcomes,
                split_registry_digest=splits,
            ),
        )

    def write_receipt(self, path: str | Path) -> None:
        """Persist catalog provenance and all discovered part references."""
        atomic_json(
            path,
            {
                "schema_version": "owl.cadc.phase4-catalog-receipt.v1",
                "catalog_id": self.catalog_id,
                "provenance": self.provenance.to_dict(),
                "factual": [
                    {
                        "root": str(item.run_root),
                        "schema": str(item.schema_path),
                        "tables": {name: str(value) for name, value in item.table_paths.items()},
                    }
                    for item in self.factual
                ],
                "counterfactual": [
                    {
                        "root": str(item.root),
                        "manifest_sha256": sha256_file(item.manifest_path),
                        "parts": [asdict(part) for part in item.parts],
                    }
                    for item in self.counterfactual
                ],
                "passed": True,
            },
        )


def hash_context_row(columns: dict[str, Any], index: int) -> str:
    """Reproduce the counterfactual context hash for a NumPy-like column mapping."""
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(columns):
        value = np.ascontiguousarray(columns[name][index])
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name.encode())
        digest.update(len(value.dtype.str).to_bytes(8, "big"))
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def validate_factual_cardinalities(
    decision_keys: list[Any],
    candidate_parent_keys: list[Any],
    direction_parent_keys: list[Any],
) -> None:
    """Validate decision uniqueness and exact candidate/direction child counts."""
    validator = TableContractValidator()
    validator.unique(decision_keys, table="decisions")
    validator.exact_children(
        decision_keys, candidate_parent_keys, expected=22, table="candidates"
    )
    validator.exact_children(
        decision_keys, direction_parent_keys, expected=16, table="action_directions"
    )
