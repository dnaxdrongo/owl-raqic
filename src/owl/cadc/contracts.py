"""Provenance, cardinality, and foreign-key contracts for CADC-MORE 2."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from owl.cadc.artifacts import sha256_file
from owl.cadc.schema import (
    EXPECTED_COUNTERFACTUAL_DIGEST,
    EXPECTED_FACTUAL_V2_DIGEST,
    EXPECTED_PHASE3_CLASSIFICATION,
    EXPECTED_PHASE3_SOURCE_SHA256,
    EXPECTED_RNG_REGISTRY_DIGEST,
)
from owl.core.actions import Action


@dataclass(frozen=True)
class DatasetProvenance:
    """Store certificate-bound source, schema, RNG, and CADC-MORE 2 registry identities."""
    phase3_source_sha256: str
    phase25_source_sha256: str
    phase25_certificate_sha256: str
    phase3_certificate_sha256: str
    factual_v2_digest: str
    counterfactual_schema_digest: str
    rng_registry_digest: str
    phase4_dataset_schema_digest: str = "pending"
    feature_schema_digest: str = "pending"
    outcome_registry_digest: str = "pending"
    split_registry_digest: str = "pending"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible provenance mapping."""
        return asdict(self)


@dataclass(frozen=True)
class ContractGate:
    """Named validation result with a reviewer-readable detail."""
    name: str
    passed: bool
    detail: str


def validate_source_identity(
    certificate: Mapping[str, Any],
    *,
    expected_source: str = EXPECTED_PHASE3_SOURCE_SHA256,
    expected_classification: str = EXPECTED_PHASE3_CLASSIFICATION,
) -> DatasetProvenance:
    """Validate the exact counterfactual certificate chain and return its provenance."""
    failures: list[str] = []
    checks = {
        "passed": certificate.get("passed") is True,
        "classification": certificate.get("classification") == expected_classification,
        "phase4_unlocked": certificate.get("phase4_unlocked") is True,
        "failures": certificate.get("failures") == [],
        "phase3_source": certificate.get("phase3_source_sha256") == expected_source,
        "factual_v2": certificate.get("factual_v2_digest") == EXPECTED_FACTUAL_V2_DIGEST,
        "counterfactual": (
            certificate.get("counterfactual_schema_digest") == EXPECTED_COUNTERFACTUAL_DIGEST
        ),
        "rng": certificate.get("rng_registry_digest") == EXPECTED_RNG_REGISTRY_DIGEST,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(f"Phase 3 source/certificate gate failed: {', '.join(failures)}")
    return DatasetProvenance(
        phase3_source_sha256=str(certificate["phase3_source_sha256"]),
        phase25_source_sha256=str(certificate["phase25_source_sha256"]),
        phase25_certificate_sha256=str(certificate["phase25_certificate_sha256"]),
        phase3_certificate_sha256="unbound",
        factual_v2_digest=str(certificate["factual_v2_digest"]),
        counterfactual_schema_digest=str(certificate["counterfactual_schema_digest"]),
        rng_registry_digest=str(certificate["rng_registry_digest"]),
    )


def load_and_validate_certificate(path: str | Path) -> DatasetProvenance:
    """Load one counterfactual certificate and bind its checksum to provenance."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Phase 3 certificate must be a JSON object")
    provenance = validate_source_identity(payload)
    values = provenance.to_dict()
    values["phase3_certificate_sha256"] = sha256_file(source)
    return DatasetProvenance(**values)


class TableContractValidator:
    """Validate normalized columnar table keys without owning table storage."""

    @staticmethod
    def unique(keys: Iterable[Any], *, table: str) -> None:
        """Require exactly one row for every supplied key."""
        counts = Counter(keys)
        duplicates = [key for key, count in counts.items() if count != 1]
        if duplicates:
            raise RuntimeError(f"{table} contains {len(duplicates)} duplicate keys")

    @staticmethod
    def exact_children(
        parent_keys: Sequence[Any],
        child_parent_keys: Iterable[Any],
        *,
        expected: int,
        table: str,
    ) -> None:
        """Require a fixed child cardinality for every parent and no orphans."""
        counts = Counter(child_parent_keys)
        failures = [key for key in parent_keys if counts.get(key, 0) != expected]
        extras = set(counts).difference(parent_keys)
        if failures or extras:
            raise RuntimeError(
                f"{table} cardinality failed: wrong={len(failures)} orphan={len(extras)} "
                f"expected={expected}"
            )

    @staticmethod
    def foreign_keys(
        values: Iterable[Any],
        parents: Iterable[Any],
        *,
        table: str,
        parent_table: str,
    ) -> None:
        """Require every foreign-key value to exist in its parent table."""
        parent_set = set(parents)
        missing = set(values).difference(parent_set)
        if missing:
            raise RuntimeError(
                f"{table} has {len(missing)} keys missing from {parent_table}"
            )

    @staticmethod
    def action_axis(values: Iterable[int]) -> None:
        """Reject action indices outside the immutable 22-action axis."""
        expected = set(range(len(Action)))
        actual = {int(value) for value in values}
        invalid = actual.difference(expected)
        if invalid:
            raise RuntimeError(f"action values outside immutable axis: {sorted(invalid)}")
