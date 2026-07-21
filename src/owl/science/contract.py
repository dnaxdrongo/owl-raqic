from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Return stable JSON used by scientific and execution certificates."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_canonical(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScientificContract:
    """Identity of the scientific law, independent of execution strategy."""

    version: str
    stage_order: tuple[str, ...]
    random_contract_version: str
    field_registry_version: str
    action_schema_hash: str
    equation_ledger_hash: str

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return sha256_canonical(self.canonical_dict())


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def current_scientific_contract(root: str | Path | None = None) -> ScientificContract:
    """Build the scientific contract from checked-in ledgers.

    The hashes make an equation/stage/action change invalidate alternate production
    certificates even when the package version was not changed accidentally.
    """
    from owl.core.actions import Action

    from .stage_contract import scientific_stage_order

    base = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    action_payload = [(item.name, int(item)) for item in Action]
    equation_candidates = (
        base / "docs" / "equation_to_code_ledger.md",
        base / "docs" / "v09_hardened" / "OWL_RAQIC_V0_9_PRODUCTION_DOCUMENTATION_ADDENDUM.md",
    )
    equation_hash = next(
        (_hash_file(path) for path in equation_candidates if path.exists()),
        sha256(b"owl-raqic-equation-ledger-v092").hexdigest(),
    )
    return ScientificContract(
        version="OWL-RAQIC-SCIENCE-0.9.2",
        stage_order=scientific_stage_order(),
        random_contract_version="counter-rng-v1",
        field_registry_version="cell-field-policy-v1",
        action_schema_hash=sha256_canonical(action_payload),
        equation_ledger_hash=equation_hash,
    )
