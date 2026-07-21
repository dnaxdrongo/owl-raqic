from __future__ import annotations

from scripts.audit_cadc_phase3_hotpaths import audit


def test_hotpath_audit_accepts_only_declared_transfer_boundaries() -> None:
    result = audit()
    assert result["passed"], result["failures"]
    assert result["documented_transfer_boundaries"]
