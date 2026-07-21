from __future__ import annotations

from owl.cadc.synthetic import verify_synthetic_contracts


def test_all_registered_synthetic_contracts_pass() -> None:
    receipt = verify_synthetic_contracts()
    assert receipt["passed"] is True
    assert receipt["case_count"] == 15
    assert len(receipt["checks"]) == 15
    assert receipt["learned_model_claims_made"] is False
