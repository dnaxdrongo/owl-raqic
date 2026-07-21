from pathlib import Path

from scripts.audit_v08_workflow import audit


def test_v08_workflow_audit_has_no_syntax_errors():
    result = audit(Path(".").resolve())
    assert result["syntax_errors"] == []
    assert result["summary"]["modules"] > 100
    assert result["summary"]["functions"] > 500


def test_direct_scripts_with_owl_imports_are_bootstrapped():
    result = audit(Path(".").resolve())
    assert all(item["bootstrapped"] for item in result["direct_scripts"])
