import numpy as np

from owl_raqic import RAQICDecisionEngine, RAQICFeaturePacket
from owl_raqic.ow_bridge.action_mapping import action_index, expected_action_names
from owl_raqic.ow_bridge.adapter_contract import decide_without_mutation
from owl_raqic.reports.audit_report import write_audit_json
from owl_raqic.reports.markdown import write_markdown_report


def test_reports_write(tmp_path):
    j = write_audit_json(tmp_path / "audit.json", {"ok": True})
    m = write_markdown_report(tmp_path / "report.md", "Title", {"Section": "Body"})
    assert j.exists()
    assert m.exists()


def test_action_schema_matches_expected_names():
    names = expected_action_names()
    assert "REST" in names and "INGEST" in names
    assert action_index("REST") == 0


def test_bridge_decide_without_mutation():
    packet = RAQICFeaturePacket(ow_id=5, scale_id=0, tick=1, feature_bins={"resource": 0.5})
    result = decide_without_mutation(RAQICDecisionEngine(), packet)
    assert np.allclose(result.action_probabilities.sum(), 1)
