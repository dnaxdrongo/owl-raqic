from __future__ import annotations

import importlib.util
from pathlib import Path


def test_docs_checker_reports_repository_consistently(tmp_path: Path):
    path = Path("scripts/check_docs_against_repo.py")
    spec = importlib.util.spec_from_file_location("docs_checker", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    result = mod.check(Path(".").resolve())
    assert isinstance(result["references"], list)
    assert isinstance(result["missing"], list)
