from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path


def _load_script(name: str):
    path = Path("scripts") / name
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_safe_collection_excludes_secrets_and_redacts(tmp_path: Path):
    mod = _load_script("safe_collect_artifacts.py")
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    (repo / "reports" / "ok.log").write_text("api_key=abc123\nhello\n")
    (repo / "reports" / "id_ed25519").write_text("PRIVATE")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "x.txt").write_text("bad")
    out = tmp_path / "a.zip"
    mod.collect(repo, out, ("reports",), 1024 * 1024)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        text = zf.read("reports/ok.log").decode()
    assert names == ["reports/ok.log"]
    assert "abc123" not in text and "<REDACTED>" in text
