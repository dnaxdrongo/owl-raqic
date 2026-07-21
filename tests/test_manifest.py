"""Manifest and schema tests."""

from __future__ import annotations

import json
from pathlib import Path

from owl.core.config import load_config, save_config_schema
from owl.core.contracts import (
    ManifestContract,
    inspect_python_signature,
    load_function_manifest,
    validate_manifest_against_signatures,
)


def _write_sample_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "function_manifest.yaml"
    manifest_path.write_text(
        """project: Observer-Window Life
version: '1'
functions:
  - id: core.state.field_shape
    module: owl.core.state
    name: field_shape
    status: implemented
    layer: core
    purpose: Return the cell-grid shape.
    parameters:
      - name: state
        type: WorldState
        required: true
    returns: tuple[int, int]
""",
        encoding="utf-8",
    )
    return manifest_path


def test_manifest_loads(tmp_path: Path) -> None:
    manifest = load_function_manifest(_write_sample_manifest(tmp_path))
    assert isinstance(manifest, ManifestContract)
    assert manifest.project == "Observer-Window Life"
    assert manifest.functions


def test_manifest_signatures_validate(tmp_path: Path) -> None:
    manifest = load_function_manifest(_write_sample_manifest(tmp_path))
    errors = validate_manifest_against_signatures(manifest)
    assert errors == []


def test_inspect_python_signature_for_shape_helper() -> None:
    sig = inspect_python_signature("owl.core.state", "field_shape")
    assert sig["name"] == "field_shape"
    assert [p["name"] for p in sig["parameters"]] == ["state"]


def test_save_config_schema(tmp_path: Path) -> None:
    out = tmp_path / "simulation_config.schema.json"
    save_config_schema(out)
    schema = json.loads(out.read_text())
    assert schema["title"] == "SimulationConfig"
    assert "properties" in schema


def test_load_config_missing_file_error_is_clear(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    try:
        load_config(missing)
    except FileNotFoundError as exc:
        assert "configuration file not found" in str(exc)
    else:
        raise AssertionError("missing config should raise FileNotFoundError")


def test_retained_documentation_files_exist() -> None:
    required = [
        Path("README.md"),
        Path("COURSE_USE_NOTICE.md"),
        Path("docs/ARCHITECTURE.md"),
        Path("docs/REFERENCES.md"),
        Path("docs/REFERENCES.json"),
    ]
    for path in required:
        assert path.exists(), path
        assert path.read_text(encoding="utf-8").strip()


def test_engine_import_graph_has_no_presentation_or_analysis_dependencies() -> None:
    import ast

    forbidden_prefixes = ("owl.viz", "owl.record", "owl.analysis", "owl.experiments")
    for source_path in Path("src/owl/engine").glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    assert not name.startswith(forbidden_prefixes), f"{source_path} imports {name}"
            if module is not None:
                assert not module.startswith(forbidden_prefixes), f"{source_path} imports {module}"
