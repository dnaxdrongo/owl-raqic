"""Function manifest schema and signature-validation utilities.

The manifest is a lightweight contract ledger. It is intentionally stricter than
plain comments but not a replacement for type checking. It verifies that planned
public functions still exist at their declared import paths and that parameter
names have not silently drifted.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ContractBaseModel(BaseModel):
    """Strict base model for manifest schemas."""

    model_config = ConfigDict(extra="forbid")


class ParameterContract(ContractBaseModel):
    """Declared input parameter for a function contract."""

    name: str
    type: str
    required: bool = True
    description: str = ""


class FunctionContract(ContractBaseModel):
    """Declared function contract entry."""

    id: str
    module: str
    name: str
    status: Literal["stub", "implemented", "deprecated"] = "stub"
    layer: str
    purpose: str = ""
    parameters: list[ParameterContract] = Field(default_factory=list)
    returns: str = "None"
    mutates: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)
    called_by: list[str] = Field(default_factory=list)


class ManifestContract(ContractBaseModel):
    """Full function manifest schema."""

    project: str
    version: str
    functions: list[FunctionContract]


def _ensure_repo_paths_for_scripts() -> None:
    """Add common repo import paths when scripts run without editable install."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    src_path = repo_root / "src"
    for candidate in (repo_root, src_path):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _resolve_dotted_object(module_name: str, object_name: str) -> Any:
    """Import ``module_name`` and resolve ``object_name``, including class methods."""
    _ensure_repo_paths_for_scripts()
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        if not hasattr(obj, part):
            raise AttributeError(f"{module_name}.{object_name} missing component {part!r}")
        obj = getattr(obj, part)
    return obj


def load_function_manifest(path: str | Path) -> ManifestContract:
    """Load and validate the YAML/JSON function manifest."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"function manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        raise ValueError(f"function manifest is empty: {manifest_path}")
    if not isinstance(data, dict):
        raise TypeError(f"function manifest root must be a mapping, got {type(data).__name__}")

    return ManifestContract.model_validate(data)


def export_manifest_json_schema(path: str | Path) -> None:
    """Export the JSON schema for :class:`ManifestContract`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(ManifestContract.model_json_schema(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def inspect_python_signature(module_name: str, function_name: str) -> dict[str, Any]:
    """Return a normalized representation of a Python callable signature.

    Parameters named ``self`` or ``cls`` are omitted so instance, class, and
    module-level functions can be compared through the same manifest format.
    """
    obj = _resolve_dotted_object(module_name, function_name)
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"cannot inspect signature for {module_name}.{function_name}: {exc}"
        ) from exc

    parameters: list[dict[str, Any]] = []
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        parameters.append(
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "required": parameter.default is inspect.Signature.empty,
                "annotation": (
                    ""
                    if parameter.annotation is inspect.Signature.empty
                    else str(parameter.annotation)
                ),
            }
        )

    return_annotation = signature.return_annotation
    return {
        "module": module_name,
        "name": function_name,
        "parameters": parameters,
        "returns": "" if return_annotation is inspect.Signature.empty else str(return_annotation),
    }


def validate_manifest_against_signatures(manifest: ManifestContract) -> list[str]:
    """Compare manifest function entries against importable Python callables.

    Validation checks that each module and callable exists and that declared
    parameter names match the callable signature. Entries excluded from validation
    are skipped. Return annotations and optional accelerator types are not treated
    as hard failures because they may vary across supported Python environments.
    """
    errors: list[str] = []

    for function in manifest.functions:
        if function.status == "deprecated":
            continue

        try:
            actual = inspect_python_signature(function.module, function.name)
        except Exception as exc:  # noqa: BLE001 - manifest validator should aggregate failures.
            errors.append(f"{function.id}: cannot import/inspect: {exc}")
            continue

        declared_names = [parameter.name for parameter in function.parameters]
        actual_names = [parameter["name"] for parameter in actual["parameters"]]

        if declared_names != actual_names:
            errors.append(
                f"{function.id}: parameter mismatch; "
                f"manifest={declared_names}, actual={actual_names}"
            )

    return errors


def main() -> None:
    """CLI entry point for manifest validation.

    Exits with status 1 when validation errors are found.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Validate Observer-Window Life function manifest.")
    parser.add_argument(
        "manifest",
        nargs="?",
        default="manifests/function_manifest.yaml",
        help="Path to function manifest YAML/JSON file.",
    )
    args = parser.parse_args()

    manifest = load_function_manifest(args.manifest)
    errors = validate_manifest_against_signatures(manifest)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    print(f"Manifest OK: {len(manifest.functions)} function contracts checked.")
