# ruff: noqa: E402 -- approved source-tree bootstrap or optional import gate
"""Validate the function manifest against Python signatures."""

from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_scripts_dir = _BootstrapPath(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from _repo_bootstrap import bootstrap_repo

bootstrap_repo()


import sys
from pathlib import Path

# Allow direct execution from a source checkout without editable installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from owl.core.contracts import load_function_manifest, validate_manifest_against_signatures


def validate_manifest(manifest_path: str | Path = "manifests/function_manifest.yaml") -> list[str]:
    """Validate function manifest entries against source-code signatures."""
    manifest = load_function_manifest(manifest_path)
    return validate_manifest_against_signatures(manifest)


def main() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Observer-Window Life function manifest.")
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default="manifests/function_manifest.yaml",
        help="Path to function manifest.",
    )
    args = parser.parse_args()

    errors = validate_manifest(args.manifest_path)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    print("Manifest validation passed.")


if __name__ == "__main__":
    main()
