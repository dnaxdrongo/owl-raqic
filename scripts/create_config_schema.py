# ruff: noqa: E402 -- approved source-tree bootstrap or optional import gate
"""Export Pydantic JSON schemas for configs and manifests."""

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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from owl.core.config import save_config_schema
from owl.core.contracts import export_manifest_json_schema


def export_schemas(out_dir: str | Path = "schemas") -> None:
    """Export JSON schemas for SimulationConfig and ManifestContract."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_schema(output_dir / "simulation_config.schema.json")
    export_manifest_json_schema(output_dir / "function_manifest.schema.json")


def main() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Export Observer-Window Life JSON schemas.")
    parser.add_argument("out_dir", nargs="?", default="schemas", help="Output directory.")
    args = parser.parse_args()
    export_schemas(args.out_dir)
    print(f"Schemas exported to {Path(args.out_dir)}")


if __name__ == "__main__":
    main()
