#!/usr/bin/env python3
"""Run the CADC-MORE 2 symbolic reference and numerical math contracts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.cadc.artifacts import atomic_json  # noqa: E402
from owl.cadc.sympy_contracts import verify_math_contracts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbolic-script", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    script = Path(args.symbolic_script).resolve()
    output = Path(args.output).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    symbolic_output = output.with_name(f"{output.stem}.sympy-reference.json")
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        check=False,
        env={**os.environ, "OWL_PHASE4_SYMPY_OUTPUT": str(symbolic_output)},
    )
    if completed.returncode != 0:
        raise RuntimeError("authoritative Phase 4 SymPy verification failed")
    result = verify_math_contracts(symbolic_output)
    result["symbolic_reference"] = str(symbolic_output)
    atomic_json(output, result)
    print(json.dumps({"passed": result["passed"], "output": str(output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
