# ruff: noqa: E402
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from owl.experiments.controller import main

if __name__ == "__main__":
    raise SystemExit(main())
