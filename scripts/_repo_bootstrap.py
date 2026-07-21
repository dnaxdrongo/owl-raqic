"""Local-repository import bootstrap for direct helper scripts.

This is deliberately limited to adding ``<repo>/src`` when the script is
executed from an unpacked source tree. Installed-package behavior is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo(start: str | Path | None = None) -> Path | None:
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    candidates = [here.parent, *here.parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "owl").is_dir():
            src = str(candidate / "src")
            if src not in sys.path:
                sys.path.insert(0, src)
            return candidate
    return None
