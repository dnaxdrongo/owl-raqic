from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def write_markdown_report(
    path: str | Path,
    title: str,
    sections: Mapping[str, str],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = [f"# {title}", ""]
    for name, body in sections.items():
        text.extend([f"## {name}", "", body.strip(), ""])
    output.write_text("\n".join(text), encoding="utf-8")
    return output
