"""PyInstaller entry point for the standalone OWL replay viewer."""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

from owl.viz.replay_app import main as replay_main


def _choose_bundle() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askdirectory(title="Select an extracted OWL replay bundle")
    if not selected:
        selected = filedialog.askopenfilename(
            title="Or select a packaged OWL experiment ZIP",
            filetypes=[("OWL experiment ZIP", "*.zip"), ("All files", "*.*")],
        )
    root.destroy()
    return selected or None


def _find_bundle(root: Path) -> Path:
    if (root / "run_manifest.json").exists():
        return root
    candidates = sorted(root.rglob("run_manifest.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one replay bundle in {root}; found {len(candidates)} manifests"
        )
    return candidates[0].parent


def _safe_extract(archive: Path, destination: Path) -> Path:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            target = (destination / info.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe ZIP member: {info.filename}")
        handle.extractall(destination)
    return _find_bundle(destination)


def main() -> int:
    arguments = list(sys.argv[1:])
    if not arguments:
        selected = _choose_bundle()
        if selected is None:
            print("Provide a replay bundle directory or experiment ZIP.", file=sys.stderr)
            return 2
        arguments = [selected]
    supplied = Path(arguments[0]).expanduser().resolve()
    if supplied.is_file() and supplied.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="owl-replay-") as temporary:
            try:
                bundle = _safe_extract(supplied, Path(temporary))
            except (ValueError, zipfile.BadZipFile) as exc:
                print(f"Could not open OWL replay ZIP: {exc}", file=sys.stderr)
                return 2
            return replay_main([str(bundle), *arguments[1:]])
    try:
        bundle = _find_bundle(supplied)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return replay_main([str(bundle), *arguments[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
