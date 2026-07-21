"""Reproducible RAQIC quantum-experiment export bundles.

The export is a model artifact. It does not claim execution on quantum hardware.
QPY/OpenQASM are written only when the installed Qiskit version supports them.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QuantumExportRecord:
    tick: int
    cell_id: int
    yx: tuple[int, int]
    circuit_kind: str
    action_names: tuple[str, ...]
    active_primes: tuple[int, ...]
    mask: tuple[bool, ...]
    expected_probabilities: tuple[float, ...]
    qiskit_probabilities: tuple[float, ...] | None
    metadata: Mapping[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_circuit_artifacts(directory: Path, circuit: Any) -> dict[str, str]:
    written: dict[str, str] = {}
    if circuit is None:
        return written
    try:
        from qiskit import qpy

        qpy_path = directory / "circuit.qpy"
        with qpy_path.open("wb") as handle:
            qpy.dump(circuit, handle)
        written["qpy"] = qpy_path.name
    except Exception as exc:
        written["qpy_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from qiskit import qasm3

        text = qasm3.dumps(circuit)
        qasm_path = directory / "circuit.qasm"
        qasm_path.write_text(text, encoding="utf-8")
        written["openqasm3"] = qasm_path.name
    except Exception as exc:
        written["qasm_error"] = f"{type(exc).__name__}: {exc}"
    return written


def export_quantum_experiment(
    output_dir: str | Path,
    record: QuantumExportRecord,
    *,
    circuit: Any | None = None,
    parameters: Mapping[str, Any] | None = None,
    validation_tolerances: Mapping[str, float] | None = None,
    make_zip: bool = True,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    circuit_files = _write_circuit_artifacts(out, circuit)
    payload = _json_ready(asdict(record))
    payload["parameters"] = _json_ready(parameters or {})
    payload["validation_tolerances"] = _json_ready(validation_tolerances or {})
    payload["circuit_files"] = circuit_files
    payload["claim_boundary"] = (
        "Circuit representation export only; not evidence of quantum-hardware execution."
    )
    (out / "experiment.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256((out / "experiment.json").read_bytes()).hexdigest()
    (out / "SHA256SUMS.txt").write_text(f"{digest}  experiment.json\n", encoding="utf-8")
    readme = [
        "# RAQIC Quantum Experiment Export",
        "",
        f"- tick: {record.tick}",
        f"- cell: {record.cell_id} at {record.yx}",
        f"- circuit kind: {record.circuit_kind}",
        "- status: model/circuit artifact, not hardware evidence",
        "",
        "The dense probability vector is the recovery target. Qiskit artifacts are",
        "included only when serialization support is present.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    if not make_zip:
        return out
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(out.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(out))
    return archive
