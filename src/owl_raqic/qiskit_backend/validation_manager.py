from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from owl.core.actions import Action
from owl.gpu.qiskit_transfer import pack_qiskit_rows, unpack_qiskit_rows
from owl.gpu.transfer_ledger import TransferKind, TransferLedger
from owl_raqic.validation.circuit_matrix import validate_circuit_matrix
from owl_raqic.validation.device_selection import select_validation_rows_device


@dataclass
class QiskitValidationManager:
    """Scheduled RAQIC circuit-equivalence validator.

    Production decisions remain dense GPU RAQIC unless an explicit per-OW
    Qiskit decision policy is selected. This manager selects rows on device and
    copies only the bounded validation slab.
    """

    output_dir: Path
    cadence: int = 0
    strict_gpu: bool = True
    allow_cpu_fallback: bool = False
    method: str = "statevector"
    limit: int = 16
    fraction: float = 0.0
    tolerance: float = 1e-8
    kl_tolerance: float = 1e-7
    max_qubits: int = 28
    families: tuple[str, ...] = ("static",)
    authoritative_family: str = "static"
    shots: int = 4096
    simulator_options: dict[str, Any] = field(default_factory=dict)
    reports: list[dict[str, Any]] = field(default_factory=list)
    transfer_ledger: TransferLedger | None = None

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        output_dir: str | Path = "reports/qiskit_validation",
        *,
        transfer_ledger: TransferLedger | None = None,
    ) -> QiskitValidationManager:
        opts = {
            "batched_shots_gpu": bool(getattr(cfg.raqic, "qiskit_batched_shots_gpu", False)),
            "cuStateVec_enable": bool(getattr(cfg.raqic, "qiskit_enable_cuStateVec", False)),
            "runtime_parameter_bind_enable": bool(
                getattr(cfg.raqic, "qiskit_runtime_parameter_bind_enable", False)
            ),
            "shot_branching_enable": bool(
                getattr(cfg.raqic, "qiskit_shot_branching_enable", False)
            ),
            "precision": "double"
            if getattr(cfg.raqic, "full_gpu_precision", "audit64") == "audit64"
            else "single",
            "seed_simulator": int(cfg.world.seed),
        }
        families = tuple(str(x) for x in getattr(cfg.raqic, "qiskit_circuit_families", ("static",)))
        return cls(
            output_dir=Path(output_dir),
            cadence=int(getattr(cfg.raqic, "full_gpu_validation_every", 0)),
            strict_gpu=bool(getattr(cfg.raqic, "full_gpu_qiskit_strict", True)),
            allow_cpu_fallback=bool(
                getattr(cfg.raqic, "full_gpu_qiskit_allow_cpu_fallback", False)
            ),
            method=str(getattr(cfg.raqic, "qiskit_gpu_method", "statevector")),
            limit=int(
                min(
                    int(getattr(cfg.raqic, "gpu_audit_limit", 16) or 16),
                    int(getattr(cfg.raqic, "qiskit_debug_ow_limit", 0) or 10**9),
                )
                if int(getattr(cfg.raqic, "qiskit_debug_ow_limit", 0) or 0)
                else int(getattr(cfg.raqic, "gpu_audit_limit", 16) or 16)
            ),
            fraction=max(
                float(getattr(cfg.raqic, "qiskit_subset_fraction", 0.0)),
                float(getattr(cfg.raqic, "gpu_audit_fraction", 0.0)),
                float(getattr(cfg.raqic, "full_gpu_audit_fraction", 0.0)),
            ),
            tolerance=float(getattr(cfg.raqic, "gpu_probability_tolerance", 1e-8)),
            kl_tolerance=float(getattr(cfg.raqic, "gpu_kl_tolerance", 1e-7)),
            max_qubits=int(getattr(cfg.raqic, "qiskit_validation_max_qubits", 28)),
            families=families,
            authoritative_family=str(
                getattr(cfg.raqic, "qiskit_authoritative_family", families[0])
            ),
            shots=int(getattr(cfg.raqic, "qiskit_validation_shots", 4096)),
            simulator_options=opts,
            transfer_ledger=transfer_ledger,
        )

    def due(self, tick: int, *, requested: bool = False) -> bool:
        return bool(requested or (self.cadence > 0 and int(tick) % self.cadence == 0))

    def validate_device_state(
        self, ds: Any, *, tick: int, requested: bool = False
    ) -> dict[str, Any] | None:
        if not self.due(tick, requested=requested):
            return None
        if "raqic_probabilities" not in ds.arrays:
            raise RuntimeError("Qiskit validation requested before RAQIC probabilities exist")

        total_rows = int(ds.health.size)
        fraction_limit = int(np.ceil(self.fraction * total_rows)) if self.fraction > 0 else 0
        selection_limit = max(1, min(total_rows or 1, max(self.limit, fraction_limit)))
        selection = select_validation_rows_device(ds, limit=selection_limit)
        indices = selection.flat_indices
        authoritative_probabilities_all = ds.arrays["raqic_probabilities"]
        use_interference_input = "interference" in self.families
        probabilities_all = (
            ds.arrays.get("raqic_pre_mixer_probabilities", authoritative_probabilities_all)
            if use_interference_input
            else authoritative_probabilities_all
        )
        actions = int(probabilities_all.shape[-1])
        flat_prob = probabilities_all.reshape(-1, actions)
        flat_expected = authoritative_probabilities_all.reshape(-1, actions)
        flat_phase = ds.arrays.get("raqic_phase")
        flat_phase = None if flat_phase is None else flat_phase.reshape(-1, actions)
        authority_all = ds.arrays.get("_authority_bool", ds.arrays.get("authority"))
        flat_authority = None if authority_all is None else authority_all.reshape(-1, actions)
        parent_all = ds.arrays.get("raqic_parent_intention")
        flat_parent = None if parent_all is None else parent_all.reshape(-1, actions)
        occupancy = ds.arrays.get("occupancy")
        if occupancy is None:
            ow_ids_device = indices
        else:
            flat_occ = occupancy.reshape(-1)
            ow_ids_device = flat_occ[indices]

        selected_prob_device = flat_prob[indices]
        selected_phase_device = None if flat_phase is None else flat_phase[indices]
        selected_authority_device = None if flat_authority is None else flat_authority[indices]
        selected_parent_device = None if flat_parent is None else flat_parent[indices]

        packed_device, packed_layout = pack_qiskit_rows(
            ds.xp,
            probabilities=selected_prob_device,
            phases=selected_phase_device,
            authority=selected_authority_device,
            parent=selected_parent_device,
            ow_ids=ow_ids_device,
            flat_indices=indices,
        )
        packed_host = ds.backend.asnumpy(packed_device)
        unpacked = unpack_qiskit_rows(packed_host, packed_layout)
        probabilities = unpacked.probabilities
        phases = unpacked.phases
        authority = unpacked.authority
        parent = unpacked.parent
        ow_ids = unpacked.ow_ids
        if unpacked.flat_indices is None:
            raise RuntimeError("packed Qiskit validation slab omitted selected indices")
        selected_indices = unpacked.flat_indices
        expected_probabilities = ds.backend.asnumpy(flat_expected[indices])
        if self.transfer_ledger is not None:
            self.transfer_ledger.record_d2h(
                packed_layout.total_bytes,
                kind=TransferKind.QISKIT,
                tick=int(tick),
                source_stream="qiskit-validation",
                synchronization="device",
                scheduled=True,
                graph_compatible=False,
                reason="bounded selected-row Qiskit validation slab",
            )

        qubits = max(1, int((actions - 1).bit_length()))
        if qubits > self.max_qubits:
            raise MemoryError(
                f"Qiskit validation requires {qubits} qubits, exceeding configured "
                f"limit {self.max_qubits}"
            )

        transferred_bytes = int(packed_layout.total_bytes)
        started = time.perf_counter()
        report = validate_circuit_matrix(
            probabilities,
            phases,
            authority,
            parent,
            ow_ids=ow_ids,
            families=self.families,
            authoritative_family=self.authoritative_family,
            limit=self.limit,
            method=self.method,
            strict_gpu=self.strict_gpu,
            allow_cpu_fallback=self.allow_cpu_fallback,
            tolerance=self.tolerance,
            kl_tolerance=self.kl_tolerance,
            shots=self.shots,
            simulator_options=self.simulator_options,
            expected_probabilities=expected_probabilities,
            interference_mixer_strength=float(
                getattr(ds.metadata.get("cfg").raqic, "interference_mixer_strength", 0.0)
            ),
            interference_trotter_steps=int(
                getattr(ds.metadata.get("cfg").raqic, "interference_trotter_steps", 1)
            ),
            action_names=tuple(action.name for action in Action),
        )
        payload = report.to_dict()
        payload.update(
            {
                "tick": int(tick),
                "elapsed_seconds": time.perf_counter() - started,
                "selected_flat_indices": selected_indices.astype(int).tolist(),
                "selected_ow_ids": ow_ids.astype(int).tolist(),
                "transferred_bytes": int(transferred_bytes),
                "full_tensor_bytes_avoided": int(
                    probabilities_all.nbytes
                    + (0 if flat_phase is None else flat_phase.nbytes)
                    + (0 if flat_authority is None else flat_authority.nbytes)
                    + (0 if flat_parent is None else flat_parent.nbytes)
                    - transferred_bytes
                ),
            }
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"raqic_quantum_validation_tick_{int(tick):08d}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.reports.append(payload)
        return cast(dict[str, Any] | None, payload)

    def summary(self) -> dict[str, Any]:
        return {
            "runs": len(self.reports),
            "passed": all(r.get("passed", False) for r in self.reports) if self.reports else None,
            "max_abs_error": max((r.get("max_abs_error", 0.0) for r in self.reports), default=0.0),
            "strict_gpu": self.strict_gpu,
            "allow_cpu_fallback": self.allow_cpu_fallback,
            "method": self.method,
            "families": self.families,
            "fraction": self.fraction,
            "kl_tolerance": self.kl_tolerance,
            "transferred_bytes": sum(int(r.get("transferred_bytes", 0)) for r in self.reports),
        }
