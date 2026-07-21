from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QiskitWorkloadEstimate:
    ow_rows: int
    circuit_families: int
    template_groups: int
    parameter_values: int
    simulator_jobs: int
    qubits_max: int
    shots_total: int
    state_memory_bytes_peak: int
    output_bytes_estimate: int

    @property
    def circuit_rows(self) -> int:
        return int(self.ow_rows) * int(self.circuit_families)

    def to_dict(self) -> dict[str, int]:
        data = asdict(self)
        data["circuit_rows"] = self.circuit_rows
        return data


def estimate_qiskit_workload(
    *,
    ow_rows: int,
    action_count: int,
    family_count: int,
    chunk_size: int,
    shots: int,
    exact_family_count: int,
    probability_dtype_bytes: int = 8,
) -> QiskitWorkloadEstimate:
    ow_rows = max(0, int(ow_rows))
    family_count = max(1, int(family_count))
    chunk_size = max(1, int(chunk_size))
    action_count = max(1, int(action_count))
    qubits = max(1, int(math.ceil(math.log2(action_count))))
    jobs = math.ceil(ow_rows / chunk_size) * family_count
    shots_total = ow_rows * max(0, int(shots)) * max(0, family_count - exact_family_count)
    # One complex128 state plus a modest executor multiplier.
    state_memory = (1 << qubits) * 16 * max(1, min(chunk_size, ow_rows or 1)) * 2
    output_bytes = ow_rows * family_count * action_count * probability_dtype_bytes
    return QiskitWorkloadEstimate(
        ow_rows=ow_rows,
        circuit_families=family_count,
        template_groups=ow_rows,  # worst case; exact amplitudes may all differ
        parameter_values=ow_rows * action_count * 2,
        simulator_jobs=jobs,
        qubits_max=qubits,
        shots_total=shots_total,
        state_memory_bytes_peak=state_memory,
        output_bytes_estimate=output_bytes,
    )
