from __future__ import annotations

from typing import Any

import numpy as np

from owl_raqic.simulation.reference_kraus import simulate_reference_kraus


def run_ensemble(amplitudes: np.ndarray, rounds: int = 1) -> dict[str, Any]:
    return simulate_reference_kraus(amplitudes, rounds=rounds)
