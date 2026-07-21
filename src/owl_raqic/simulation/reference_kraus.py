from __future__ import annotations

from typing import Any

import numpy as np

from owl_raqic.math.instruments import (
    feedback_unitaries,
    outcome_probabilities_from_kraus,
    preparation_kraus_from_amplitudes,
    recursive_channel,
)
from owl_raqic.math.states import density_from_state, ket0


def simulate_reference_kraus(amplitudes: np.ndarray, rounds: int = 1) -> dict[str, Any]:
    kraus, Uprep, projectors = preparation_kraus_from_amplitudes(amplitudes)
    feedback = feedback_unitaries(len(amplitudes))
    rho = density_from_state(ket0(len(amplitudes)))
    probs = []
    for _ in range(rounds):
        probs.append(outcome_probabilities_from_kraus(kraus, rho))
        rho = recursive_channel(kraus, feedback, rho)
        rho = (rho + rho.conjugate().T) / 2
    return {"rho": rho, "probabilities": probs, "kraus": kraus, "feedback": feedback}
