from __future__ import annotations

from typing import Any

import numpy as np

from owl_raqic.math.instruments import (
    feedback_unitaries,
    outcome_probabilities_from_kraus,
    post_measurement_state,
    preparation_kraus_from_amplitudes,
)


def sample_trajectory(amplitudes: np.ndarray, rounds: int, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    kraus, _, _ = preparation_kraus_from_amplitudes(amplitudes)
    feedback = feedback_unitaries(len(amplitudes))
    rho = np.zeros((len(amplitudes), len(amplitudes)), dtype=complex)
    rho[0, 0] = 1
    records = []
    for _ in range(rounds):
        probs = outcome_probabilities_from_kraus(kraus, rho)
        y = int(rng.choice(len(probs), p=probs / probs.sum()))
        rho = post_measurement_state(kraus[y], rho, probs[y])
        rho = feedback[y] @ rho @ feedback[y].conjugate().T
        records.append({"outcome": y, "probability": float(probs[y])})
    return {"rho": rho, "records": records}
