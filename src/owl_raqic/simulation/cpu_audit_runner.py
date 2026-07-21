from __future__ import annotations

from typing import Any

import numpy as np

from owl_raqic.math.channels import channel_from_kraus_feedback
from owl_raqic.math.checks import (
    check_density_matrix,
    check_kraus_completeness,
    check_trace_preservation,
)
from owl_raqic.math.instruments import feedback_unitaries, preparation_kraus_from_amplitudes
from owl_raqic.math.states import density_from_state, ket0


def run_cpu_audit(amplitudes: np.ndarray) -> dict[str, Any]:
    kraus, Uprep, projectors = preparation_kraus_from_amplitudes(amplitudes)
    feedback = feedback_unitaries(len(amplitudes))
    rho0 = density_from_state(ket0(len(amplitudes)))
    channel = channel_from_kraus_feedback(kraus, feedback)
    rho1 = channel(rho0)
    return {
        "kraus": check_kraus_completeness(kraus),
        "trace": check_trace_preservation(channel, rho0),
        "density": check_density_matrix(rho1),
    }
