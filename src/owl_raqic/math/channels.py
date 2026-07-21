from __future__ import annotations

from typing import Any

import numpy as np

from .instruments import recursive_channel


def channel_from_kraus_feedback(kraus_ops: Any, feedback_ops: Any) -> Any:
    def _channel(rho: Any) -> Any:
        return recursive_channel(kraus_ops, feedback_ops, rho)

    return _channel


def dephasing_channel(rho: np.ndarray, p: float) -> np.ndarray:
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0,1]")
    D = np.diag(np.diag(rho))
    return (1 - p) * rho + p * D
