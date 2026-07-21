from __future__ import annotations

import numpy as np

from owl.core.actions import Action
from owl_raqic.types import RAQICActionSet


def owl_action_names() -> tuple[str, ...]:
    return tuple(a.name for a in Action)


def assert_action_basis_compatible(action_set: RAQICActionSet) -> None:
    names = tuple(str(n).upper() for n in action_set.names)
    expected = owl_action_names()
    if names != expected:
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(
            f"RAQIC action set must match OWL Action enum. missing={missing}, extra={extra}"
        )


def map_raqic_probs_to_owl_probs(
    probabilities: np.ndarray, action_set: RAQICActionSet
) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    out = np.zeros((len(Action),), dtype=float)
    for i, name in enumerate(action_set.names):
        key = str(name).upper()
        if key not in Action.__members__:
            raise ValueError(f"RAQIC action {name!r} has no OWL Action mapping")
        if i < probs.size:
            out[int(Action[key])] += float(probs[i])
    s = out.sum()
    if s <= 0 or not np.isfinite(s):
        out[:] = 0.0
        out[int(Action.REST)] = 1.0
    else:
        out /= s
    return out.astype(np.float32)
