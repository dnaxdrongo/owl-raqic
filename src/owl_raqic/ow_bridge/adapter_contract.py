from __future__ import annotations

import copy
from typing import Any

from owl_raqic.algorithms.raqic_driver import RAQICDecisionEngine
from owl_raqic.types import RAQICFeaturePacket


def decide_without_mutation(engine: RAQICDecisionEngine, packet: RAQICFeaturePacket) -> Any:
    before = copy.deepcopy(packet)
    result = engine.decide(packet)
    assert packet == before
    return result
