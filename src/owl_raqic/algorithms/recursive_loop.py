from __future__ import annotations

from typing import Any

from owl_raqic.algorithms.raqic_driver import RAQICDecisionEngine
from owl_raqic.types import RAQICFeaturePacket


def run_recursive_packets(
    engine: RAQICDecisionEngine, packets: list[RAQICFeaturePacket], rounds: int = 1
) -> Any:
    out = []
    current = packets
    for _ in range(rounds):
        results = engine.decide_batch(current, sample=True)
        out.append(results)
        current = [
            RAQICFeaturePacket(
                ow_id=p.ow_id,
                scale_id=p.scale_id,
                tick=p.tick + 1,
                feature_bins=p.feature_bins,
                adelic_codes=p.adelic_codes,
                parent_intention=p.parent_intention,
                authority_mask=p.authority_mask,
                metadata={**dict(p.metadata), "previous_result": results[i].measurement_record},
            )
            for i, p in enumerate(current)
        ]
    return out
