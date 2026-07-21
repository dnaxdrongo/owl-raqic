from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from owl_raqic.types import RAQICFeaturePacket


def feature_signature(packet: RAQICFeaturePacket, n_actions: int) -> str:
    payload = {
        "scale_id": packet.scale_id,
        "features": dict(sorted(packet.feature_bins.items())),
        "adelic": dict(sorted(packet.adelic_codes.items())),
        "mask": None
        if packet.authority_mask is None
        else [bool(x) for x in packet.authority_mask[:n_actions]],
        "intention": None
        if packet.parent_intention is None
        else [round(float(x), 8) for x in packet.parent_intention[:n_actions]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def group_by_signature(packets: list[RAQICFeaturePacket], n_actions: int) -> Any:
    groups = defaultdict(list)
    for p in packets:
        groups[feature_signature(p, n_actions)].append(p)
    return dict(groups)
