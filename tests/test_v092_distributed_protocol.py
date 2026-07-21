from __future__ import annotations

import numpy as np

from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.gpu.device_state import OWLDeviceState
from owl.gpu.distributed.halo_protocol import generate_halo_protocol
from owl.gpu.distributed.launch import _certify_collective_ledgers


def _device_state():
    cfg = load_config("configs/gpu_v09_multi_gpu_small.yaml")
    state = initialize_world(cfg, np.random.default_rng(cfg.world.seed))
    ds = OWLDeviceState.from_world_state(state, cfg, force_backend="numpy")
    return ds


def test_halo_protocol_is_generated_from_stage_contract_and_conservative():
    ds = _device_state()
    protocol = generate_halo_protocol(ds)
    assert protocol.halo_width >= 1
    assert protocol.conservative_expansion
    for required in ("health", "occupancy", "parent_id", "phase", "signal"):
        assert required in protocol.fields
    assert set(protocol.fields).issubset(ds.arrays)


def test_distributed_certificate_distinguishes_equal_size_field_payloads():
    base = {
        "success": True,
        "halo_stats": {"boundary_checks": 1, "boundary_elements": 4},
    }
    rank0 = dict(
        base,
        rank=0,
        collective_ledger=[
            {
                "operation": "send",
                "count": 4,
                "dtype": "float32",
                "peer_or_root": 1,
                "tick": 1,
                "phase": "halo",
                "field_group": "health",
            },
            {
                "operation": "recv",
                "count": 4,
                "dtype": "float32",
                "peer_or_root": 1,
                "tick": 1,
                "phase": "halo",
                "field_group": "resource",
            },
        ],
    )
    rank1 = dict(
        base,
        rank=1,
        collective_ledger=[
            {
                "operation": "recv",
                "count": 4,
                "dtype": "float32",
                "peer_or_root": 0,
                "tick": 1,
                "phase": "halo",
                "field_group": "resource",
            },
            {
                "operation": "send",
                "count": 4,
                "dtype": "float32",
                "peer_or_root": 0,
                "tick": 1,
                "phase": "halo",
                "field_group": "health",
            },
        ],
    )
    result = _certify_collective_ledgers([rank0, rank1])
    assert not result["passed"]
    assert any("unmatched" in item for item in result["failures"])
