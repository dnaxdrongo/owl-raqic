from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from owl.gpu.array_write import write_global_array

from .partition import SpatialShard


@dataclass
class GlobalReductionResult:
    patch_count: int
    global_integration: float | None
    global_policy_bias: Any

    def to_dict(self) -> Any:
        return {"patch_count": int(self.patch_count), "global_integration": self.global_integration}


def synchronize_global_context(
    ds: Any, cfg: Any, shard: SpatialShard, transport: Any, stream: Any, *, tick: int
) -> Any:
    """All-reduce compact owned-patch summaries and update replicated context."""
    xp = ds.xp
    patch_size = int(cfg.world.patch_size)
    halo_patch_rows = shard.halo_width // patch_size
    owned_patch_rows = shard.owned_height // patch_size
    local_slice = slice(halo_patch_rows, halo_patch_rows + owned_patch_rows)
    integration = ds.patch_arrays["integration"][local_slice, ...]
    local_sum = xp.asarray([xp.sum(integration)], dtype=xp.float64)
    global_sum = xp.zeros_like(local_sum)
    local_count = xp.asarray([integration.size], dtype=xp.int64)
    global_count = xp.zeros_like(local_count)
    transport.all_reduce(local_sum, global_sum, op="sum", stream=stream, tick=tick)
    transport.all_reduce(local_count, global_count, op="sum", stream=stream, tick=tick)
    if "policy_bias" in ds.patch_arrays:
        policy = ds.patch_arrays["policy_bias"][local_slice, ...]
        local_policy = xp.sum(policy, axis=(0, 1), dtype=xp.float64)
    else:
        local_policy = xp.zeros((ds.possibility.shape[-1],), dtype=xp.float64)
    global_policy = xp.zeros_like(local_policy)
    transport.all_reduce(local_policy, global_policy, op="sum", stream=stream, tick=tick)
    global_policy /= xp.maximum(xp.sum(global_policy), 1e-15)
    write_global_array(ds, "policy_bias", global_policy.astype(ds.possibility.dtype))
    ds.scalars["global_integration"] = global_sum[0] / xp.maximum(global_count[0], 1)
    ds.scalars["global_intention"] = xp.argmax(global_policy).astype(xp.int32)
    return GlobalReductionResult(
        patch_count=int(owned_patch_rows * integration.shape[1]),
        global_integration=None,
        global_policy_bias=global_policy,
    )
