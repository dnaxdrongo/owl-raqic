from __future__ import annotations

import numpy as np
import pytest

from owl.gpu.counter_rng import counter_uniform
from owl.gpu.distributed.boundary_events import (
    BoundaryCandidate,
    resolve_boundary_candidates,
)
from owl.gpu.distributed.fake_transport import FakeCollectiveGroup
from owl.gpu.distributed.partition import partition_rows


def test_patch_aligned_partition_covers_world():
    shards = partition_rows(20, 10, 2, 5, boundary_mode="toroidal")
    assert [(s.owned_start, s.owned_stop) for s in shards] == [(0, 10), (10, 20)]
    assert shards[0].north_rank == 1
    assert shards[1].south_rank == 0


def test_partition_rejects_more_ranks_than_patch_rows():
    with pytest.raises(ValueError):
        partition_rows(10, 10, 3, 5)


def test_boundary_resolution_is_rank_order_independent():
    candidates = [
        BoundaryCandidate(10, 99, 5, 0, 1),
        BoundaryCandidate(11, 99, 7, 1, 1),
        BoundaryCandidate(12, 99, 7, 0, 1),
    ]
    left = resolve_boundary_candidates(candidates)
    right = resolve_boundary_candidates(reversed(candidates))
    assert left == right
    assert left[0].source_global_id == 11


def test_fake_transport_enforces_matching_sequence():
    group = FakeCollectiveGroup(2)
    rank0 = group.endpoint(0)
    rank1 = group.endpoint(1)
    source = np.arange(4, dtype=np.float32)
    target = np.zeros_like(source)
    rank0.send(source, peer=1)
    rank1.recv(target, peer=0)
    np.testing.assert_array_equal(target, source)


def test_counter_rng_is_partition_invariant():
    ids = np.arange(100, dtype=np.int64)
    whole = counter_uniform(123, 7, ids, np, stream_id=9)
    parts = np.concatenate(
        [
            counter_uniform(123, 7, ids[:40], np, stream_id=9),
            counter_uniform(123, 7, ids[40:], np, stream_id=9),
        ]
    )
    np.testing.assert_array_equal(parts, whole)
