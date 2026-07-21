from owl.gpu.multi_gpu import partition_rows
from owl.gpu.roofline import KernelRooflineEstimate


def test_partition_rows_covers_world():
    shards = partition_rows(10, [0, 1, 2])
    assert shards[0].row_start == 0
    assert shards[-1].row_stop == 10
    assert sum(s.rows for s in shards) == 10


def test_roofline_accounting():
    r = KernelRooflineEstimate("x", 100, 400, 200, 1000, 0.1)
    assert r.total_bytes == 600
    assert r.arithmetic_intensity == 1000 / 600
    assert r.achieved_bandwidth_bytes_s == 6000
