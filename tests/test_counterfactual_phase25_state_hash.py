from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from owl.core.state import EventRecord
from owl.counterfactual.state_hash import compare_state_science, differing_leaves, hash_state
from tests.counterfactual_phase25_helpers import source_run


def test_complete_hash_detects_registered_leaf_change(tmp_path) -> None:
    _, run, source = source_run(tmp_path)
    try:
        branch = source.state.branch_clone()
        original = hash_state(branch)
        branch.readout[5, 5] = (int(branch.readout[5, 5]) + 1) % 22
        changed = hash_state(branch)
        assert original.root != changed.root
        assert differing_leaves(original, changed) == ("arrays.readout",)
        assert not compare_state_science(branch, source.state).passed
    finally:
        run.close(checkpoint=False)


@dataclass(frozen=True)
class _HashManifest:
    metadata_names: tuple[str, ...] = ("event_queue",)


class _FakeCuPyArray:
    __module__ = "cupy._core.core"

    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value)
        self.dtype = self.value.dtype
        self.shape = self.value.shape
        self.nbytes = self.value.nbytes


def _metadata_state(value: object) -> SimpleNamespace:
    return SimpleNamespace(
        arrays={},
        patch_arrays={},
        global_arrays={},
        scalars={"tick": 2},
        metadata={
            "event_queue": [
                EventRecord(
                    kind="hash_test",
                    tick=2,
                    payload={"nested_array": value, "nonfinite": [np.nan, np.inf, -np.inf]},
                )
            ]
        },
        manifest=_HashManifest(),
    )


def test_metadata_hash_accepts_cupy_arrays_slots_dataclasses_and_nonfinite(
    monkeypatch,
) -> None:
    host = np.asarray([2, 4, 8], dtype=np.int16)
    fake_cupy = SimpleNamespace(
        ascontiguousarray=lambda value: value,
        asnumpy=lambda value: value.value.copy(),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    gpu = hash_state(_metadata_state(_FakeCuPyArray(host)))
    cpu = hash_state(_metadata_state(host))
    assert gpu.root == cpu.root
    assert gpu.device_to_host_bytes == host.nbytes
