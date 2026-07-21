from __future__ import annotations

import inspect

from owl.cadc import pipeline


def test_gpu_tensor_loader_enforces_join_contract_without_cudf_validate() -> None:
    source = inspect.getsource(pipeline._load_gpu)
    assert "validate=" not in source
    assert "targets.duplicated(subset=target_keys)" in source
    assert "externality.duplicated(subset=target_keys)" in source
    assert "externality one-to-one join changed branch target cardinality" in source
