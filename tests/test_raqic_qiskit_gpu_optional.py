from __future__ import annotations

import pytest

from owl_raqic.qiskit_backend.gpu_execution import aer_gpu_available_runtime


def test_qiskit_aer_gpu_runtime_optional():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    info = aer_gpu_available_runtime()
    # CPU-only CI may not have qiskit-aer-gpu; the API must return a structured object.
    assert isinstance(info.available, bool)
