import pytest

from owl_raqic.qiskit_backend import gpu_execution


def test_qiskit_gpu_strict_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(
        gpu_execution,
        "build_aer_simulator_from_config",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced GPU failure")),
    )
    with pytest.raises(RuntimeError, match="strict Qiskit-Aer-GPU"):
        gpu_execution.run_statevector_probabilities_gpu(
            object(), n_actions=2, strict_gpu=True, allow_cpu_fallback=True
        )
