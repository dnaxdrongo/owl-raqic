from owl.gpu.backend import get_array_backend
from owl.gpu.graphs import GpuTickGraphManager


def test_graph_manager_honest_on_numpy():
    backend = get_array_backend(strict=False, allow_fallback=True)
    gm = GpuTickGraphManager(backend, mode="segments")
    status = gm.graph_status()
    if backend.name == "numpy":
        assert not status["can_capture"]
        assert "CUDA" in status["reason"]
    else:
        assert "can_capture" in status
