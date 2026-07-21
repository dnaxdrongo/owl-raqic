"""Import-safe optional ML dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    torch: Any
    nn: Any

    class TorchModule:
        """Minimal static surface of ``torch.nn.Module`` used by this package."""

        def __init__(self) -> None: ...

        def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

        def register_buffer(self, name: str, tensor: Any) -> None: ...
else:
    try:
        import torch
        from torch import nn
    except ImportError:  # pragma: no cover - target extras exercise this path
        torch = None
        nn = None
    TorchModule = nn.Module if nn is not None else object


def require_torch() -> tuple[Any, Any]:
    """Return Torch modules or fail with the exact missing-extra diagnosis."""
    if torch is None or nn is None:
        raise RuntimeError("Phase 4 neural models require the cadc training extra (Torch)")
    return torch, nn
