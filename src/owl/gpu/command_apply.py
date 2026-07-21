from __future__ import annotations

from typing import Any

from owl.gpu.commands import CommandKind, GPUCommand


def apply_command(run: Any, command: GPUCommand) -> dict[str, Any]:
    """Apply commands only at tick boundaries."""
    kind = command.kind
    payload = dict(command.payload)
    if kind == CommandKind.PAUSE:
        run.paused = True
    elif kind == CommandKind.RESUME:
        run.paused = False
    elif kind == CommandKind.CHECKPOINT:
        run.checkpoint_requested = True
    elif kind == CommandKind.REQUEST_VALIDATION:
        run.validation_requested = True
    elif kind == CommandKind.VISUAL_SETTING:
        run.visual_settings.update(payload)
    elif kind in (CommandKind.INJECT_FOOD, CommandKind.INJECT_TOXIN):
        if not command.state_mutating:
            raise PermissionError(f"{kind.value} requires state_mutating=True")
        y, x = int(payload["y"]), int(payload["x"])
        amount = float(payload.get("amount", 0.0))
        field = "food" if kind == CommandKind.INJECT_FOOD else "toxin"
        arr = run.ds.arrays[field]
        if not (0 <= y < arr.shape[0] and 0 <= x < arr.shape[1]):
            raise IndexError((y, x))
        arr[y, x] = run.ds.xp.clip(arr[y, x] + amount, 0.0, 1.0)
    else:
        raise ValueError(f"unsupported GPU command: {kind}")
    return {"kind": kind.value, "applied": True}
