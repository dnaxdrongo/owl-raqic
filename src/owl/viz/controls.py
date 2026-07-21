"""Interactive control hooks for visualization and parameter steering.

The controls module is intentionally import-safe in headless environments:
Pygame is optional, and ASCII fallbacks are accepted in tests.
"""

from __future__ import annotations

from types import ModuleType

from owl.core.config import SimulationConfig
from owl.core.state import WorldState

_pygame: ModuleType | None
try:  # pragma: no cover - exercised only when pygame is installed.
    import pygame as _pygame_module
except Exception:  # noqa: BLE001 - visualization must import without pygame.
    _pygame = None
else:
    _pygame = _pygame_module


def _key_matches(key: int, *names_or_chars: str) -> bool:
    """Return whether ``key`` matches any Pygame key name or ASCII char."""
    for item in names_or_chars:
        if len(item) == 1 and key == ord(item):
            return True
        if _pygame is not None:
            value = getattr(_pygame, item, None)
            if value is not None and key == value:
                return True
    return False


def handle_parameter_hotkeys(state: WorldState, cfg: SimulationConfig, key: int) -> None:
    """Apply runtime visualization/parameter hotkey adjustments.

    Mutates only configuration fields used for live steering and selected
    bounded state fields for debug exploration. It does not change readouts,
    possibility vectors, or engine rules.

    Supported keys
    --------------
    ``+``/``=``:
        Increase visualization scale.
    ``-``:
        Decrease visualization scale.
    ``[``/``]``:
        Decrease/increase action softmax beta.
    ``n``/``m``:
        Decrease/increase phase noise.
    ``c``:
        Cycle the configured default visualization layer.
    ``h``:
        Toggle debug invariant assertions.
    """
    del state  # currently no direct state steering; keep signature stable.

    if _key_matches(key, "K_PLUS", "K_EQUALS", "+", "="):
        cfg.visualization.scale = min(64, int(cfg.visualization.scale) + 1)
    elif _key_matches(key, "K_MINUS", "-"):
        cfg.visualization.scale = max(1, int(cfg.visualization.scale) - 1)
    elif _key_matches(key, "K_LEFTBRACKET", "["):
        cfg.actions.beta = max(0.1, float(cfg.actions.beta) * 0.9)
    elif _key_matches(key, "K_RIGHTBRACKET", "]"):
        cfg.actions.beta = min(25.0, float(cfg.actions.beta) * 1.1)
    elif _key_matches(key, "K_n", "n"):
        cfg.phase.phase_noise_sigma = max(0.0, float(cfg.phase.phase_noise_sigma) * 0.8)
    elif _key_matches(key, "K_m", "m"):
        cfg.phase.phase_noise_sigma = min(1.0, float(cfg.phase.phase_noise_sigma) * 1.25 + 1e-4)
    elif _key_matches(key, "K_h", "h"):
        cfg.debug.assert_invariants = not bool(cfg.debug.assert_invariants)
    elif _key_matches(key, "K_c", "c"):
        order = ["integration", "type", "action", "food", "toxin", "patches"]
        current = str(cfg.visualization.color_by)
        cfg.visualization.color_by = (
            order[(order.index(current) + 1) % len(order)] if current in order else order[0]
        )


def pause_or_step_controls() -> dict[str, bool]:
    """Poll Pygame events and return playback-control requests.

    Returns
    -------
    dict[str, bool]
        Keys: ``quit``, ``toggle_pause``, ``step``, ``rewind``,
        ``fast_forward``, ``zoom_in``, and ``zoom_out``. In a headless/no-Pygame
        environment all values are ``False``.

    Notes
    -----
    This helper is intentionally small. ``PygameViewer.handle_events`` owns
    detailed pan/zoom/hover state; this function is a reusable polling hook for
    simple scripts.
    """
    controls = {
        "quit": False,
        "toggle_pause": False,
        "step": False,
        "rewind": False,
        "fast_forward": False,
        "zoom_in": False,
        "zoom_out": False,
    }
    if _pygame is None:
        return controls

    for event in _pygame.event.get():  # pragma: no cover - needs pygame runtime.
        if event.type == _pygame.QUIT:
            controls["quit"] = True
        elif event.type == _pygame.KEYDOWN:
            key = int(event.key)
            if _key_matches(key, "K_SPACE", "K_p", " ", "p"):
                controls["toggle_pause"] = True
            elif _key_matches(key, "K_PERIOD", "."):
                controls["step"] = True
            elif _key_matches(key, "K_LEFT", "K_COMMA", ","):
                controls["rewind"] = True
            elif _key_matches(key, "K_RIGHT", "K_SLASH", "/"):
                controls["fast_forward"] = True
            elif _key_matches(key, "K_PLUS", "K_EQUALS", "+", "="):
                controls["zoom_in"] = True
            elif _key_matches(key, "K_MINUS", "-"):
                controls["zoom_out"] = True
        elif getattr(_pygame, "MOUSEWHEEL", None) is not None and event.type == _pygame.MOUSEWHEEL:
            if event.y > 0:
                controls["zoom_in"] = True
            elif event.y < 0:
                controls["zoom_out"] = True
    return controls
