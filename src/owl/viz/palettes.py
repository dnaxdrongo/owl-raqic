"""Color palettes for dense array visualization.

The palette functions are pure: they read :class:`owl.core.state.WorldState` and
return RGB arrays in Pygame surfarray orientation ``(width, height, 3)`` with
``dtype=np.uint8``. They do not import or require Pygame, so visualization tests
can run in headless environments.
"""

from __future__ import annotations

import numpy as np

from owl.core.actions import Action, SignalChannel
from owl.core.state import WorldState, field_shape

_TYPE_COLORS = np.array(
    [
        (112, 214, 255),  # grazer-like
        (171, 255, 161),  # cooperator-like
        (255, 107, 107),  # predator-like
        (255, 209, 102),  # scavenger-like
        (196, 181, 253),  # explorer-like
        (245, 158, 11),
        (52, 211, 153),
        (244, 114, 182),
    ],
    dtype=np.uint8,
)

_ACTION_COLORS = np.zeros((len(Action), 3), dtype=np.uint8)
_ACTION_COLORS[int(Action.REST)] = (28, 32, 42)
_ACTION_COLORS[int(Action.SENSE)] = (90, 200, 255)
for _move in (
    Action.MOVE_N,
    Action.MOVE_S,
    Action.MOVE_E,
    Action.MOVE_W,
    Action.MOVE_NE,
    Action.MOVE_NW,
    Action.MOVE_SE,
    Action.MOVE_SW,
):
    _ACTION_COLORS[int(_move)] = (190, 190, 190)
_ACTION_COLORS[int(Action.FEED)] = (44, 220, 80)
_ACTION_COLORS[int(Action.COMMUNICATE)] = (92, 140, 255)
_ACTION_COLORS[int(Action.INHIBIT)] = (255, 120, 70)
_ACTION_COLORS[int(Action.INTEGRATE)] = (180, 130, 255)
_ACTION_COLORS[int(Action.REPAIR)] = (255, 220, 90)
_ACTION_COLORS[int(Action.REPRODUCE)] = (255, 140, 210)
_ACTION_COLORS[int(Action.INGEST)] = (230, 60, 60)
_ACTION_COLORS[int(Action.FLEE)] = (255, 170, 40)
_ACTION_COLORS[int(Action.PURSUE)] = (180, 40, 40)
_ACTION_COLORS[int(Action.EXPEL)] = (120, 80, 40)
_ACTION_COLORS[int(Action.SPLIT)] = (80, 200, 200)
_ACTION_COLORS[int(Action.MERGE)] = (200, 200, 255)

_SIGNAL_COLORS = np.array(
    [
        (40, 220, 90),  # FOOD
        (255, 80, 70),  # DANGER
        (255, 130, 60),  # THREAT
        (90, 150, 255),  # COORDINATION
        (255, 70, 170),  # DISTRESS
        (255, 180, 230),  # REPRODUCTION
        (255, 220, 90),  # TERRITORY
        (180, 120, 255),  # INTEGRATION
    ],
    dtype=np.uint8,
)


def _empty_rgb(state: WorldState) -> np.ndarray:
    """Return a black RGB array in Pygame surfarray orientation."""
    height, width = field_shape(state)
    return np.zeros((width, height, 3), dtype=np.uint8)


def _cell_to_pygame_rgb(cell_rgb: np.ndarray) -> np.ndarray:
    """Convert ``(height, width, 3)`` RGB to Pygame ``(width, height, 3)``."""
    if cell_rgb.ndim != 3 or cell_rgb.shape[-1] != 3:
        raise ValueError(f"cell_rgb must have shape (height, width, 3), got {cell_rgb.shape}")
    return np.ascontiguousarray(np.swapaxes(cell_rgb, 0, 1).astype(np.uint8, copy=False))


def _living_alpha(state: WorldState) -> np.ndarray:
    """Return a cell-level visibility multiplier for living/non-obstacle cells."""
    return ((state.health > 0.0) & (~state.obstacle)).astype(np.float32)


def integration_palette(state: WorldState) -> np.ndarray:
    """Map integration, toxin, food, and coordination fields to RGB.

    Returns
    -------
    np.ndarray
        RGB array with shape ``(width, height, 3)`` and dtype ``uint8``.
        Green encodes integration, red encodes toxin, and blue encodes
        coordination signal pressure. Food mildly brightens green.
    """
    height, width = field_shape(state)
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    integration = np.clip(state.integration, 0.0, 1.0)
    toxin = np.clip(state.toxin, 0.0, 1.0)
    food = np.clip(state.food, 0.0, 1.0)

    coord = np.zeros((height, width), dtype=np.float32)
    idx = int(SignalChannel.COORDINATION)
    if state.signal.shape[:2] == (height, width) and idx < state.signal.shape[-1]:
        coord = np.clip(state.signal[..., idx], 0.0, 1.0)

    alive = _living_alpha(state)
    rgb[..., 0] = 60.0 * alive + 180.0 * toxin
    rgb[..., 1] = 30.0 + 215.0 * integration * alive + 90.0 * food
    rgb[..., 2] = 35.0 + 180.0 * coord
    rgb[state.obstacle, :] = (45, 45, 52)

    return _cell_to_pygame_rgb(np.clip(rgb, 0.0, 255.0))


def type_palette(state: WorldState) -> np.ndarray:
    """Map OW type ids to distinct RGB colors.

    Dead cells are dark, obstacles are gray, and living cells are colored by
    ``state.ow_type`` modulo the palette length. These colors correspond to the
    sprite colors used by :class:`owl.viz.pygame_viewer.PygameViewer`.
    """
    height, width = field_shape(state)
    type_ids = np.asarray(state.ow_type, dtype=np.int64)
    if type_ids.shape != (height, width):
        raise ValueError(f"state.ow_type must have shape {(height, width)}, got {type_ids.shape}")

    colors = _TYPE_COLORS[np.mod(type_ids, len(_TYPE_COLORS))]
    rgb = colors.astype(np.float32)
    alive = _living_alpha(state)
    rgb *= alive[..., None]
    rgb[~(state.health > 0.0), :] = (8, 10, 14)
    rgb[state.obstacle, :] = (55, 55, 60)
    return _cell_to_pygame_rgb(np.clip(rgb, 0.0, 255.0))


def action_palette(state: WorldState) -> np.ndarray:
    """Map actualized action readouts to RGB colors.

    Returns an RGB array with shape ``(width, height, 3)``. Invalid readout
    values are wrapped modulo ``len(Action)`` so corrupt diagnostics remain
    visible instead of crashing the viewer.
    """
    height, width = field_shape(state)
    readout = np.asarray(state.readout, dtype=np.int64)
    if readout.shape != (height, width):
        raise ValueError(f"state.readout must have shape {(height, width)}, got {readout.shape}")

    rgb = _ACTION_COLORS[np.mod(readout, len(Action))].astype(np.float32)
    rgb *= (0.25 + 0.75 * _living_alpha(state))[..., None]
    rgb[state.obstacle, :] = (55, 55, 60)
    return _cell_to_pygame_rgb(np.clip(rgb, 0.0, 255.0))


def signal_overlay_palette(state: WorldState, channel: int) -> np.ndarray:
    """Map a communication channel to an RGB visualization layer.

    Parameters
    ----------
    state:
        Runtime dense state. This function does not mutate state.
    channel:
        Channel index to visualize.

    Returns
    -------
    np.ndarray
        RGB array with shape ``(width, height, 3)``. The selected channel is
        rendered in a channel-specific color, blended with food/toxin context.
    """
    height, width = field_shape(state)
    if state.signal.shape[:2] != (height, width):
        raise ValueError(
            f"state.signal must begin with shape {(height, width)}, got {state.signal.shape}"
        )
    if channel < 0 or channel >= state.signal.shape[-1]:
        raise ValueError(f"channel {channel} outside available range [0, {state.signal.shape[-1]})")

    value = np.clip(state.signal[..., int(channel)], 0.0, 1.0)
    color = _SIGNAL_COLORS[int(channel) % len(_SIGNAL_COLORS)].astype(np.float32)
    rgb = value[..., None] * color[None, None, :]
    rgb[..., 0] += 60.0 * np.clip(state.toxin, 0.0, 1.0)
    rgb[..., 1] += 45.0 * np.clip(state.food, 0.0, 1.0)
    rgb[state.obstacle, :] = (45, 45, 52)
    return _cell_to_pygame_rgb(np.clip(rgb, 0.0, 255.0))


def food_palette(state: WorldState) -> np.ndarray:
    """Return a green food-region visualization with living cells subtly visible."""
    height, width = field_shape(state)
    food = np.clip(state.food, 0.0, 1.0)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 1] = 255.0 * food
    rgb[..., 0] = 20.0 * _living_alpha(state)
    rgb[..., 2] = 35.0 * _living_alpha(state)
    rgb[state.obstacle, :] = (45, 45, 52)
    return _cell_to_pygame_rgb(rgb)


def toxin_palette(state: WorldState) -> np.ndarray:
    """Return a red toxin-region visualization with living cells subtly visible."""
    height, width = field_shape(state)
    toxin = np.clip(state.toxin, 0.0, 1.0)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 0] = 255.0 * toxin + 20.0 * _living_alpha(state)
    rgb[..., 1] = 20.0 * _living_alpha(state)
    rgb[..., 2] = 35.0 * _living_alpha(state)
    rgb[state.obstacle, :] = (45, 45, 52)
    return _cell_to_pygame_rgb(np.clip(rgb, 0.0, 255.0))


def patch_palette(state: WorldState) -> np.ndarray:
    """Color cells by current parent patch id and patch integration.

    This visualization follows constituent cells through ``state.parent_id`` rather
    than only drawing fixed rectangular patch tiles. When cells move across
    patches and their parent id changes, the colored patch membership region
    moves with them.
    """
    height, width = field_shape(state)
    parent = np.asarray(state.parent_id, dtype=np.int64)
    if parent.shape != (height, width):
        raise ValueError(f"state.parent_id must have shape {(height, width)}, got {parent.shape}")

    palette = np.array(
        [
            (80, 120, 255),
            (100, 220, 150),
            (255, 190, 70),
            (235, 100, 160),
            (170, 120, 255),
            (120, 230, 230),
            (255, 120, 90),
            (190, 220, 90),
        ],
        dtype=np.float32,
    )
    valid = parent >= 0
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[valid] = palette[np.mod(parent[valid], len(palette))]

    # Dim by cell integration/health so patch regions are still formal, not
    # decorative. Empty/dead cells remain dark.
    multiplier = (0.25 + 0.75 * np.clip(state.integration, 0.0, 1.0)) * _living_alpha(state)
    rgb *= multiplier[..., None]
    rgb[state.obstacle, :] = (45, 45, 52)
    return _cell_to_pygame_rgb(np.clip(rgb, 0.0, 255.0))


# --- Advanced build palettes -------------------------------------------------


def waste_palette(state: WorldState) -> np.ndarray:
    """Render waste/recycled nutrient pressure as amber regions."""
    height, width = field_shape(state)
    waste = getattr(state, "waste", None)
    if not isinstance(waste, np.ndarray):
        waste = np.zeros((height, width), dtype=np.float32)
    waste = np.clip(waste, 0.0, 1.0)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 0] = 220.0 * waste
    rgb[..., 1] = 120.0 * waste + 20.0 * _living_alpha(state)
    rgb[..., 2] = 20.0 * _living_alpha(state)
    rgb[state.obstacle, :] = (45, 45, 52)
    return _cell_to_pygame_rgb(np.clip(rgb, 0.0, 255.0))


def trust_palette(state: WorldState) -> np.ndarray:
    """Render source/deception trust diagnostics."""
    height, width = field_shape(state)
    trust = getattr(state, "neighbor_trust", None)
    deception = getattr(state, "deception_memory", None)
    trust_mean = (
        np.ones((height, width), dtype=np.float32)
        if not isinstance(trust, np.ndarray)
        else np.mean(trust, axis=(2, 3))
    )
    deception_mean = (
        np.zeros((height, width), dtype=np.float32)
        if not isinstance(deception, np.ndarray)
        else np.mean(deception, axis=-1)
    )
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 1] = 255.0 * np.clip(trust_mean, 0.0, 1.0) * _living_alpha(state)
    rgb[..., 0] = 255.0 * np.clip(deception_mean, 0.0, 1.0)
    rgb[..., 2] = 80.0 * _living_alpha(state)
    rgb[state.obstacle, :] = (45, 45, 52)
    return _cell_to_pygame_rgb(np.clip(rgb, 0.0, 255.0))


def genome_palette(state: WorldState) -> np.ndarray:
    """Render genome channels as RGB compressed diagnostics."""
    height, width = field_shape(state)
    genome = getattr(state, "genome", None)
    if not isinstance(genome, np.ndarray) or genome.shape[:2] != (height, width):
        return type_palette(state)
    g = np.clip(genome, 0.0, 1.0)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 0] = 255.0 * g[..., 0]
    rgb[..., 1] = 255.0 * g[..., min(1, g.shape[-1] - 1)]
    rgb[..., 2] = 255.0 * g[..., min(2, g.shape[-1] - 1)]
    rgb *= _living_alpha(state)[..., None]
    rgb[state.obstacle, :] = (45, 45, 52)
    return _cell_to_pygame_rgb(np.clip(rgb, 0.0, 255.0))
