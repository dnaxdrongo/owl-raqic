"""Multiscale update scheduling utilities.

The scheduler keeps cadence decisions out of the mathematical modules. It does
not mutate state; it only answers whether patch/global/recording work should run
at a given tick.
"""

from __future__ import annotations

from owl.core.config import SimulationConfig


def _validate_tick(tick: int) -> int:
    """Return a nonnegative integer tick or raise a clear error."""
    try:
        value = int(tick)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tick must be an integer-like value, got {tick!r}") from exc
    if value < 0:
        raise ValueError(f"tick must be nonnegative, got {tick!r}")
    return value


def should_update_patches(tick: int, cfg: SimulationConfig) -> bool:
    """Return whether patch state should update this tick.

    Parameters
    ----------
    tick:
        Current simulation tick. Tick zero is treated as an initialization tick.
    cfg:
        Simulation configuration. The baseline updates patch state every tick because
        patches are the immediate parent observer windows used by top-down bias.

    Returns
    -------
    bool
        Always ``True`` for valid nonnegative ticks in the baseline.
    """
    del cfg
    _validate_tick(tick)
    return True


def should_update_global(tick: int, cfg: SimulationConfig) -> bool:
    """Return whether apex/global state should update this tick.

    Parameters
    ----------
    tick:
        Current simulation tick.
    cfg:
        Simulation configuration. ``cfg.topdown.apex_update_every`` controls
        cadence after tick zero.

    Returns
    -------
    bool
        ``True`` at tick zero and when ``tick`` is an exact multiple of the apex
        update cadence.
    """
    value = _validate_tick(tick)
    cadence = int(cfg.topdown.apex_update_every)
    if cadence <= 0:
        raise ValueError(f"cfg.topdown.apex_update_every must be positive, got {cadence}")
    return value == 0 or value % cadence == 0


def should_record(tick: int, cfg: SimulationConfig) -> bool:
    """Return whether state should be recorded this tick.

    Parameters
    ----------
    tick:
        Current simulation tick.
    cfg:
        Simulation configuration. ``cfg.recording.enabled`` and
        ``cfg.recording.record_every`` determine recording cadence.

    Returns
    -------
    bool
        ``False`` when recording is disabled; otherwise ``True`` at tick zero
        and exact multiples of ``record_every``.
    """
    value = _validate_tick(tick)
    if not cfg.recording.enabled:
        return False
    cadence = int(cfg.recording.record_every)
    if cadence <= 0:
        raise ValueError(f"cfg.recording.record_every must be positive, got {cadence}")
    return value == 0 or value % cadence == 0
