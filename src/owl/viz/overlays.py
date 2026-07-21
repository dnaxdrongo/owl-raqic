"""Define visualization layers and the interfaces used to switch views.

Each layer is represented by a small data record that can be imported without
Pygame and contains the information needed to select a rendering mode.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OverlaySpec:
    """Describe one named visualization layer.

    Parameters
    ----------
    name:
        Stable identifier used by the viewer.
    description:
        Human-readable text for the help display.
    hotkey:
        Single-character key used to select the layer.
    channel:
        Optional communication-channel index for signal views.
    """

    name: str
    description: str
    hotkey: str
    channel: int | None = None


def default_overlays() -> list[OverlaySpec]:
    """Return the default OWL visualization layers.

    Returns
    -------
    list[OverlaySpec]
        Ordered visualization layers. The first entry is the default display.
        Signal views use channel indices that match ``SignalChannel`` values.
    """
    return [
        OverlaySpec("integration", "Integration/toxin/coordination composite", "1"),
        OverlaySpec("type", "Observer-window type sprites/colors", "2"),
        OverlaySpec("action", "Actualized action readout", "3"),
        OverlaySpec("food", "Food/nutrient density region", "4"),
        OverlaySpec("toxin", "Toxin/damage region", "5"),
        OverlaySpec("patches", "Patch membership and patch integration regions", "6"),
        OverlaySpec("waste", "Advanced ecology waste/digestion pressure", "w"),
        OverlaySpec("trust", "Source trust and deception diagnostics", "t"),
        OverlaySpec("genome", "Genome channels compressed to RGB", "g"),
        OverlaySpec("signal:0", "Signal FOOD channel", "7", channel=0),
        OverlaySpec("signal:1", "Signal DANGER channel", "8", channel=1),
        OverlaySpec("signal:3", "Signal COORDINATION channel", "9", channel=3),
        OverlaySpec("signal:7", "Signal INTEGRATION channel", "0", channel=7),
    ]
