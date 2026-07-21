from __future__ import annotations

from typing import Any

from owl.viz.environment_sprites import EnvironmentKind
from owl.viz.sprites import SPRITE_SPECS

_TRAIT_NIBBLES = (
    ("R-high", "Predation / aggression"),
    ("R-low", "Metabolism / reproduction"),
    ("G-high", "Cooperation / grazing"),
    ("G-low", "Resilience / boundary"),
    ("B-high", "Curiosity / memory"),
    ("B-low", "Coupling / communication"),
)

_ENVIRONMENT_LABELS = {
    EnvironmentKind.FOOD: "rounded nutrient granules",
    EnvironmentKind.TOXIN: "pulsing hazard bubbles",
    EnvironmentKind.WASTE: "amber dotted motes",
    EnvironmentKind.OBSTACLE: "solid angular crystal",
    EnvironmentKind.BOUNDARY: "high-contrast membrane",
    EnvironmentKind.SIGNAL: "channel-coded wave",
    EnvironmentKind.DEAD_SHELL: "desaturated former OW",
}


def action_legend_rows() -> list[dict[str, Any]]:
    return [
        {
            "action": int(action),
            "name": spec.name,
            "glyph": spec.glyph,
            "description": spec.description,
            "color": spec.color,
            "effect_family": spec.effect_family,
            "directional": spec.directional,
        }
        for action, spec in sorted(SPRITE_SPECS.items(), key=lambda item: int(item[0]))
    ]


def trait_hex_legend_rows() -> list[dict[str, str]]:
    return [{"nibble": nibble, "meaning": meaning} for nibble, meaning in _TRAIT_NIBBLES]


def environment_legend_rows() -> list[dict[str, str]]:
    return [{"kind": kind.value, "visual": _ENVIRONMENT_LABELS[kind]} for kind in EnvironmentKind]


def legend_text() -> str:
    sections = ["Trait Hex #RRGGBB"]
    sections.extend(f"  {row['nibble']:<6} {row['meaning']}" for row in trait_hex_legend_rows())
    sections.append("\nEnvironment")
    sections.extend(f"  {row['kind']:<12} {row['visual']}" for row in environment_legend_rows())
    sections.append("\nActions")
    sections.extend(
        f"  {row['action']:02d} {row['name']:<14} {row['effect_family']:<18} {row['description']}"
        for row in action_legend_rows()
    )
    return "\n".join(sections)
