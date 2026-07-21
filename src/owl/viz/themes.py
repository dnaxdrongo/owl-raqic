from __future__ import annotations

from dataclasses import dataclass

Color = tuple[int, int, int, int]


@dataclass(frozen=True)
class Theme:
    name: str
    background: Color
    grid: Color
    text: Color
    health_good: Color
    health_bad: Color
    resource: Color
    uncertainty: Color
    patch: Color
    empty_space: Color = (7, 10, 18, 255)
    food: Color = (120, 225, 95, 220)
    toxin: Color = (205, 75, 235, 205)
    waste: Color = (210, 145, 65, 215)
    obstacle: Color = (85, 96, 115, 255)
    boundary: Color = (125, 150, 190, 230)
    signal: Color = (70, 175, 255, 175)
    dead_shell: Color = (135, 145, 160, 165)
    panel: Color = (13, 19, 31, 242)
    panel_border: Color = (70, 95, 135, 210)
    selected: Color = (255, 245, 120, 255)
    action_alpha: int = 230


THEMES: dict[str, Theme] = {
    "owl_dark_neon": Theme(
        "owl_dark_neon",
        (4, 7, 14, 255),
        (38, 55, 85, 60),
        (225, 235, 255, 255),
        (48, 230, 130, 255),
        (255, 65, 90, 255),
        (80, 210, 255, 255),
        (255, 210, 70, 255),
        (150, 110, 255, 130),
    ),
    "owl_colorblind_safe": Theme(
        "owl_colorblind_safe",
        (9, 12, 17, 255),
        (85, 90, 100, 75),
        (250, 250, 250, 255),
        (0, 158, 115, 255),
        (213, 94, 0, 255),
        (86, 180, 233, 255),
        (240, 228, 66, 255),
        (204, 121, 167, 130),
        food=(0, 158, 115, 225),
        toxin=(204, 121, 167, 220),
        waste=(230, 159, 0, 220),
        signal=(86, 180, 233, 180),
    ),
    "owl_minimal_debug": Theme(
        "owl_minimal_debug",
        (0, 0, 0, 255),
        (60, 60, 60, 140),
        (255, 255, 255, 255),
        (0, 255, 0, 255),
        (255, 0, 0, 255),
        (0, 128, 255, 255),
        (255, 255, 0, 255),
        (128, 0, 255, 160),
        empty_space=(0, 0, 0, 255),
        food=(0, 255, 0, 255),
        toxin=(255, 0, 255, 255),
        waste=(255, 150, 0, 255),
        obstacle=(125, 125, 125, 255),
        signal=(0, 128, 255, 200),
    ),
}


def get_theme(name: str = "owl_dark_neon") -> Theme:
    return THEMES.get(str(name), THEMES["owl_dark_neon"])
