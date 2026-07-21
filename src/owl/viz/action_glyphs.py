from __future__ import annotations

import math


def glyph_polyline(glyph: str, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """Procedural glyph geometry in normalized cell coordinates."""
    if glyph.startswith("arrow"):
        angle = {
            "arrow_n": -math.pi / 2,
            "arrow_s": math.pi / 2,
            "arrow_e": 0,
            "arrow_w": math.pi,
            "arrow_ne": -math.pi / 4,
            "arrow_nw": -3 * math.pi / 4,
            "arrow_se": math.pi / 4,
            "arrow_sw": 3 * math.pi / 4,
            "arrow_away": math.pi,
            "chevron": 0,
        }.get(glyph, 0.0)
        tip = (cx + r * math.cos(angle), cy + r * math.sin(angle))
        left = (cx + 0.4 * r * math.cos(angle + 2.5), cy + 0.4 * r * math.sin(angle + 2.5))
        right = (cx + 0.4 * r * math.cos(angle - 2.5), cy + 0.4 * r * math.sin(angle - 2.5))
        return [left, tip, right]
    if glyph == "plus":
        return [(cx - r, cy), (cx + r, cy), (cx, cy), (cx, cy - r), (cx, cy + r)]
    if glyph == "bar":
        return [(cx - r, cy - r), (cx + r, cy + r), (cx + r, cy - r), (cx - r, cy + r)]
    if glyph == "leaf":
        return [(cx, cy - r), (cx + r * 0.7, cy), (cx, cy + r), (cx - r * 0.7, cy), (cx, cy - r)]
    if glyph in ("merge", "nodes"):
        return [(cx - r, cy), (cx, cy), (cx + r, cy)]
    if glyph == "split":
        return [(cx - r, cy), (cx, cy), (cx + r, cy - r * 0.7), (cx, cy), (cx + r, cy + r * 0.7)]
    if glyph == "broadcast":
        return [(cx - r * 0.2, cy), (cx + r * 0.2, cy)]
    return [(cx, cy)]
