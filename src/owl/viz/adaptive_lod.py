from __future__ import annotations


def adaptive_lod(
    cell_pixels: float,
    event_density: float,
    glyph_density: float = 1.0,
) -> str:
    """Choose an identity-preserving detail tier.

    Automatic LOD never removes all OW bodies.  Dense scientific heatmaps remain
    an explicitly selected view rather than an automatic replacement.
    """

    effective = float(cell_pixels) * max(0.25, float(glyph_density))
    density = max(0.0, float(event_density))
    if effective < 4.0:
        return "overview"
    if effective < 9.0 or density > 0.35:
        return "medium"
    if effective < 20.0 or density > 0.15:
        return "detail"
    return "focus"


def event_stride(event_count: int, clutter_budget: int) -> int:
    if clutter_budget <= 0:
        return max(1, event_count)
    return max(1, (int(event_count) + int(clutter_budget) - 1) // int(clutter_budget))
