from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from owl.viz.visual_snapshot import VisualSnapshot


@dataclass(frozen=True)
class TraitVector:
    predatory_pressure: float
    energetic_growth: float
    cooperative_ecology: float
    resilience: float
    cognition_exploration: float
    cross_scale_communication: float

    def values(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.predatory_pressure,
            self.energetic_growth,
            self.cooperative_ecology,
            self.resilience,
            self.cognition_exploration,
            self.cross_scale_communication,
        )


@dataclass(frozen=True)
class TraitColor:
    raw_hex: str
    rgb: tuple[int, int, int]
    nibbles: tuple[int, int, int, int, int, int]
    outline_rgba: tuple[int, int, int, int]
    display_rgb: tuple[int, int, int] | None = None

    @property
    def rendered_rgb(self) -> tuple[int, int, int]:
        return self.display_rgb or self.rgb


def normalize_trait(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    if upper <= lower:
        raise ValueError("trait upper bound must be greater than lower bound")
    normalized = (numeric - float(lower)) / (float(upper) - float(lower))
    return max(0.0, min(1.0, normalized))


def quantize_nibble(value: float) -> int:
    # Python round uses bankers rounding, matching np.rint for half-way values.
    return max(0, min(15, int(round(normalize_trait(value) * 15.0))))


def combine_trait_group(*values: float) -> float:
    finite = [normalize_trait(float(value)) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0


def _sample(snapshot: VisualSnapshot, name: str, y: int, x: int, default: float = 0.0) -> float:
    value = snapshot.arrays.get(name)
    if value is None:
        return float(default)
    return float(np.asarray(value)[y, x])


def trait_vector_from_snapshot(snapshot: VisualSnapshot, y: int, x: int) -> TraitVector:
    return TraitVector(
        predatory_pressure=combine_trait_group(
            _sample(snapshot, "aggression", y, x),
            _sample(snapshot, "predation", y, x),
        ),
        energetic_growth=combine_trait_group(
            _sample(snapshot, "metabolism", y, x),
            _sample(snapshot, "reproduction_rate", y, x),
        ),
        cooperative_ecology=combine_trait_group(
            _sample(snapshot, "cooperation", y, x),
            _sample(snapshot, "grazing", y, x),
        ),
        resilience=combine_trait_group(
            _sample(snapshot, "toxin_resistance", y, x),
            _sample(snapshot, "boundary", y, x),
        ),
        cognition_exploration=combine_trait_group(
            _sample(snapshot, "curiosity", y, x),
            _sample(snapshot, "memory_capacity", y, x),
            _sample(snapshot, "mobility", y, x),
        ),
        cross_scale_communication=combine_trait_group(
            _sample(snapshot, "coupling_strength", y, x),
            _sample(snapshot, "emit_strength", y, x),
            _sample(snapshot, "signal_precision", y, x),
        ),
    )


def encode_trait_hex(traits: TraitVector) -> TraitColor:
    nibbles = tuple(quantize_nibble(value) for value in traits.values())
    raw_hex = "#" + "".join(f"{value:X}" for value in nibbles)
    rgb = hex_to_rgb(raw_hex)
    return TraitColor(raw_hex, rgb, nibbles, contrasting_outline(rgb))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    raw = value.strip().lstrip("#")
    if len(raw) != 6:
        raise ValueError("trait hex must contain exactly six hexadecimal digits")
    try:
        return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"invalid trait hex: {value!r}") from exc


def _srgb_component(value: int) -> float:
    channel = float(value) / 255.0
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = (_srgb_component(channel) for channel in rgb)
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def contrasting_outline(rgb: tuple[int, int, int]) -> tuple[int, int, int, int]:
    return (245, 249, 255, 255) if relative_luminance(rgb) < 0.34 else (7, 10, 18, 255)


def _matrix_transform(rgb: tuple[int, int, int], matrix: np.ndarray) -> tuple[int, int, int]:
    linear = np.asarray(rgb, dtype=float) / 255.0
    transformed = np.clip(matrix @ linear, 0.0, 1.0)
    return tuple(int(round(value * 255.0)) for value in transformed)  # type: ignore[return-value]


def _srgb_to_linear_channel(value: int) -> float:
    channel = float(value) / 255.0
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _linear_to_srgb_channel(value: float) -> int:
    bounded = max(0.0, min(1.0, float(value)))
    channel = 12.92 * bounded if bounded <= 0.0031308 else 1.055 * bounded ** (1.0 / 2.4) - 0.055
    return int(round(max(0.0, min(1.0, channel)) * 255.0))


def _rgb_to_oklab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    red, green, blue = (_srgb_to_linear_channel(value) for value in rgb)

    linear_light = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    linear_medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    linear_short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue

    light_root, medium_root, short_root = np.cbrt((linear_light, linear_medium, linear_short))

    return (
        float(0.2104542553 * light_root + 0.7936177850 * medium_root - 0.0040720468 * short_root),
        float(1.9779984951 * light_root - 2.4285922050 * medium_root + 0.4505937099 * short_root),
        float(0.0259040371 * light_root + 0.7827717662 * medium_root - 0.8086757660 * short_root),
    )


def _oklab_to_rgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    lightness, a_axis, b_axis = lab

    light_root = lightness + 0.3963377774 * a_axis + 0.2158037573 * b_axis
    medium_root = lightness - 0.1055613458 * a_axis - 0.0638541728 * b_axis
    short_root = lightness - 0.0894841775 * a_axis - 1.2914855480 * b_axis

    linear_light = light_root**3
    linear_medium = medium_root**3
    linear_short = short_root**3

    red_linear = (
        4.0767416621 * linear_light - 3.3077115913 * linear_medium + 0.2309699292 * linear_short
    )
    green_linear = (
        -1.2684380046 * linear_light + 2.6097574011 * linear_medium - 0.3413193965 * linear_short
    )
    blue_linear = (
        -0.0041960863 * linear_light - 0.7034186147 * linear_medium + 1.7076147010 * linear_short
    )

    return tuple(
        _linear_to_srgb_channel(value) for value in (red_linear, green_linear, blue_linear)
    )  # type: ignore[return-value]


@lru_cache(maxsize=65536)
def transform_perceptual_color(color: TraitColor) -> TraitColor:
    """Create a display-balanced OKLab variant while preserving the raw Trait Hex.

    Stable identity remains the six-nibble ``raw_hex`` value.  Only the rendered
    sRGB triplet is adjusted to keep very dark colors legible against the dark
    world background and to limit extreme chroma clipping.
    """
    lightness, a_axis, b_axis = _rgb_to_oklab(color.rgb)
    chroma = float(np.hypot(a_axis, b_axis))
    target_lightness = float(np.clip(lightness, 0.48, 0.78))
    if chroma > 0.19:
        scale = 0.19 / chroma
        a_axis *= scale
        b_axis *= scale
    elif chroma < 0.045 and chroma > 0.0:
        scale = 0.045 / chroma
        a_axis *= scale
        b_axis *= scale
    rendered = _oklab_to_rgb((target_lightness, a_axis, b_axis))
    return TraitColor(
        raw_hex=color.raw_hex,
        rgb=color.rgb,
        nibbles=color.nibbles,
        outline_rgba=contrasting_outline(rendered),
        display_rgb=rendered,
    )


@lru_cache(maxsize=65536)
def transform_accessibility_color(color: TraitColor, mode: str) -> TraitColor:
    mode_name = str(mode).lower()
    if mode_name == "standard":
        return color
    matrices: dict[str, np.ndarray] = {
        "deuteranopia": np.asarray(((0.625, 0.375, 0.0), (0.70, 0.30, 0.0), (0.0, 0.30, 0.70))),
        "protanopia": np.asarray(((0.567, 0.433, 0.0), (0.558, 0.442, 0.0), (0.0, 0.242, 0.758))),
        "tritanopia": np.asarray(((0.95, 0.05, 0.0), (0.0, 0.433, 0.567), (0.0, 0.475, 0.525))),
    }
    if mode_name in matrices:
        rendered = _matrix_transform(color.rgb, matrices[mode_name])
    elif mode_name == "monochrome":
        gray = int(round(relative_luminance(color.rgb) ** (1.0 / 2.2) * 255.0))
        rendered = (gray, gray, gray)
    elif mode_name == "high_contrast":
        h, s, v = colorsys.rgb_to_hsv(*(channel / 255.0 for channel in color.rgb))
        s = max(0.72, s)
        v = 0.92 if v >= 0.45 else 0.30
        rendered = tuple(int(round(channel * 255.0)) for channel in colorsys.hsv_to_rgb(h, s, v))
    else:
        raise ValueError(f"unknown accessibility color mode: {mode}")
    return TraitColor(
        raw_hex=color.raw_hex,
        rgb=color.rgb,
        nibbles=color.nibbles,
        outline_rgba=contrasting_outline(rendered),
        display_rgb=rendered,
    )


def trait_distance(left: TraitVector, right: TraitVector) -> float:
    return float(np.linalg.norm(np.asarray(left.values()) - np.asarray(right.values())))
