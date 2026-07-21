from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from owl.core.actions import Action


@dataclass(frozen=True)
class SpritePose:
    position: tuple[float, float]
    rotation_degrees: float = 0.0
    scale: float = 1.0
    alpha: float = 1.0
    squash: tuple[float, float] = (1.0, 1.0)


@dataclass(frozen=True)
class EffectInstance:
    kind: str
    source: tuple[float, float]
    target: tuple[float, float] | None
    color: tuple[int, int, int, int]
    progress: float
    intensity: float
    layer: int
    seed: int
    channel: int = -1
    label: str = ""


@dataclass(frozen=True)
class AnimationClip:
    action: Action
    duration: float
    directional: bool
    context: Any
    family: str


def clamp_progress(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def ease_in_out(value: float) -> float:
    t = clamp_progress(value)
    return t * t * (3.0 - 2.0 * t)


def ease_out_back(value: float) -> float:
    t = clamp_progress(value) - 1.0
    c1 = 1.70158
    c3 = c1 + 1.0
    return float(1.0 + c3 * t**3 + c1 * t**2)


def pulse(value: float, cycles: float = 1.0) -> float:
    t = clamp_progress(value)
    return float(0.5 - 0.5 * math.cos(2.0 * math.pi * cycles * t))


def sample_clip(
    clip: AnimationClip,
    progress: float,
) -> tuple[SpritePose, tuple[EffectInstance, ...]]:
    from owl.viz.action_animation import sample_action_animation

    return sample_action_animation(clip.action, clip.context, progress)
