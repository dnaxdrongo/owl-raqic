from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from owl.core.actions import MOVE_DELTAS, Action
from owl.viz.animation import (
    AnimationClip,
    EffectInstance,
    SpritePose,
    clamp_progress,
    ease_in_out,
    ease_out_back,
    pulse,
)
from owl.viz.event_bus import VisualEvent, VisualEventType, match_event_for_action
from owl.viz.visual_snapshot import VisualSnapshot

_MOVE_ACTIONS = frozenset(MOVE_DELTAS)
_SIGNAL_COLORS: tuple[tuple[int, int, int, int], ...] = (
    (85, 235, 130, 225),
    (250, 210, 65, 230),
    (255, 90, 90, 230),
    (80, 170, 255, 230),
    (255, 135, 70, 230),
    (255, 105, 220, 230),
    (185, 130, 255, 230),
    (90, 245, 230, 230),
)


@dataclass(frozen=True)
class ActionContext:
    ow_id: int
    tick: int
    source: tuple[float, float]
    target: tuple[float, float] | None
    attempted_delta: tuple[float, float] | None
    successful_move: bool | None
    channel: int
    intensity: float
    event_type: VisualEventType | None
    boundary_mode: str
    world_shape: tuple[int, int]
    visual_seed: int = 0


def _seed_for(context: ActionContext, salt: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(context.visual_seed).encode())
    digest.update(str(context.tick).encode())
    digest.update(str(context.ow_id).encode())
    digest.update(salt.encode())
    return int.from_bytes(digest.digest(), "little", signed=False)


def shortest_toroidal_delta(
    source: tuple[float, float],
    target: tuple[float, float],
    shape: tuple[int, int],
) -> tuple[float, float]:
    dy = float(target[0] - source[0])
    dx = float(target[1] - source[1])
    height, width = float(shape[0]), float(shape[1])
    if height > 0 and abs(dy) > height / 2.0:
        dy -= math.copysign(height, dy)
    if width > 0 and abs(dx) > width / 2.0:
        dx -= math.copysign(width, dx)
    return dy, dx


def _event_target(event: VisualEvent | None) -> tuple[float, float] | None:
    if event is None or event.target_y < 0 or event.target_x < 0:
        return None
    return float(event.target_y), float(event.target_x)


def resolve_action_context(
    previous: VisualSnapshot | None,
    current: VisualSnapshot,
    events: Sequence[VisualEvent],
    ow_id: int,
    action: Action,
    *,
    visual_seed: int = 0,
) -> ActionContext:
    current_position = current.position_of(ow_id)
    previous_position = previous.position_of(ow_id) if previous is not None else None
    source_positions = tuple(
        position for position in (previous_position, current_position) if position is not None
    )
    event = match_event_for_action(
        events,
        ow_id,
        action,
        source_positions=source_positions,
    )
    event_target = _event_target(event)

    if previous_position is not None:
        source = (float(previous_position[0]), float(previous_position[1]))
    elif event is not None:
        source = (float(event.y), float(event.x))
    elif current_position is not None:
        source = (float(current_position[0]), float(current_position[1]))
    else:
        source = (0.0, 0.0)

    target: tuple[float, float] | None = event_target
    successful_move: bool | None = None
    attempted_delta: tuple[float, float] | None = None

    if action in _MOVE_ACTIONS:
        delta = MOVE_DELTAS[action]
        attempted_delta = (float(delta[0]), float(delta[1]))
        if previous_position is not None and current_position is not None:
            target = (float(current_position[0]), float(current_position[1]))
            successful_move = previous_position != current_position
        elif event_target is not None:
            successful_move = (
                event.event_type != VisualEventType.MOVEMENT_REJECTED if event else None
            )
        else:
            successful_move = False
    elif action in (Action.FLEE, Action.PURSUE):
        if previous_position is not None and current_position is not None:
            target = (float(current_position[0]), float(current_position[1]))
            successful_move = previous_position != current_position
    elif action == Action.FEED and event is None:
        target = None

    if event is not None:
        intensity = max(0.0, float(event.intensity))
        channel = int(event.channel)
        event_type = event.event_type
    else:
        intensity = 1.0
        channel = -1
        event_type = None

    return ActionContext(
        ow_id=int(ow_id),
        tick=int(current.tick),
        source=source,
        target=target,
        attempted_delta=attempted_delta,
        successful_move=successful_move,
        channel=channel,
        intensity=intensity,
        event_type=event_type,
        boundary_mode=str(current.boundary_mode),
        world_shape=current.world_shape,
        visual_seed=int(visual_seed),
    )


def _movement_delta(context: ActionContext) -> tuple[float, float]:
    if context.target is not None:
        if context.boundary_mode == "toroidal":
            return shortest_toroidal_delta(context.source, context.target, context.world_shape)
        return context.target[0] - context.source[0], context.target[1] - context.source[1]
    return context.attempted_delta or (0.0, 0.0)


def movement_pose(context: ActionContext, progress: float) -> SpritePose:
    t = ease_in_out(progress)
    dy, dx = _movement_delta(context)
    angle = math.degrees(math.atan2(dy, dx)) if dy or dx else 0.0
    if context.successful_move:
        position = (context.source[0] + dy * t, context.source[1] + dx * t)
        stretch = 1.0 + 0.18 * math.sin(math.pi * t)
        return SpritePose(position, angle, 1.0, 1.0, (stretch, 1.0 / stretch))
    shake = math.sin(5.0 * math.pi * t) * (1.0 - t) * 0.18
    magnitude = math.hypot(dy, dx) or 1.0
    position = (
        context.source[0] + (dy / magnitude) * shake,
        context.source[1] + (dx / magnitude) * shake,
    )
    return SpritePose(position, angle, 1.0, 1.0, (1.0 - abs(shake), 1.0 + abs(shake)))


def _effect(
    kind: str,
    context: ActionContext,
    progress: float,
    color: tuple[int, int, int, int],
    *,
    target: tuple[float, float] | None = None,
    layer: int = 40,
    intensity: float | None = None,
    label: str = "",
) -> EffectInstance:
    return EffectInstance(
        kind=kind,
        source=context.source,
        target=target,
        color=color,
        progress=clamp_progress(progress),
        intensity=float(context.intensity if intensity is None else intensity),
        layer=int(layer),
        seed=_seed_for(context, kind),
        channel=int(context.channel),
        label=label,
    )


def rest_effects(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    return (_effect("breath", context, progress, (150, 190, 230, 105), layer=25),)


def _sense(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    return (_effect("scan_arc", context, progress, (80, 225, 255, 220)),)


def _feed(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    return (
        _effect("nutrient_spiral", context, progress, (110, 245, 115, 235), target=context.target),
    )


def _communicate(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    color = (
        _SIGNAL_COLORS[context.channel % len(_SIGNAL_COLORS)]
        if context.channel >= 0
        else (80, 170, 255, 230)
    )
    return (_effect("signal_wave", context, progress, color, target=context.target),)


def _inhibit(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    kind = "inhibit_beam" if context.target is not None else "inhibit_field"
    return (_effect(kind, context, progress, (255, 110, 50, 235), target=context.target),)


def _integrate(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    return (
        _effect("integration_orbit", context, progress, (175, 120, 255, 235)),
        _effect("coherence_halo", context, progress, (95, 235, 230, 160), layer=24),
    )


def _repair(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    return (
        _effect("repair_stitch", context, progress, (255, 220, 75, 240)),
        _effect("repair_sparks", context, progress, (255, 250, 185, 210)),
    )


def _reproduce(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    kind = "birth_bridge" if context.target is not None else "birth_pulse"
    return (_effect(kind, context, progress, (255, 95, 220, 240), target=context.target),)


def _ingest(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    kind = "ingest_lunge" if context.target is not None else "ingest_bite"
    return (
        _effect(kind, context, progress, (255, 65, 85, 240), target=context.target),
        _effect(
            "resource_transfer", context, progress, (125, 245, 135, 225), target=context.target
        ),
    )


def _expel(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    return (_effect("expel_burst", context, progress, (255, 160, 70, 235), target=context.target),)


def _split(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    return (
        _effect("split_membrane", context, progress, (80, 245, 220, 235), target=context.target),
    )


def _merge(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    return (
        _effect("merge_bridge", context, progress, (185, 145, 255, 235), target=context.target),
    )


def _flee(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    return (_effect("flee_trail", context, progress, (255, 205, 65, 220), target=context.target),)


def _pursue(context: ActionContext, progress: float) -> tuple[EffectInstance, ...]:
    return (
        _effect("pursuit_trail", context, progress, (235, 70, 85, 220), target=context.target),
        _effect("target_reticle", context, progress, (255, 150, 155, 210), target=context.target),
    )


_EFFECT_BUILDERS: dict[Action, Callable[[ActionContext, float], tuple[EffectInstance, ...]]] = {
    Action.REST: rest_effects,
    Action.SENSE: _sense,
    Action.FEED: _feed,
    Action.COMMUNICATE: _communicate,
    Action.INHIBIT: _inhibit,
    Action.INTEGRATE: _integrate,
    Action.REPAIR: _repair,
    Action.REPRODUCE: _reproduce,
    Action.INGEST: _ingest,
    Action.EXPEL: _expel,
    Action.SPLIT: _split,
    Action.MERGE: _merge,
    Action.FLEE: _flee,
    Action.PURSUE: _pursue,
}
for _move_action in _MOVE_ACTIONS:
    _EFFECT_BUILDERS[_move_action] = lambda context, progress: (
        _effect(
            "move_trail" if context.successful_move else "movement_recoil",
            context,
            progress,
            (220, 235, 255, 180) if context.successful_move else (255, 100, 100, 220),
            target=context.target,
        ),
    )


def animation_for_action(action: Action, context: ActionContext) -> AnimationClip:
    directional = action in _MOVE_ACTIONS or action in {
        Action.FLEE,
        Action.PURSUE,
        Action.REPRODUCE,
        Action.INGEST,
        Action.EXPEL,
        Action.SPLIT,
        Action.MERGE,
    }
    duration = 1.0
    family = _EFFECT_BUILDERS[action].__name__.lstrip("_")
    return AnimationClip(action, duration, directional, context, family)


def sample_action_animation(
    action: Action,
    context: ActionContext,
    progress: float,
) -> tuple[SpritePose, tuple[EffectInstance, ...]]:
    t = clamp_progress(progress)
    if action in _MOVE_ACTIONS or action in {Action.FLEE, Action.PURSUE}:
        pose = movement_pose(context, t)
    elif action == Action.INGEST and context.target is not None:
        dy, dx = _movement_delta(context)
        lunge = math.sin(math.pi * t) * 0.28
        magnitude = math.hypot(dy, dx) or 1.0
        angle = math.degrees(math.atan2(dy, dx))
        pose = SpritePose(
            (
                context.source[0] + dy / magnitude * lunge,
                context.source[1] + dx / magnitude * lunge,
            ),
            angle,
            1.0 + 0.10 * pulse(t),
        )
    elif action == Action.REPRODUCE:
        scale = 1.0 + 0.12 * ease_out_back(math.sin(math.pi * t))
        pose = SpritePose(context.source, 0.0, scale)
    elif action == Action.REPAIR:
        pose = SpritePose(context.source, 0.0, 1.0 + 0.05 * pulse(t, 2.0))
    else:
        pose = SpritePose(context.source, 0.0, 1.0 + 0.04 * pulse(t))
    return pose, _EFFECT_BUILDERS[action](context, t)


def validate_all_actions_covered() -> None:
    missing = set(Action) - set(_EFFECT_BUILDERS)
    if missing:
        raise RuntimeError(
            f"missing action animation mappings: {sorted(action.name for action in missing)}"
        )


validate_all_actions_covered()
