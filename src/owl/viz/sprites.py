from __future__ import annotations

from dataclasses import dataclass

from owl.core.actions import Action


@dataclass(frozen=True)
class SpriteSpec:
    """Compatibility name for action-effect metadata.

    Stable OW body identity is encoded by SpriteDescriptor/TraitColor.  These
    specifications describe only the current action effect and its visual grammar.
    """

    action: Action
    name: str
    glyph: str
    shape: str
    color: tuple[int, int, int, int]
    secondary: tuple[int, int, int, int]
    ring: str = "none"
    trail: bool = False
    pulse: bool = False
    priority: int = 0
    ttl_ticks: int = 1
    description: str = ""
    duration: float = 1.0
    directional: bool = False
    fallback_mode: str = "nondirectional"
    effect_family: str = "ambient"


ActionEffectSpec = SpriteSpec


def build_sprite_specs() -> dict[Action, SpriteSpec]:
    palette = {
        "rest": (130, 155, 190, 150),
        "sense": (60, 220, 255, 230),
        "move": (225, 238, 255, 220),
        "feed": (65, 235, 130, 240),
        "comm": (70, 145, 255, 240),
        "inhibit": (255, 110, 45, 245),
        "integrate": (170, 110, 255, 240),
        "repair": (255, 220, 70, 245),
        "repro": (255, 90, 220, 245),
        "ingest": (255, 45, 80, 245),
        "topo": (80, 255, 220, 245),
        "pursue": (220, 60, 75, 245),
    }
    specs = {
        Action.REST: SpriteSpec(
            Action.REST,
            "rest",
            "dot",
            "circle",
            palette["rest"],
            palette["rest"],
            description="quiet/resting",
            effect_family="breathing",
        ),
        Action.SENSE: SpriteSpec(
            Action.SENSE,
            "sense",
            "eye",
            "circle",
            palette["sense"],
            (180, 245, 255, 210),
            pulse=True,
            description="sensing/local observation",
            effect_family="scan",
        ),
        Action.FEED: SpriteSpec(
            Action.FEED,
            "feed",
            "leaf",
            "circle",
            palette["feed"],
            (180, 255, 200, 220),
            pulse=True,
            priority=4,
            description="feeding/resource intake",
            effect_family="nutrient",
        ),
        Action.COMMUNICATE: SpriteSpec(
            Action.COMMUNICATE,
            "communicate",
            "broadcast",
            "circle",
            palette["comm"],
            (140, 190, 255, 220),
            ring="signal",
            pulse=True,
            priority=3,
            description="emitting signal",
            effect_family="signal",
        ),
        Action.INHIBIT: SpriteSpec(
            Action.INHIBIT,
            "inhibit",
            "bar",
            "diamond",
            palette["inhibit"],
            (255, 190, 120, 220),
            pulse=True,
            priority=5,
            description="suppressing/inhibiting neighbor",
            directional=True,
            effect_family="inhibition",
        ),
        Action.INTEGRATE: SpriteSpec(
            Action.INTEGRATE,
            "integrate",
            "nodes",
            "circle",
            palette["integrate"],
            (210, 180, 255, 220),
            ring="coherence",
            priority=3,
            description="integration/coherence action",
            effect_family="integration",
        ),
        Action.REPAIR: SpriteSpec(
            Action.REPAIR,
            "repair",
            "plus",
            "circle",
            palette["repair"],
            (255, 245, 180, 220),
            ring="heal",
            pulse=True,
            priority=4,
            description="repair/healing",
            effect_family="repair",
        ),
        Action.REPRODUCE: SpriteSpec(
            Action.REPRODUCE,
            "reproduce",
            "bud",
            "hex",
            palette["repro"],
            (255, 180, 240, 220),
            pulse=True,
            priority=6,
            ttl_ticks=3,
            description="child creation",
            directional=True,
            effect_family="birth",
        ),
        Action.INGEST: SpriteSpec(
            Action.INGEST,
            "ingest",
            "bite",
            "diamond",
            palette["ingest"],
            (255, 155, 170, 220),
            pulse=True,
            priority=6,
            description="predation/ingestion",
            directional=True,
            effect_family="ingestion",
        ),
        Action.EXPEL: SpriteSpec(
            Action.EXPEL,
            "expel",
            "burst",
            "triangle",
            (255, 150, 65, 240),
            (255, 210, 140, 220),
            pulse=True,
            priority=5,
            description="expel/release",
            directional=True,
            effect_family="expulsion",
        ),
        Action.SPLIT: SpriteSpec(
            Action.SPLIT,
            "split",
            "split",
            "hex",
            palette["topo"],
            (170, 255, 240, 220),
            pulse=True,
            priority=5,
            description="topological split",
            directional=True,
            effect_family="topology",
        ),
        Action.MERGE: SpriteSpec(
            Action.MERGE,
            "merge",
            "merge",
            "hex",
            (185, 150, 255, 240),
            (220, 210, 255, 220),
            pulse=True,
            priority=5,
            description="topological merge",
            directional=True,
            effect_family="topology",
        ),
        Action.FLEE: SpriteSpec(
            Action.FLEE,
            "flee",
            "arrow_away",
            "triangle",
            (255, 190, 50, 240),
            (255, 235, 160, 220),
            trail=True,
            priority=4,
            description="fleeing threat",
            directional=True,
            effect_family="flight",
        ),
        Action.PURSUE: SpriteSpec(
            Action.PURSUE,
            "pursue",
            "chevron",
            "triangle",
            palette["pursue"],
            (255, 120, 130, 220),
            trail=True,
            priority=4,
            description="pursuing target",
            directional=True,
            effect_family="pursuit",
        ),
    }
    move_glyphs = {
        Action.MOVE_N: "arrow_n",
        Action.MOVE_S: "arrow_s",
        Action.MOVE_E: "arrow_e",
        Action.MOVE_W: "arrow_w",
        Action.MOVE_NE: "arrow_ne",
        Action.MOVE_NW: "arrow_nw",
        Action.MOVE_SE: "arrow_se",
        Action.MOVE_SW: "arrow_sw",
    }
    for action, glyph in move_glyphs.items():
        specs[action] = SpriteSpec(
            action,
            action.name.lower(),
            glyph,
            "triangle",
            palette["move"],
            (190, 210, 255, 180),
            trail=True,
            priority=2,
            description=f"movement {action.name}",
            directional=True,
            fallback_mode="attempted_direction",
            effect_family="movement",
        )
    missing = set(Action) - set(specs)
    if missing:
        raise RuntimeError(f"missing SpriteSpec entries: {missing}")
    return specs


SPRITE_SPECS = build_sprite_specs()
ACTION_EFFECT_SPECS = SPRITE_SPECS


def validate_all_actions_covered() -> None:
    missing = set(Action) - set(SPRITE_SPECS)
    if missing:
        raise RuntimeError(f"missing action effect specifications: {missing}")


validate_all_actions_covered()
