from __future__ import annotations

from dataclasses import dataclass

from owl.core.actions import Action
from owl.viz.sprites import SPRITE_SPECS
from owl.viz.trait_color import TraitColor, TraitVector, encode_trait_hex


@dataclass(frozen=True)
class SpriteDescriptor:
    ow_id: int
    trait_color: TraitColor
    archetype: str
    membrane_pattern: str
    cilia_level: int
    spike_level: int
    nucleus_style: str
    lineage_marker: int
    developmental_stage: int


@dataclass(frozen=True)
class SpriteStatus:
    health_fraction: float
    resource_fraction: float
    toxin_fraction: float
    starvation_fraction: float
    integration: float
    phase: float
    confidence: float
    entropy: float
    selected: bool
    parent_pressure: float = 0.0
    age_fraction: float = 0.0
    reproduction_ready: bool = False


@dataclass(frozen=True)
class SpriteState:
    descriptor: SpriteDescriptor
    status: SpriteStatus
    action: Action
    communication_channel: int = -1
    topology_marker: str = ""
    debug_marker: str = ""
    invalid_action: bool = False

    @property
    def body_shape(self) -> str:
        return self.descriptor.archetype

    @property
    def body_color(self) -> tuple[int, int, int, int]:
        r, g, b = self.descriptor.trait_color.rendered_rgb
        return r, g, b, self.body_alpha

    @property
    def body_alpha(self) -> int:
        return int(75 + 180 * self.status.health_fraction)

    @property
    def health_fraction(self) -> float:
        return self.status.health_fraction

    @property
    def resource_fraction(self) -> float:
        return self.status.resource_fraction

    @property
    def confidence(self) -> float:
        return self.status.confidence

    @property
    def entropy(self) -> float:
        return self.status.entropy

    @property
    def coherence(self) -> float:
        return self.status.integration

    @property
    def stress(self) -> float:
        return max(
            self.status.toxin_fraction,
            self.status.starvation_fraction,
            1.0 - self.status.health_fraction,
        )

    @property
    def action_glyph(self) -> str:
        return "unknown" if self.invalid_action else SPRITE_SPECS[self.action].glyph

    @property
    def ring_style(self) -> str:
        if self.invalid_action:
            return "invalid"
        if self.status.selected:
            return "selected"
        if self.status.entropy > 1.5:
            return "uncertain"
        if self.status.confidence > 0.9:
            return "confident"
        if self.status.integration > 0.75:
            return "coherent"
        if self.status.parent_pressure > 0.75:
            return "parent_pressure"
        return SPRITE_SPECS[self.action].ring

    @property
    def ring_alpha(self) -> int:
        return int(
            30
            + 225
            * max(
                self.status.confidence,
                self.status.integration,
                self.status.parent_pressure,
            )
        )

    @property
    def pulse(self) -> bool:
        return bool(SPRITE_SPECS[self.action].pulse or self.status.selected)

    @property
    def trail(self) -> bool:
        return bool(SPRITE_SPECS[self.action].trail)

    @property
    def hollow_center(self) -> bool:
        return self.status.resource_fraction < 0.20

    @property
    def cracked_outline(self) -> bool:
        return self.status.health_fraction < 0.25

    @property
    def hazard_outline(self) -> bool:
        return self.status.toxin_fraction > 0.55 or self.status.starvation_fraction > 0.60

    @property
    def developmental_stage(self) -> int:
        return self.descriptor.developmental_stage

    @property
    def lineage_marker(self) -> int:
        return self.descriptor.lineage_marker

    @property
    def age_fraction(self) -> float:
        return self.status.age_fraction

    @property
    def parent_pressure(self) -> float:
        return self.status.parent_pressure

    @property
    def phase_notch(self) -> float:
        return self.status.phase

    @property
    def reproduction_ready(self) -> bool:
        return self.status.reproduction_ready

    @property
    def selected(self) -> bool:
        return self.status.selected


def _safe_action(value: int) -> tuple[Action, bool]:
    try:
        return Action(int(value)), False
    except (ValueError, TypeError, OverflowError):
        return Action.REST, True


def _clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def descriptor_from_traits(
    *,
    ow_id: int,
    traits: TraitVector,
    developmental_stage: int,
    lineage_marker: int,
) -> SpriteDescriptor:
    predatory = traits.predatory_pressure
    cooperative = traits.cooperative_ecology
    mobile = traits.cognition_exploration
    communication = traits.cross_scale_communication
    if predatory > 0.67:
        archetype = "spined"
    elif cooperative > 0.67:
        archetype = "lobed"
    elif mobile > 0.67:
        archetype = "ciliated"
    elif communication > 0.67:
        archetype = "radiant"
    else:
        archetype = "amoeba"
    membrane_pattern = (
        "banded" if traits.resilience > 0.66 else "dotted" if cooperative > 0.60 else "plain"
    )
    return SpriteDescriptor(
        ow_id=int(ow_id),
        trait_color=encode_trait_hex(traits),
        archetype=archetype,
        membrane_pattern=membrane_pattern,
        cilia_level=int(round(mobile * 3.0)),
        spike_level=int(round(predatory * 3.0)),
        nucleus_style="double" if traits.cognition_exploration > 0.75 else "single",
        lineage_marker=int(lineage_marker),
        developmental_stage=int(developmental_stage),
    )


def build_sprite_state(
    *,
    action: int,
    health: float,
    resource: float,
    confidence: float = 1.0,
    entropy: float = 0.0,
    coherence: float = 0.0,
    toxin: float = 0.0,
    starvation: float = 0.0,
    communication_channel: int = -1,
    debug_marker: str = "",
    developmental_stage: int = 0,
    lineage_marker: int = -1,
    age_fraction: float = 0.0,
    parent_pressure: float = 0.0,
    phase: float = 0.0,
    reproduction_ready: bool = False,
    selected: bool = False,
    ow_id: int = -1,
    traits: TraitVector | None = None,
) -> SpriteState:
    safe_action, invalid = _safe_action(action)
    if invalid:
        debug_marker = debug_marker or "UNKNOWN_ACTION"
    traits = traits or TraitVector(0.35, 0.45, 0.50, 0.60, 0.55, 0.50)
    descriptor = descriptor_from_traits(
        ow_id=ow_id,
        traits=traits,
        developmental_stage=developmental_stage,
        lineage_marker=lineage_marker,
    )
    status = SpriteStatus(
        health_fraction=_clamp(health),
        resource_fraction=_clamp(resource),
        toxin_fraction=_clamp(toxin),
        starvation_fraction=_clamp(starvation),
        integration=_clamp(coherence),
        phase=float(phase) % (2.0 * 3.141592653589793),
        confidence=_clamp(confidence),
        entropy=max(0.0, float(entropy)),
        selected=bool(selected),
        parent_pressure=_clamp(parent_pressure),
        age_fraction=_clamp(age_fraction),
        reproduction_ready=bool(reproduction_ready),
    )
    spec = SPRITE_SPECS[safe_action]
    topology = spec.glyph if safe_action in (Action.MERGE, Action.SPLIT, Action.EXPEL) else ""
    return SpriteState(
        descriptor=descriptor,
        status=status,
        action=safe_action,
        communication_channel=int(communication_channel),
        topology_marker=topology,
        debug_marker=str(debug_marker),
        invalid_action=invalid,
    )
