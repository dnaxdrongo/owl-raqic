"""Observer-Window Life visualization package."""

from owl.viz.frame_scheduler import VisualFrameScheduler, VisualScheduleMode
from owl.viz.overlays import OverlaySpec, default_overlays
from owl.viz.palettes import (
    action_palette,
    food_palette,
    integration_palette,
    patch_palette,
    signal_overlay_palette,
    toxin_palette,
    type_palette,
)
from owl.viz.pygame_viewer import PygameViewer, create_viewer
from owl.viz.trait_color import TraitColor, TraitVector, encode_trait_hex
from owl.viz.visual_snapshot import VisualSnapshot

__all__ = [
    "TraitColor",
    "TraitVector",
    "VisualFrameScheduler",
    "VisualScheduleMode",
    "VisualSnapshot",
    "encode_trait_hex",
    "OverlaySpec",
    "PygameViewer",
    "action_palette",
    "create_viewer",
    "default_overlays",
    "food_palette",
    "integration_palette",
    "patch_palette",
    "signal_overlay_palette",
    "toxin_palette",
    "type_palette",
]
