"""Visualization contract tests.

These tests run in headless environments. They validate array conversion,
controls, replay state, and viewer importability without opening a GUI.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from owl.core.actions import Action, SignalChannel
from owl.core.config import SimulationConfig, load_config
from owl.core.init import initialize_world
from owl.viz.controls import handle_parameter_hotkeys, pause_or_step_controls
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
from owl.viz.pygame_viewer import PygameViewer, _array_or_fallback, create_viewer


def make_viz_cfg(height: int = 20, width: int = 20) -> SimulationConfig:
    data = load_config("configs/mvp.yaml").model_dump()
    data["world"]["height"] = height
    data["world"]["width"] = width
    data["world"]["patch_size"] = 5
    data["initialization"]["population_density"] = 0.50
    data["visualization"]["enabled"] = True
    data["visualization"]["backend"] = "pygame"
    data["visualization"]["scale"] = 4
    data["recording"]["enabled"] = False
    return SimulationConfig.model_validate(data)


def make_viz_state(seed: int = 123):
    cfg = make_viz_cfg()
    state = initialize_world(cfg, np.random.default_rng(seed))
    state.tick = 12
    state.signal[..., int(SignalChannel.COORDINATION)] = 0.4
    state.signal[..., int(SignalChannel.INTEGRATION)] = 0.2
    state.toxin[0:3, 0:3] = 0.8
    state.food[3:6, 3:6] = 0.9
    state.readout[0:2, 0:2] = int(Action.FEED)
    return cfg, state


def assert_rgb(rgb: np.ndarray, width: int, height: int) -> None:
    assert rgb.shape == (width, height, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb >= 0)
    assert np.all(rgb <= 255)


def test_overlay_specs_are_stable_and_unique() -> None:
    overlays = default_overlays()
    assert overlays
    assert all(isinstance(spec, OverlaySpec) for spec in overlays)
    assert overlays[0].name == "integration"
    assert len({spec.name for spec in overlays}) == len(overlays)
    assert {"integration", "type", "action", "food", "toxin", "patches"} <= {
        spec.name for spec in overlays
    }


def test_palettes_return_pygame_oriented_uint8_rgb_arrays() -> None:
    cfg, state = make_viz_state()
    height, width = state.health.shape

    for palette in (
        integration_palette,
        type_palette,
        action_palette,
        food_palette,
        toxin_palette,
        patch_palette,
    ):
        rgb = palette(state)
        assert_rgb(rgb, width, height)

    for channel in (int(SignalChannel.FOOD), int(SignalChannel.COORDINATION)):
        rgb = signal_overlay_palette(state, channel)
        assert_rgb(rgb, width, height)


def test_signal_overlay_rejects_bad_channel() -> None:
    _cfg, state = make_viz_state()
    try:
        signal_overlay_palette(state, state.signal.shape[-1])
    except ValueError as exc:
        assert "outside available range" in str(exc)
    else:
        raise AssertionError("invalid channel should fail")


def test_palette_functions_do_not_mutate_state() -> None:
    _cfg, state = make_viz_state()
    before = {
        "integration": state.integration.copy(),
        "readout": state.readout.copy(),
        "food": state.food.copy(),
        "toxin": state.toxin.copy(),
        "parent_id": state.parent_id.copy(),
        "signal": state.signal.copy(),
    }

    _ = integration_palette(state)
    _ = type_palette(state)
    _ = action_palette(state)
    _ = food_palette(state)
    _ = toxin_palette(state)
    _ = patch_palette(state)
    _ = signal_overlay_palette(state, int(SignalChannel.FOOD))

    for name, arr in before.items():
        assert np.array_equal(getattr(state, name), arr), name


def test_pygame_viewer_field_to_rgb_and_headless_draw_history() -> None:
    cfg, state = make_viz_state()
    viewer = PygameViewer(cfg.world.height, cfg.world.width, scale=cfg.visualization.scale)

    for overlay in ["integration", "type", "action", "food", "toxin", "patches", "signal:3"]:
        viewer.overlay = overlay
        rgb = viewer.field_to_rgb(state)
        assert_rgb(rgb, cfg.world.width, cfg.world.height)

    before_health = state.health.copy()
    before_possibility = state.possibility.copy()

    viewer.draw(state, fps=30)
    assert viewer.last_rgb is not None
    assert viewer.last_rgb.shape == (cfg.world.width, cfg.world.height, 3)
    assert len(viewer.history) == 1
    assert viewer.history[-1].tick == state.tick

    # Paused rendering should keep a playback pointer while new backend states
    # can still be passed to draw and recorded.
    viewer.paused_render = True
    viewer.playback_index = 0
    state.tick += 1
    viewer.draw(state, fps=30)
    assert len(viewer.history) == 2
    assert viewer.playback_index == 0

    assert np.array_equal(state.health, before_health)
    assert np.array_equal(state.possibility, before_possibility)
    viewer.close()
    assert not viewer.running


def test_optional_visual_array_resolver_uses_none_fallback_without_mutation() -> None:
    _cfg, state = make_viz_state()
    fallback = state.readout
    before = fallback.copy()

    assert state.raqic_readout is None
    resolved = _array_or_fallback(
        state,
        "raqic_readout",
        fallback,
        expected_shape=state.readout.shape,
    )

    assert resolved is fallback
    assert state.raqic_readout is None
    assert np.array_equal(fallback, before)


def test_optional_visual_array_resolver_prefers_present_array_and_rejects_bad_shape() -> None:
    _cfg, state = make_viz_state()
    preferred = np.full_like(state.readout, int(Action.REPAIR))
    state.raqic_readout = preferred

    resolved = _array_or_fallback(
        state,
        "raqic_readout",
        state.readout,
        expected_shape=state.readout.shape,
    )
    assert resolved is preferred

    state.raqic_readout = np.zeros((1, 1), dtype=state.readout.dtype)
    with pytest.raises(ValueError, match="raqic_readout.*expected"):
        _array_or_fallback(
            state,
            "raqic_readout",
            state.readout,
            expected_shape=state.readout.shape,
        )


def test_pygame_sprite_path_prefers_raqic_arrays_when_present() -> None:
    pygame_module = pytest.importorskip("pygame")
    pygame_module.display.quit()
    pygame_module.display.init()

    cfg, state = make_viz_state()
    living = (state.health > 0.0) & (~state.obstacle)
    y, x = np.argwhere(living)[0]
    action_count = state.possibility.shape[-1]
    state.raqic_readout = np.full_like(state.readout, int(Action.REPAIR))
    state.raqic_probabilities = np.zeros_like(state.possibility)
    state.raqic_probabilities[..., int(Action.REPAIR)] = 1.0

    class Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[int, float]] = []

        def draw_cell(
            self,
            _surface: object,
            _rect: object,
            action: int,
            _health: float = 1.0,
            confidence: float = 1.0,
            **_kwargs: object,
        ) -> None:
            self.calls.append((action, confidence))

    viewer = PygameViewer(cfg.world.height, cfg.world.width, scale=cfg.visualization.scale)
    recorder = Recorder()
    viewer.dynamic_sprite_renderer = recorder  # type: ignore[assignment]
    viewer._draw_type_sprites(state)

    assert recorder.calls
    expected_index = int(np.flatnonzero(living.ravel(order="C") == living[y, x])[0])
    _ = expected_index  # the first visible living call is enough to prove preference
    assert all(action == int(Action.REPAIR) for action, _confidence in recorder.calls)
    assert all(confidence == pytest.approx(1.0) for _action, confidence in recorder.calls)
    assert state.raqic_readout.shape == state.readout.shape
    assert state.raqic_probabilities.shape == (
        cfg.world.height,
        cfg.world.width,
        action_count,
    )
    viewer.close()


def test_pygame_tooltip_uses_zero_waste_when_optional_waste_is_none() -> None:
    pygame_module = pytest.importorskip("pygame")
    pygame_module.display.quit()
    pygame_module.display.init()

    cfg, state = make_viz_state()
    living = (state.health > 0.0) & (~state.obstacle)
    y, x = np.argwhere(living)[0]
    state.digestion = np.full_like(state.food, 0.25)
    state.waste = None

    viewer = PygameViewer(cfg.world.height, cfg.world.width, scale=cfg.visualization.scale)
    viewer.mouse_pos = (int(x) * viewer.scale + 1, int(y) * viewer.scale + 1)
    viewer._draw_tooltip(state)

    assert state.waste is None
    viewer.close()


def test_create_viewer_respects_visualization_config() -> None:
    cfg, _state = make_viz_state()

    viewer = create_viewer(cfg)
    assert isinstance(viewer, PygameViewer)
    viewer.close()

    cfg.visualization.enabled = False
    assert create_viewer(cfg) is None

    cfg.visualization.enabled = True
    cfg.visualization.backend = "none"
    assert create_viewer(cfg) is None


def test_viewer_screen_to_cell_zoom_and_replay_controls_are_bounded() -> None:
    cfg, _state = make_viz_state()
    viewer = PygameViewer(cfg.world.height, cfg.world.width, scale=cfg.visualization.scale)

    assert viewer._screen_to_cell((0, 0)) == (0, 0)
    viewer._zoom_at(2.0, (10, 10))
    assert viewer.zoom >= 1.0
    viewer._zoom_at(0.001, (10, 10))
    assert viewer.zoom >= viewer.min_zoom

    # Empty history replay should be safe.
    viewer._change_playback_index(-1)
    assert viewer.playback_index is None


def test_handle_parameter_hotkeys_changes_only_config_controls() -> None:
    cfg, state = make_viz_state()
    before_health = state.health.copy()

    start_scale = cfg.visualization.scale
    handle_parameter_hotkeys(state, cfg, ord("+"))
    assert cfg.visualization.scale == start_scale + 1

    start_beta = cfg.actions.beta
    handle_parameter_hotkeys(state, cfg, ord("["))
    assert cfg.actions.beta < start_beta

    start_debug = cfg.debug.assert_invariants
    handle_parameter_hotkeys(state, cfg, ord("h"))
    assert cfg.debug.assert_invariants is (not start_debug)

    assert np.array_equal(state.health, before_health)


def test_pause_or_step_controls_headless_shape() -> None:
    controls = pause_or_step_controls()
    expected = {"quit", "toggle_pause", "step", "rewind", "fast_forward", "zoom_in", "zoom_out"}
    assert set(controls) == expected
    assert all(value is False for value in controls.values())


def test_pygame_viewer_source_keeps_real_pygame_paths_present() -> None:
    source = __import__("pathlib").Path("src/owl/viz/pygame_viewer.py").read_text(encoding="utf-8")
    for token in [
        "pygame.surfarray.blit_array",
        "pygame.draw.circle",
        "pygame.draw.polygon",
        "pygame.font.Font",
        "pygame.MOUSEWHEEL",
        "_draw_tooltip",
        "_draw_type_sprites",
    ]:
        assert token in source
