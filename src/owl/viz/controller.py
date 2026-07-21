from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from owl.viz.camera import CameraState, fit_world
from owl.viz.event_bus import (
    VisualEvent,
    VisualEventBuffer,
    VisualEventType,
    events_from_event_records,
    events_from_state,
    events_from_topology_buffer,
)
from owl.viz.frame_model import VisualFrame, VisualSelection
from owl.viz.frame_scheduler import VisualFrameScheduler, VisualScheduleMode
from owl.viz.gpu_compositor import compose_frame_device
from owl.viz.scene import build_visual_scene
from owl.viz.themes import get_theme
from owl.viz.visual_snapshot import VisualSnapshot, snapshot_from_device_state


class NullVisualBackend:
    def submit(self, frame: VisualFrame) -> None:
        del frame

    def close(self) -> None:
        return None


@dataclass
class VisualControllerMetrics:
    frames_submitted: int = 0
    frames_dropped: int = 0
    compositor_ms: float = 0.0
    transfer_ms: float = 0.0
    snapshot_ms: float = 0.0
    scene_build_ms: float = 0.0
    render_ms: float = 0.0
    original_events: int = 0
    rendered_events: int = 0
    aggregated_events: int = 0
    event_overflow: int = 0
    critical_event_drops: int = 0
    sprite_count: int = 0
    environment_count: int = 0
    effect_count: int = 0
    lod_changes: int = 0
    snapshots_captured: int = 0


class VisualController:
    """One-way scientific-to-visual interpretation boundary.

    The controller copies selected arrays to immutable host snapshots.  All
    camera, sprite, animation, color, HUD, and subframe work happens after that
    boundary and cannot mutate scientific state or consume scientific RNG.
    """

    def __init__(
        self,
        *,
        backend_name: str,
        render_every: int,
        event_capacity: int,
        clutter_budget: int,
        adaptive: bool,
        max_slowdown_fraction: float,
        theme: str,
        output_dir: str | Path = "results/visual_frames",
        schedule_mode: str = "live_adaptive",
        frames_per_tick: int = 1,
        tick_stride: int = 1,
        fail_on_drop: bool = True,
        renderer_mode: str = "interpretability",
        world_shape: tuple[int, int] | None = None,
        window_size: tuple[int, int] = (1920, 1080),
        sidebar_width: int = 320,
        resizable: bool = True,
        fps: int = 30,
        camera_mode: str = "fit",
        min_zoom: float = 0.5,
        max_zoom: float = 64.0,
        selected_ow_id: int | None = None,
        accessibility_mode: str = "standard",
        trait_color_mode: str = "raw_hex",
        show_environment_sprites: bool = True,
        show_action_effects: bool = True,
        show_patch_overlay: bool = True,
        max_high_detail_effects: int = 4096,
        atlas_max_entries: int = 8192,
        visual_theme_seed: int = 0,
        output_resolution: tuple[int, int] = (1920, 1080),
    ) -> None:
        del max_slowdown_fraction
        self.atlas_max_entries = max(128, int(atlas_max_entries))
        self.backend_name = str(backend_name)
        self.renderer_mode = str(renderer_mode)
        self.event_bus = VisualEventBuffer(capacity=max(1, int(event_capacity)))
        self.clutter_budget = max(1, int(clutter_budget))
        self.adaptive = bool(adaptive)
        self.theme_name = str(theme)
        self.theme = get_theme(theme)
        self.scheduler = VisualFrameScheduler(
            VisualScheduleMode(str(schedule_mode)),
            frames_per_tick=max(1, int(frames_per_tick)),
            tick_stride=max(1, int(tick_stride)),
            render_every=max(1, int(render_every)),
            fail_on_drop=bool(fail_on_drop),
        )
        self.metrics = VisualControllerMetrics()
        self.window_size = window_size
        self.sidebar_width = max(0, int(sidebar_width))
        self.resizable = bool(resizable)
        self.fps = max(1, int(fps))
        self.output_resolution = output_resolution
        initial_shape = world_shape or (1, 1)
        viewport_width = max(320, int(window_size[0]) - self.sidebar_width)
        self.camera = CameraState(
            viewport=(0, 0, viewport_width, int(window_size[1])),
            world_shape=initial_shape,
            center=(initial_shape[0] / 2.0, initial_shape[1] / 2.0),
            zoom=8.0,
            follow_ow_id=selected_ow_id if camera_mode == "follow" else None,
            mode=str(camera_mode),
            min_zoom=float(min_zoom),
            max_zoom=float(max_zoom),
        )
        fit_world(self.camera)
        self.selection = VisualSelection(
            selected_ow_id=selected_ow_id,
            include_effects=bool(show_action_effects),
        )
        self.accessibility_mode = str(accessibility_mode)
        self.trait_color_mode = str(trait_color_mode)
        self.show_environment_sprites = bool(show_environment_sprites)
        self.show_patch_overlay = bool(show_patch_overlay)
        self.max_high_detail_effects = max(0, int(max_high_detail_effects))
        self.visual_theme_seed = int(visual_theme_seed)
        self.backend = self._make_backend(output_dir)
        self.closed = False
        self.previous_snapshot: VisualSnapshot | None = None
        self.current_snapshot: VisualSnapshot | None = None
        self._camera_initialized = world_shape is not None

    @classmethod
    def from_config(cls, cfg: Any, *, backend_name: str | None = None) -> Any:
        raw = backend_name or str(getattr(cfg.raqic, "full_gpu_visual_backend", "none"))
        mapping = {
            "none": "none",
            "pygame_copy": "pygame",
            "vispy_gpu": "vispy",
            "headless_export": "headless_export",
            "pygame": "pygame",
            "vispy": "vispy",
        }
        output_dir = getattr(cfg, "output_dir", None)
        if output_dir is None:
            metrics_path = Path(getattr(cfg.recording, "metrics_path", "results/metrics.jsonl"))
            output_dir = metrics_path.parent
        viz = cfg.visualization
        fixed = str(viz.schedule_mode) != "live_adaptive"
        return cls(
            backend_name=mapping.get(raw, raw),
            render_every=int(getattr(cfg.raqic, "full_gpu_render_every", viz.render_every)),
            event_capacity=int(getattr(cfg.raqic, "full_gpu_visual_event_capacity", 16384)),
            clutter_budget=int(getattr(cfg.raqic, "full_gpu_visual_clutter_budget", 2048)),
            adaptive=bool(getattr(cfg.raqic, "full_gpu_visual_adaptive_lod", viz.adaptive_lod))
            and not fixed,
            max_slowdown_fraction=float(
                getattr(
                    cfg.raqic,
                    "full_gpu_visual_max_slowdown_fraction",
                    viz.max_slowdown_fraction,
                )
            ),
            theme=str(getattr(cfg.raqic, "full_gpu_sprite_theme", "owl_dark_neon")),
            output_dir=Path(output_dir) / "visual_frames",
            schedule_mode=str(viz.schedule_mode),
            frames_per_tick=int(viz.frames_per_tick),
            tick_stride=int(viz.fixed_tick_stride),
            fail_on_drop=bool(viz.fail_on_dropped_recording_frame),
            renderer_mode=str(viz.renderer_mode),
            world_shape=(int(cfg.world.height), int(cfg.world.width)),
            window_size=(int(viz.window_width), int(viz.window_height)),
            sidebar_width=int(viz.viewport_sidebar_width),
            resizable=bool(viz.resizable),
            fps=int(viz.fps),
            camera_mode=str(viz.camera_mode),
            min_zoom=float(viz.min_zoom),
            max_zoom=float(viz.max_zoom),
            selected_ow_id=viz.follow_ow_id,
            accessibility_mode=str(viz.accessibility_mode),
            trait_color_mode=str(viz.trait_color_mode),
            show_environment_sprites=bool(viz.show_environment_sprites),
            show_action_effects=bool(viz.show_action_effects),
            show_patch_overlay=bool(viz.show_patch_overlay),
            max_high_detail_effects=int(viz.max_high_detail_effects),
            atlas_max_entries=int(viz.atlas_max_entries),
            visual_theme_seed=int(viz.visual_theme_seed),
            output_resolution=tuple(int(value) for value in viz.output_resolution),
        )

    def _make_backend(self, output_dir: Any) -> Any:
        if self.backend_name == "none":
            return NullVisualBackend()
        if self.backend_name == "pygame":
            from owl.viz.backends.pygame_backend import PygameVisualBackend

            return PygameVisualBackend(
                theme=self.theme_name,
                window_size=self.window_size,
                resizable=self.resizable,
                fps=self.fps,
                atlas_max_entries=self.atlas_max_entries,
            )
        if self.backend_name == "vispy":
            from owl.viz.backends.vispy_backend import VisPyVisualBackend

            return VisPyVisualBackend()
        if self.backend_name == "headless_export":
            from owl.viz.backends.headless_backend import HeadlessVisualBackend

            return HeadlessVisualBackend(
                output_dir,
                resolution=self.output_resolution,
                theme=self.theme_name,
                atlas_max_entries=self.atlas_max_entries,
            )
        raise ValueError(f"unknown visual backend: {self.backend_name}")

    def update_settings(self, settings: dict[str, Any] | None) -> None:
        if not settings:
            return
        overlay = str(settings.get("overlay", self.selection.overlay))
        allowed = {
            "none",
            "health",
            "resource",
            "toxin",
            "integration",
            "phase",
            "coherence",
            "action",
            "raqic",
        }
        if overlay not in allowed:
            raise ValueError(f"unknown visual overlay: {overlay}")
        selected = settings.get("selected_ow_id", self.selection.selected_ow_id)
        self.selection = VisualSelection(
            overlay=overlay,
            include_events=bool(settings.get("include_events", self.selection.include_events)),
            include_glyphs=bool(settings.get("include_glyphs", self.selection.include_glyphs)),
            include_debug=bool(settings.get("include_debug", self.selection.include_debug)),
            include_effects=bool(settings.get("include_effects", self.selection.include_effects)),
            selected_ow_id=None if selected is None else int(selected),
            fields=self.selection.fields,
        )
        if "render_every" in settings:
            self.scheduler.render_every = max(1, int(settings["render_every"]))

    def render_due(self, tick: int) -> bool:
        return self.backend_name != "none" and bool(self.scheduler.requests_for_tick(int(tick)))

    def _snapshot_namespace(self, snapshot: VisualSnapshot) -> Any:
        return SimpleNamespace(tick=snapshot.tick, **snapshot.arrays)

    @staticmethod
    def _event_key(event: VisualEvent) -> tuple[int, int, int, int, int, int]:
        return (
            int(event.event_type),
            int(event.source_id),
            int(event.y),
            int(event.x),
            int(event.target_y),
            int(event.target_x),
        )

    def _merge_event_bus(self, fresh: VisualEventBuffer) -> VisualEventBuffer:
        self.event_bus.prune()
        merged = {self._event_key(event): event for event in self.event_bus.events}
        for event in fresh.events:
            key = self._event_key(event)
            old = merged.get(key)
            if old is None or event.effective_priority >= old.effective_priority:
                merged[key] = event
        new_bus = VisualEventBuffer(capacity=self.event_bus.capacity)
        for event in sorted(
            merged.values(),
            key=lambda item: (
                -item.effective_priority,
                -item.ttl,
                int(item.event_type),
                item.y,
                item.x,
            ),
        ):
            new_bus.add(event, replace_lower_priority=True)
        new_bus.overflow_count += self.event_bus.overflow_count + fresh.overflow_count
        new_bus.truncated_count += self.event_bus.truncated_count + fresh.truncated_count
        new_bus.critical_drop_count += (
            self.event_bus.critical_drop_count + fresh.critical_drop_count
        )
        self.event_bus = new_bus
        return new_bus

    def capture_snapshot(self, device_state: Any) -> VisualSnapshot:
        started = time.perf_counter()
        base = snapshot_from_device_state(device_state, self.selection)
        self.metrics.snapshot_ms += (time.perf_counter() - started) * 1000.0
        self.metrics.snapshots_captured += 1
        return base

    def collect_events(
        self,
        ds: Any,
        *,
        snapshot: VisualSnapshot | None = None,
        state_snapshot: Any | None = None,
    ) -> VisualEventBuffer:
        current = snapshot or self.capture_snapshot(ds)
        namespace = state_snapshot or self._snapshot_namespace(current)
        fresh = events_from_state(
            namespace,
            max_events=self.event_bus.capacity,
            strict=False,
        )
        records = ds.metadata.get("event_queue", ())
        fresh.extend(
            events_from_event_records(
                records,
                tick=int(ds.tick),
                occupancy=current.arrays.get("occupancy"),
                max_events=self.event_bus.capacity,
            )
        )
        topology = ds.metadata.get("last_topology_events")
        if topology is not None:
            fresh.extend(
                events_from_topology_buffer(
                    topology,
                    ds.backend,
                    tick=int(ds.tick),
                    max_events=self.event_bus.capacity,
                )
            )
        if ds.metadata.get("last_qiskit_mismatch"):
            fresh.add(
                VisualEvent(
                    tick=int(ds.tick),
                    event_type=VisualEventType.QISKIT_MISMATCH,
                    y=0,
                    x=0,
                    ttl=8,
                ),
                replace_lower_priority=True,
            )
        if ds.metadata.get("gpu_fallback_count", 0):
            fresh.add(
                VisualEvent(
                    tick=int(ds.tick),
                    event_type=VisualEventType.GPU_FALLBACK,
                    y=0,
                    x=1,
                    ttl=8,
                ),
                replace_lower_priority=True,
            )
        merged = self._merge_event_bus(fresh)
        self.metrics.original_events += len(fresh.events)
        self.metrics.rendered_events += len(merged.events)
        self.metrics.event_overflow += merged.overflow_count
        self.metrics.critical_event_drops += merged.critical_drop_count
        return merged

    def _ensure_camera(self, snapshot: VisualSnapshot) -> None:
        if snapshot.world_shape != self.camera.world_shape:
            self.camera.world_shape = snapshot.world_shape
            self._camera_initialized = False
        if not self._camera_initialized or self.camera.mode == "fit":
            fit_world(self.camera)
            self._camera_initialized = True
        selected = getattr(self.backend, "selected_ow_id", None)
        if selected is not None and selected != self.selection.selected_ow_id:
            self.selection = replace(self.selection, selected_ow_id=int(selected))
            self.camera.follow_ow_id = int(selected)

    def _compose_underlay(self, device_state: Any) -> np.ndarray | None:
        if self.selection.overlay == "none":
            return None
        field = self.selection.overlay
        if field in {"action", "raqic", "coherence"}:
            field = (
                "health"
                if field != "coherence"
                else ("noetic_C" if "noetic_C" in device_state.arrays else "integration")
            )
        if field not in device_state.arrays:
            field = "integration"
        started = time.perf_counter()
        frame_device = compose_frame_device(
            device_state,
            field=field,
            theme_name=self.theme_name,
            show_actions=False,
            show_uncertainty=False,
            alpha=0.20,
        )
        composed = time.perf_counter()
        rgba = device_state.backend.asnumpy(frame_device)
        self.metrics.compositor_ms += (composed - started) * 1000.0
        self.metrics.transfer_ms += (time.perf_counter() - composed) * 1000.0
        return rgba

    def prepare_frames(
        self,
        device_state: Any,
        diagnostics: dict[str, Any] | None = None,
    ) -> tuple[VisualFrame, ...]:
        current = self.capture_snapshot(device_state)
        events = self.collect_events(device_state, snapshot=current)
        current = replace(current, events=tuple(events.events))
        self._ensure_camera(current)
        previous = self.current_snapshot or current
        self.previous_snapshot = previous
        self.current_snapshot = current
        underlay = self._compose_underlay(device_state)
        requests = self.scheduler.requests_for_tick(current.tick)
        frames: list[VisualFrame] = []
        for request in requests:
            started = time.perf_counter()
            scene = build_visual_scene(
                previous,
                current,
                request.progress,
                self.camera,
                self.selection,
                current.events,
                theme=self.theme,
                subframe_index=request.subframe_index,
                subframe_count=request.subframe_count,
                max_high_detail_effects=self.max_high_detail_effects,
                visual_seed=self.visual_theme_seed,
                accessibility_mode=self.accessibility_mode,
                trait_color_mode=self.trait_color_mode,
                show_environment_sprites=self.show_environment_sprites,
                show_patch_overlay=self.show_patch_overlay,
            )
            metadata = dict(scene.metadata)
            metadata.update(
                {
                    "patch_size": int(
                        getattr(
                            getattr(device_state.metadata.get("cfg"), "world", None),
                            "patch_size",
                            5,
                        )
                    ),
                    "diagnostics": diagnostics or {},
                    "schedule_mode": str(self.scheduler.mode),
                    "render_every": self.scheduler.render_every,
                }
            )
            scene = replace(
                scene,
                overlays=(() if underlay is None else (underlay,)),
                metadata=metadata,
            )
            self.metrics.scene_build_ms += (time.perf_counter() - started) * 1000.0
            self.metrics.sprite_count += len(scene.sprites)
            self.metrics.environment_count += len(scene.environment)
            self.metrics.effect_count += len(scene.effects)
            frames.append(
                VisualFrame(
                    rgba=underlay,
                    scene=scene,
                    scientific_tick=current.tick,
                    subframe_index=request.subframe_index,
                    subframe_count=request.subframe_count,
                    events=current.events,
                    metadata={
                        "tick": current.tick,
                        "subframe_index": request.subframe_index,
                        "subframe_count": request.subframe_count,
                        "sprite_count": len(scene.sprites),
                        "effect_count": len(scene.effects),
                        "environment_count": len(scene.environment),
                        "diagnostics": diagnostics or {},
                    },
                )
            )
        if (
            not frames
            and self.scheduler.fail_on_drop
            and self.scheduler.mode == VisualScheduleMode.RECORD_FIXED
        ):
            self.metrics.frames_dropped += 1
            raise RuntimeError(f"fixed visual schedule produced no frames for tick {current.tick}")
        return tuple(frames)

    def prepare_frame(
        self,
        *,
        device_state: Any,
        diagnostics: Any | None = None,
    ) -> VisualFrame:
        frames = self.prepare_frames(device_state, diagnostics=diagnostics)
        if frames:
            return frames[-1]
        return VisualFrame(
            rgba=None,
            scientific_tick=int(device_state.tick),
            metadata={"tick": int(device_state.tick), "skipped": True},
        )

    def submit(self, frame: VisualFrame, *, simulation_ms: float = 0.0) -> None:
        started = time.perf_counter()
        self.backend.submit(frame)
        render_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.render_ms += render_ms
        self.metrics.frames_submitted += 1
        if self.adaptive:
            self.scheduler.observe_live_cost(simulation_ms, render_ms)

    def submit_many(
        self,
        frames: tuple[VisualFrame, ...],
        *,
        simulation_ms: float = 0.0,
    ) -> None:
        for frame in frames:
            self.submit(frame, simulation_ms=simulation_ms)

    def close(self) -> None:
        if self.closed:
            return
        self.backend.close()
        self.closed = True

    def summary(self) -> dict[str, Any]:
        atlas_summary = getattr(getattr(self.backend, "atlas", None), "summary", lambda: {})()
        return {
            "backend": self.backend_name,
            "renderer_mode": self.renderer_mode,
            "schedule_mode": str(self.scheduler.mode),
            "render_every": self.scheduler.render_every,
            "frames_per_tick": self.scheduler.frames_per_tick,
            "selection": {
                "overlay": self.selection.overlay,
                "include_events": self.selection.include_events,
                "include_glyphs": self.selection.include_glyphs,
                "include_debug": self.selection.include_debug,
                "include_effects": self.selection.include_effects,
                "selected_ow_id": self.selection.selected_ow_id,
            },
            "camera": {
                "mode": self.camera.mode,
                "center": self.camera.center,
                "zoom": self.camera.zoom,
                "viewport": self.camera.viewport,
            },
            "atlas": atlas_summary,
            "trait_color_mode": self.trait_color_mode,
            "accessibility_mode": self.accessibility_mode,
            **self.metrics.__dict__,
        }
