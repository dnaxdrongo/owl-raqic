from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from owl.replay.manifest import ReplayManifest
from owl.viz.event_bus import VisualEvent
from owl.viz.visual_snapshot import VisualSnapshot


@dataclass(frozen=True)
class OWReplayDetails:
    tick: int
    ow_id: int
    position: tuple[int, int] | None
    values: dict[str, Any]
    action_math: tuple[dict[str, Any], ...] = ()
    recent_events: tuple[dict[str, Any], ...] = ()


class ReplayDataSource(Protocol):
    @property
    def manifest(self) -> ReplayManifest: ...

    def tick_count(self) -> int: ...

    def available_ticks(self) -> Sequence[int]: ...

    def load_snapshot(
        self,
        tick: int,
        fields: Collection[str] | None = None,
    ) -> VisualSnapshot: ...

    def load_events(self, start_tick: int, end_tick: int) -> tuple[VisualEvent, ...]: ...

    def load_ow_details(self, tick: int, ow_id: int) -> OWReplayDetails | None: ...

    def load_action_math(self, tick: int, ow_id: int) -> tuple[dict[str, Any], ...]: ...

    def available_cadc_tables(self) -> Sequence[str]: ...

    def load_cadc_table(
        self,
        name: str,
        *,
        tick: int | None = None,
        ow_id: int | None = None,
        columns: Sequence[str] | None = None,
    ) -> Any | None: ...

    def export_selection_csv(
        self,
        destination: str,
        *,
        ow_id: int,
        start_tick: int,
        end_tick: int,
    ) -> str: ...
