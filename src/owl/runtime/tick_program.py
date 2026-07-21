from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .stage_registry import GRAPH_SEGMENTS, TICK_STAGES, TickStageSpec


@dataclass
class TickProgram:
    """Ordered semantic program used to audit every execution backend."""

    stages: tuple[TickStageSpec, ...] = TICK_STAGES

    def validate(self) -> None:
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("tick stage names must be unique")
        seen_segments = tuple(dict.fromkeys(stage.graph_segment for stage in self.stages))
        if seen_segments != GRAPH_SEGMENTS:
            raise ValueError(
                f"tick graph segment order {seen_segments!r} does not match {GRAPH_SEGMENTS!r}"
            )

    def metadata(self) -> list[dict[str, Any]]:
        self.validate()
        return [
            {
                "name": stage.name,
                "phase": stage.phase,
                "inputs": list(stage.inputs),
                "outputs": list(stage.outputs),
                "graph_segment": stage.graph_segment,
                "distributed_policy": stage.distributed_policy,
                "visual_event_types": list(stage.visual_event_types),
            }
            for stage in self.stages
        ]


DEFAULT_TICK_PROGRAM = TickProgram()
DEFAULT_TICK_PROGRAM.validate()
