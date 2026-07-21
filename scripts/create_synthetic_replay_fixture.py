# ruff: noqa: E402 -- approved source-tree bootstrap
"""Create a deterministic small replay bundle for viewer acceptance tests."""

from __future__ import annotations

try:
    from _repo_bootstrap import bootstrap_repo
except ModuleNotFoundError:  # imported as scripts.<module>
    from scripts._repo_bootstrap import bootstrap_repo

bootstrap_repo()

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from owl.record.replay_recorder import ReplayRecorder
from owl.viz.event_bus import VisualEvent, VisualEventType
from owl.viz.visual_snapshot import snapshot_from_arrays


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ticks", type=int, default=25)
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args(argv)
    output = Path(args.output)
    action_names = tuple(f"ACTION_{index:02d}" for index in range(22))
    recorder = ReplayRecorder(
        output,
        run_id="v098_synthetic_replay",
        condition="synthetic_all_on",
        seed=9303,
        requested_ticks=args.ticks,
        recording_tier="analysis_full",
        action_names=action_names,
        source_sha256="synthetic",
        config_sha256="synthetic",
        table_flush_ticks=5,
    )
    shape = (args.size, args.size)
    for tick in range(1, args.ticks + 1):
        health = np.zeros(shape, dtype=np.float32)
        resource = np.zeros(shape, dtype=np.float32)
        occupancy = np.full(shape, -1, dtype=np.int64)
        y = (3 + tick // 4) % args.size
        x = (2 + tick) % args.size
        health[y, x] = 0.9
        resource[y, x] = 0.75
        occupancy[y, x] = 101
        action = tick % 22
        probabilities = np.zeros((*shape, 22), dtype=np.float32)
        probabilities[..., 0] = 1.0
        probabilities[y, x] = 0.01
        probabilities[y, x, action] = 0.79
        probabilities[y, x] /= probabilities[y, x].sum()
        arrays = {
            "health": health,
            "resource": resource,
            "toxin": np.zeros(shape, dtype=np.float32),
            "food": np.zeros(shape, dtype=np.float32),
            "waste": np.zeros(shape, dtype=np.float32),
            "obstacle": np.zeros(shape, dtype=bool),
            "occupancy": occupancy,
            "readout": np.where(health > 0, action, 0).astype(np.int16),
            "raqic_readout": np.where(health > 0, action, 0).astype(np.int16),
            "integration": health * 0.8,
            "lineage_id": np.where(health > 0, 7, -1).astype(np.int64),
            "parent_id": np.where(health > 0, 55, -1).astype(np.int64),
            "age": health * tick,
            "development_stage": health * 0.5,
            "starvation_debt": np.zeros(shape, dtype=np.float32),
            "raqic_record_confidence": health * 0.9,
            "raqic_probabilities": probabilities,
            "possibility": probabilities.copy(),
            "raqic_score": np.log(np.maximum(probabilities, 1e-8)),
            "raqic_phase": np.zeros_like(probabilities),
            "raqic_parent_intention": probabilities.copy(),
            "raqic_pre_mixer_probabilities": probabilities.copy(),
            "last_utilities": np.broadcast_to(
                np.linspace(-1.0, 1.0, 22, dtype=np.float32), (*shape, 22)
            ),
            "pre_utilities": np.broadcast_to(
                np.linspace(-0.5, 0.5, 22, dtype=np.float32), (*shape, 22)
            ),
            "last_logits": np.log(np.maximum(probabilities, 1e-8)),
            "last_action_probabilities": probabilities.copy(),
            "raqic_utility_innovation": np.zeros_like(probabilities),
            "raqic_phase_alignment": np.zeros_like(probabilities),
            "raqic_resonant_parent_intention": np.zeros_like(probabilities),
            "raqic_utility_innovation_norm": health * 0.05,
            "raqic_interference_delta_l1": health * 0.02,
            "_authority_bool": np.ones_like(probabilities, dtype=bool),
        }
        event = VisualEvent(
            tick=tick,
            event_type=VisualEventType.MOVE,
            y=y,
            x=x,
            action=action,
            source_id=101,
        )
        recorder.record(
            snapshot_from_arrays(
                tick=tick,
                boundary_mode="toroidal",
                arrays=arrays,
                events=(event,),
                metadata={"fixture": True},
            ),
            diagnostics={},
        )
    recorder.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
