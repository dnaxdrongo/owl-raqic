"""CLI interface for running one simulation through the unified runtime dispatcher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from owl.core.config import load_config
from owl.core.init import initialize_world
from owl.engine.loop import step
from owl.experiments.presets import get_condition, list_conditions
from owl.record.metrics import collect_metrics, save_metrics, summarize_metrics
from owl.record.snapshots import save_snapshot
from owl.record.zarr_recorder import create_recorder
from owl.runtime.capabilities import detect_runtime_capabilities
from owl.runtime.dispatcher import dispatch_run
from owl.runtime.execution_plan import compile_execution_plan


def _with_run_paths(cfg: Any, run_dir: Path) -> Any:
    """Return cfg copy with recording paths placed under run_dir when relative."""
    out = cfg.model_copy(deep=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(out.recording.metrics_path)
    zarr_path = Path(out.recording.zarr_path)
    if not metrics_path.is_absolute():
        out.recording.metrics_path = str(run_dir / metrics_path.name)
    if not zarr_path.is_absolute():
        out.recording.zarr_path = str(run_dir / zarr_path.name)
    return out


def _run_reference_with_recording(cfg: Any, plan: Any, condition_name: str) -> Any:
    """Execute CPU/stage-once modes while retaining existing Zarr cadence."""
    rng = np.random.default_rng(cfg.world.seed)
    state = initialize_world(cfg, rng)
    if getattr(cfg.raqic, "enabled", False):
        from owl.raqic.state import ensure_raqic_fields

        ensure_raqic_fields(state, cfg)
    recorder = create_recorder(cfg, state, max_steps=cfg.world.max_steps)
    metrics: list[dict[str, Any]] = []
    if recorder is not None:
        recorder.maybe_record(state)
    try:
        for _ in range(int(cfg.world.max_steps)):
            step(state, cfg, rng)
            row: dict[str, Any] = dict(collect_metrics(state, cfg))
            row["condition"] = condition_name
            row["seed"] = int(cfg.world.seed)
            metrics.append(row)
            if recorder is not None:
                recorder.maybe_record(state)
    finally:
        if recorder is not None:
            recorder.close()
    metadata = {
        "simulation_backend": plan.simulation_backend,
        "decision_backend": plan.decision_backend,
        "device_state_instances": (
            int(cfg.world.max_steps) if plan.simulation_backend == "gpu_stage_once" else 0
        ),
        "checkpoint_count": 0,
        "fallback_count": 0,
    }
    return state, metrics, metadata


def run_validated_config(cfg: Any) -> Any:
    """Run an already validated configuration through the canonical dispatcher."""
    capabilities = detect_runtime_capabilities()
    plan = compile_execution_plan(cfg, capabilities)
    return dispatch_run(cfg, plan)


def run_single(config_path: str | Path, condition: str | None = None) -> Path:
    """Run one simulation using the execution tier requested by configuration.

    CPU and stage-once modes retain the established recording path. Persistent,
    graph, and distributed modes are delegated to the unified dispatcher, so
    ``owl-run`` and the dedicated GPU entry points execute the same engine.
    """
    cfg = load_config(config_path)
    condition_name = "default" if condition is None else str(condition)
    if condition is not None:
        cfg = get_condition(condition_name, cfg)

    metrics_target = Path(cfg.recording.metrics_path)
    run_dir = metrics_target.parent if metrics_target.parent != Path("") else Path("results")
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    cfg = _with_run_paths(cfg, run_dir)

    capabilities = detect_runtime_capabilities()
    plan = compile_execution_plan(cfg, capabilities)

    normalized_json = json.dumps(
        cfg.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    config_sha256 = hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()
    normalized_path = run_dir / "normalized_config.json"
    normalized_path.write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    certification_path = None
    if plan.require_certification:
        from owl.gpu.certification_store import CertificationStore

        certification_path = CertificationStore(
            cfg.raqic.full_gpu_certification_dir
        ).require_compatible(plan, config_hash=config_sha256)

    if plan.simulation_backend in {"cpu", "gpu_stage_once"}:
        state, metrics, execution_metadata = _run_reference_with_recording(
            cfg, plan, condition_name
        )
    else:
        result = dispatch_run(cfg, plan)
        state = result.state
        metrics = list(result.metrics)
        for row in metrics:
            row.setdefault("condition", condition_name)
            row.setdefault("seed", int(cfg.world.seed))
        execution_metadata = dict(result.execution_metadata)

        # Persistent GPU recording owns compact asynchronous outputs internally.
        # A final dense Zarr snapshot is still emitted when recording is enabled,
        # without forcing per-tick full-state writeback.
        if cfg.recording.enabled:
            recorder = create_recorder(cfg, state, max_steps=0)
            if recorder is not None:
                try:
                    recorder.maybe_record(state)
                finally:
                    recorder.close()

    execution_metadata["config_sha256"] = config_sha256
    execution_metadata["normalized_config"] = str(normalized_path)
    execution_metadata["certification_record"] = (
        None if certification_path is None else str(certification_path)
    )

    metrics_path = Path(cfg.recording.metrics_path)
    save_metrics(metrics, str(metrics_path))

    summary = summarize_metrics(metrics)
    summary["condition"] = condition_name
    summary["seed"] = int(cfg.world.seed)
    summary["execution_plan"] = plan.to_dict()
    summary["execution_metadata"] = execution_metadata
    summary_path = metrics_path.with_name(metrics_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    snapshot_path = metrics_path.with_name(metrics_path.stem + "_final_snapshot.npz")
    save_snapshot(state, str(snapshot_path))

    plan_path = metrics_path.with_name(metrics_path.stem + "_execution_plan.json")
    plan_path.write_text(
        json.dumps(
            {
                "execution_plan": plan.to_dict(),
                "runtime_capabilities": capabilities.details,
                "execution_metadata": execution_metadata,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Observer-Window Life simulation.")
    parser.add_argument("config_path", help="Path to YAML/JSON config.")
    parser.add_argument(
        "--condition",
        choices=list_conditions(),
        default=None,
        help="Optional experiment condition.",
    )
    args = parser.parse_args()
    path = run_single(args.config_path, condition=args.condition)
    print(path)


if __name__ == "__main__":
    main()
