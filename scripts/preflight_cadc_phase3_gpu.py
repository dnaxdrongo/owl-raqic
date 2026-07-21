#!/usr/bin/env python3
"""Check target-GPU hashing and parity evidence requirements before execution."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cadc_phase3_acceptance import synthetic_initial_state  # noqa: E402

from owl.core.actions import Action  # noqa: E402
from owl.core.config import SimulationConfig, load_config  # noqa: E402
from owl.core.state import EventRecord  # noqa: E402
from owl.counterfactual.rng_registry import branch_seed  # noqa: E402
from owl.counterfactual.scheduler import CounterfactualScheduler  # noqa: E402
from owl.counterfactual.schema import BranchStatus  # noqa: E402
from owl.counterfactual.source import CounterfactualSourceCollector  # noqa: E402
from owl.counterfactual.state_hash import compare_state_science, hash_state  # noqa: E402
from owl.experiments.controller import _release_hash  # noqa: E402
from owl.gpu.run_context import PersistentOWLDeviceRun  # noqa: E402


@dataclass(frozen=True)
class _Manifest:
    metadata_names: tuple[str, ...] = ("event_queue",)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _state(xp: Any) -> Any:
    return SimpleNamespace(
        arrays={
            "categorical": xp.asarray([1, 2, 3], dtype=xp.int32),
            "floating": xp.asarray([0.25, np.nan, np.inf, -np.inf], dtype=xp.float32),
        },
        patch_arrays={},
        global_arrays={},
        scalars={"tick": 3},
        metadata={
            "event_queue": [
                EventRecord(
                    kind="phase3_preflight",
                    tick=3,
                    payload={"device_vector": xp.arange(4, dtype=xp.int16)},
                )
            ]
        },
        manifest=_Manifest(),
    )


def _branch_smoke(source_root: Path) -> dict[str, Any]:
    """Execute real H=1/H=2 CuPy branches through authoritative action stages."""
    base = load_config(source_root / "configs/cadc_phase3_phase25_h100_acceptance.yaml")
    data = base.model_dump(mode="json")
    data["world"]["max_steps"] = 2
    data["counterfactual"]["repeats"] = 1
    data["counterfactual"]["horizons"] = [1, 2]
    data["counterfactual"]["family_horizons"] = {}
    data["counterfactual"]["max_active_branches"] = 1
    data["counterfactual"]["stream_lanes"] = 1
    cfg = SimulationConfig.model_validate(data)
    collector = CounterfactualSourceCollector(
        cfg,
        "0" * 64,
        run_id="phase3-gpu-preflight",
        condition="selected-anchor-smoke",
    )
    with TemporaryDirectory(prefix="owl_phase3_gpu_preflight_") as temporary:
        run = PersistentOWLDeviceRun.from_config(
            cfg,
            initial_state=copy.deepcopy(synthetic_initial_state(cfg)),
            force_backend="cupy",
            output_root=Path(temporary),
            counterfactual_observer=collector,
        )
        try:
            run.step()
            run.step()
            if len(collector.sources) != 1:
                raise RuntimeError(
                    f"GPU branch smoke expected one source; found {len(collector.sources)}"
                )
            source = collector.sources[0]
            before = hash_state(run.ds)
            scheduler = CounterfactualScheduler(run, cfg, active_branch_limit=1)
            decision_id = source.decisions.materialize_ids(run.ds.backend)[0]
            selected = int(run.ds.backend.asnumpy(source.decisions.selected_action)[0])
            executable = run.ds.backend.asnumpy(source.decisions.prechoice_executable)[0]
            paired_seed = branch_seed(int(cfg.world.seed), source.state.source_state_id, 0)
            branch_specs: list[tuple[int, int, int, bool]] = [
                (selected, -1, int(cfg.world.seed), True)
            ]
            for required_action in (Action.SENSE, Action.FLEE, Action.PURSUE):
                if not bool(executable[int(required_action)]):
                    raise AssertionError(
                        f"synthetic GPU preflight must make {required_action.name} executable"
                    )
                branch_specs.append((int(required_action), 0, paired_seed, False))
            branch_records: list[dict[str, Any]] = []
            for forced_action, repeat_index, seed, anchor in branch_specs:
                result = scheduler._execute_branch(  # noqa: SLF001 - certification seam
                    source,
                    0,
                    decision_id,
                    forced_action,
                    repeat_index,
                    seed,
                    anchor=anchor,
                )
                if result.status != BranchStatus.COMPLETED:
                    detail = "\n".join(result.failure_traceback) or str(result.failure)
                    raise RuntimeError(
                        f"GPU {Action(forced_action).name} branch smoke failed: {detail}"
                    )
                if set(result.outcomes) != {1, 2}:
                    raise AssertionError(
                        f"GPU {Action(forced_action).name} branch missed H=1/H=2 outcomes: "
                        f"{sorted(result.outcomes)}"
                    )
                if anchor and result.anchor_matches != {1: True, 2: True}:
                    raise AssertionError(
                        "GPU selected anchor failed H=1/H=2 equivalence: "
                        f"{result.anchor_matches}"
                    )
                branch_records.append(
                    {
                        "branch_id": result.branch_id,
                        "forced_action": forced_action,
                        "forced_action_name": Action(forced_action).name,
                        "selected_anchor": anchor,
                        "horizons": sorted(result.outcomes),
                        "anchor_matches": result.anchor_matches,
                    }
                )
            after = hash_state(run.ds)
            if before.root != after.root:
                raise AssertionError("GPU branch smoke mutated factual state")
            return {
                "passed": True,
                "source_state_id": source.state.source_state_id,
                "source_decision_id": decision_id,
                "branches": branch_records,
                "factual_root": before.root,
                "defer_host_metrics": bool(source.state.metadata.get("defer_host_metrics")),
            }
        finally:
            run.close(checkpoint=False)


def run(require_target: str, source_root: Path) -> dict[str, Any]:
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    device = {
        "name": str(name),
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "device_id": int(cp.cuda.Device().id),
        "total_global_memory_bytes": int(props["totalGlobalMem"]),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cupy_version": cp.__version__,
    }
    if require_target.upper() not in device["name"].upper():
        raise RuntimeError(f"wrong target GPU: required {require_target}, found {device['name']}")
    if device["device_count"] < 1 or device["total_global_memory_bytes"] <= 0:
        raise RuntimeError(f"invalid CUDA device metadata: {device}")

    gpu_left = _state(cp)
    gpu_right = _state(cp)
    cpu_equivalent = _state(np)
    left_hash = hash_state(gpu_left)
    right_hash = hash_state(gpu_right)
    cpu_hash = hash_state(cpu_equivalent)
    if left_hash.root != right_hash.root or left_hash.root != cpu_hash.root:
        raise AssertionError("canonical state hash is not backend invariant")
    if left_hash.device_to_host_bytes <= left_hash.array_bytes:
        raise AssertionError("metadata device transfer was not included in hash telemetry")
    if not compare_state_science(gpu_left, gpu_right).passed:
        raise AssertionError("equal CuPy states failed scientific comparison")
    if not compare_state_science(gpu_left, cpu_equivalent).passed:
        raise AssertionError("equivalent CuPy/NumPy states failed scientific comparison")

    gpu_right.arrays["categorical"][1] = 9
    categorical = compare_state_science(gpu_left, gpu_right)
    if categorical.passed or categorical.categorical_failures != ("arrays.categorical",):
        raise AssertionError(f"categorical mismatch was not exact: {categorical}")
    gpu_right = _state(cp)
    gpu_right.arrays["floating"][0] += cp.float32(2e-5)
    floating = compare_state_science(gpu_left, gpu_right)
    if floating.passed or floating.floating_failures != ("arrays.floating",):
        raise AssertionError(f"floating mismatch was not detected: {floating}")
    branch_smoke = _branch_smoke(source_root)
    if branch_smoke["defer_host_metrics"] is not True:
        raise AssertionError("counterfactual source did not preserve deferred device metrics")
    return {
        "schema_version": "owl.cadc.phase3-gpu-preflight.v1",
        "passed": True,
        "classification": f"{require_target.upper()}_PHASE3_GPU_PREFLIGHT_PASSED",
        "source_sha256": _release_hash(source_root),
        "device": device,
        "state_root": left_hash.root,
        "array_bytes": left_hash.array_bytes,
        "device_to_host_bytes": left_hash.device_to_host_bytes,
        "branch_smoke": branch_smoke,
        "checks": {
            "cupy_metadata_hash": True,
            "backend_invariant_hash": True,
            "cupy_science_comparison": True,
            "mixed_backend_science_comparison": True,
            "categorical_mismatch_detection": True,
            "nan_infinity_mask_comparison": True,
            "floating_tolerance_detection": True,
            "real_branch_postdecision_execution": True,
            "branch_deferred_device_scalars": branch_smoke["defer_host_metrics"] is True,
            "branch_factual_nonmutation": True,
            "selected_anchor_horizon_one": True,
        },
        "failures": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-target", choices=("H100", "H200", "B200"), default="H100")
    parser.add_argument("--source-root", default=str(ROOT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        payload = run(args.require_target, Path(args.source_root).resolve())
    except Exception as exc:
        payload = {
            "schema_version": "owl.cadc.phase3-gpu-preflight.v1",
            "passed": False,
            "classification": "FAILED_CLOSED_GPU_PREFLIGHT",
            "failures": [f"{type(exc).__name__}: {exc}"],
            "traceback": traceback.format_exc().splitlines(),
        }
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
