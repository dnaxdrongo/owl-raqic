from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from owl.core.config import SimulationConfig, load_config

_ALLOWED_OVERRIDES = {
    "world.max_steps",
    "world.seed",
    "recording.enabled",
    "recording.metrics_path",
    "recording.zarr_path",
    "visualization.enabled",
    "visualization.backend",
    "raqic.qiskit_runtime_binding_policy",
    "raqic.qiskit_state_preparation_strategy",
    "raqic.qiskit_runtime_parameter_bind_enable",
    "raqic.qiskit_preflight_required",
    "raqic.qiskit_allow_automatic_execution_fallback",
}


def _set_nested(payload: dict[str, Any], dotted: str, value: Any) -> None:
    target = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = target.get(part)
        if not isinstance(current, dict):
            raise KeyError(f"override path does not identify a mapping: {dotted}")
        target = current
    target[parts[-1]] = value


def build_validated_config(
    source: str | Path,
    destination: str | Path,
    overrides: dict[str, Any],
) -> SimulationConfig:
    unknown = sorted(set(overrides) - _ALLOWED_OVERRIDES)
    if unknown:
        raise ValueError(f"unapproved experiment config overrides: {unknown}")
    cfg = load_config(source)
    payload = cfg.model_dump(mode="json")
    for dotted, override_value in overrides.items():
        _set_nested(payload, dotted, override_value)
    # Headless rendering is an out-of-band controller choice, never a schema value.
    if payload["visualization"]["backend"] not in {"pygame", "none"}:
        raise ValueError("visualization.backend must remain 'pygame' or 'none'")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    normalized = load_config(target)
    normalized_payload = normalized.model_dump(mode="json")
    for dotted, expected in overrides.items():
        normalized_value: Any = normalized_payload
        for part in dotted.split("."):
            normalized_value = normalized_value[part]
        if normalized_value != expected:
            raise ValueError(
                f"normalized override mismatch for {dotted}: {normalized_value!r} != {expected!r}"
            )
    (target.with_suffix(target.suffix + ".normalized.json")).write_text(
        json.dumps(normalized_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return normalized
