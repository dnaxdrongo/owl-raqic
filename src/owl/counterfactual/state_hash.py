"""Canonical complete-state Merkle SHA-256 contract."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from owl.counterfactual.schema import STATE_HASH_VERSION
from owl.counterfactual.state_clone import ordered_array_groups


@dataclass(frozen=True)
class StateHashResult:
    algorithm: str
    root: str
    leaf_hashes: tuple[tuple[str, str], ...]
    array_bytes: int
    device_to_host_bytes: int


@dataclass(frozen=True)
class StateComparison:
    passed: bool
    categorical_failures: tuple[str, ...]
    floating_failures: tuple[str, ...]
    max_abs_difference: float


def _lp(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def _metadata_host_array(value: Any) -> tuple[np.ndarray, int]:
    """Return a canonical host array and its actual device-transfer byte count."""
    if value.__class__.__module__.startswith("cupy"):
        import cupy as cp  # pragma: no cover - target GPU only

        contiguous = cp.ascontiguousarray(value)
        return cp.asnumpy(contiguous), int(contiguous.nbytes)
    return np.ascontiguousarray(value), 0


def _canonical_json_with_transfer(value: Any) -> tuple[bytes, int]:
    """Serialize nested scientific metadata identically on NumPy and CuPy."""
    device_to_host_bytes = 0

    def normalize(item: Any) -> Any:
        nonlocal device_to_host_bytes
        is_backend_array = isinstance(item, np.ndarray) or (
            item.__class__.__module__.startswith("cupy")
            and hasattr(item, "dtype")
            and hasattr(item, "shape")
        )
        if is_backend_array:
            host, transferred = _metadata_host_array(item)
            device_to_host_bytes += transferred
            contiguous = np.ascontiguousarray(host)
            return {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "bytes_hex": contiguous.tobytes(order="C").hex(),
            }
        if isinstance(item, np.generic):
            return normalize(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            label = "nan" if np.isnan(item) else ("+inf" if item > 0 else "-inf")
            return {"nonfinite_float": label}
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        if isinstance(item, Enum):
            return normalize(item.value)
        if isinstance(item, bytes):
            return {"bytes_hex": item.hex()}
        if isinstance(item, Mapping):
            normalized: dict[str, Any] = {}
            for key, nested in item.items():
                name = key if isinstance(key, str) else str(key)
                if name in normalized:
                    raise ValueError(f"metadata key collision after normalization: {name!r}")
                normalized[name] = normalize(nested)
            return normalized
        if isinstance(item, (list, tuple)):
            return [normalize(nested) for nested in item]
        if isinstance(item, (set, frozenset)):
            values = [normalize(nested) for nested in item]
            return sorted(
                values,
                key=lambda nested: json.dumps(
                    nested, sort_keys=True, separators=(",", ":"), allow_nan=False
                ),
            )
        if is_dataclass(item) and not isinstance(item, type):
            return {field.name: normalize(getattr(item, field.name)) for field in fields(item)}
        if hasattr(item, "to_dict"):
            return normalize(item.to_dict())
        if hasattr(item, "__dict__"):
            return normalize(vars(item))
        raise TypeError(type(item).__name__)

    encoded = json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return encoded, device_to_host_bytes


def _canonical_json(value: Any) -> bytes:
    return _canonical_json_with_transfer(value)[0]


def _host_chunks(value: Any, chunk_bytes: int) -> Iterator[bytes]:
    if value.__class__.__module__.startswith("cupy"):
        import cupy as cp  # pragma: no cover - target GPU only

        flat = cp.ascontiguousarray(value).view(cp.uint8).reshape(-1)
        for start in range(0, int(flat.size), chunk_bytes):
            yield cp.asnumpy(flat[start : start + chunk_bytes]).tobytes(order="C")
        return
    flat = np.ascontiguousarray(value).view(np.uint8).reshape(-1)
    view = memoryview(flat)
    for start in range(0, int(flat.size), chunk_bytes):
        yield bytes(view[start : start + chunk_bytes])


def _array_leaf(group: str, name: str, value: Any, chunk_bytes: int) -> str:
    dtype = np.dtype(value.dtype)
    header = b"".join(
        (
            _lp(group.encode()),
            _lp(name.encode()),
            _lp(dtype.str.encode()),
            _lp(str(int(value.ndim)).encode()),
            _lp(_canonical_json(tuple(int(item) for item in value.shape))),
        )
    )
    digest = hashlib.sha256(_lp(b"array") + header)
    for chunk in _host_chunks(value, chunk_bytes):
        digest.update(_lp(chunk))
    return digest.hexdigest()


def _metadata_subset(state: Any) -> Mapping[str, Any]:
    names = getattr(getattr(state, "manifest", None), "metadata_names", ())
    if names:
        return {name: state.metadata[name] for name in names}
    allowed = (
        "field_epochs",
        "event_queue",
        "cfg_mode",
        "precision_policy",
        "precision_promoted_fields",
        "raqic_real_dtype",
    )
    return {name: state.metadata[name] for name in allowed if name in state.metadata}


def hash_state(state: Any, *, chunk_bytes: int = 4 * 1024**2) -> StateHashResult:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    leaves: list[tuple[str, str]] = []
    total = 0
    d2h = 0
    for group, mapping in ordered_array_groups(state):
        for name, value in mapping.items():
            key = f"{group}.{name}"
            leaves.append((key, _array_leaf(group, name, value, chunk_bytes)))
            size = int(value.nbytes)
            total += size
            if value.__class__.__module__.startswith("cupy"):
                d2h += size
    for group, value in (
        ("scalars", state.scalars),
        ("metadata", _metadata_subset(state)),
    ):
        encoded, metadata_d2h = _canonical_json_with_transfer(value)
        d2h += metadata_d2h
        leaves.append((group, hashlib.sha256(_lp(group.encode()) + _lp(encoded)).hexdigest()))
    root = hashlib.sha256(_lp(STATE_HASH_VERSION.encode()))
    for name, leaf in leaves:
        root.update(_lp(name.encode()))
        root.update(_lp(bytes.fromhex(leaf)))
    return StateHashResult(
        algorithm=STATE_HASH_VERSION,
        root=root.hexdigest(),
        leaf_hashes=tuple(leaves),
        array_bytes=total,
        device_to_host_bytes=d2h,
    )


def differing_leaves(left: StateHashResult, right: StateHashResult) -> tuple[str, ...]:
    left_map = dict(left.leaf_hashes)
    right_map = dict(right.leaf_hashes)
    return tuple(
        name
        for name in sorted(set(left_map) | set(right_map))
        if left_map.get(name) != right_map.get(name)
    )


def compare_state_science(
    left: Any,
    right: Any,
    *,
    float32_atol: float = 9.5367431640625e-7,
    float64_atol: float = 1e-10,
) -> StateComparison:
    """Compare exact categorical science and declared floating tolerances."""
    categorical: list[str] = []
    floating: list[str] = []
    maximum = 0.0
    left_groups = dict(ordered_array_groups(left))
    right_groups = dict(ordered_array_groups(right))
    for group in sorted(set(left_groups) | set(right_groups)):
        left_mapping = left_groups.get(group, {})
        right_mapping = right_groups.get(group, {})
        for name in sorted(set(left_mapping) | set(right_mapping)):
            key = f"{group}.{name}"
            if name not in left_mapping or name not in right_mapping:
                categorical.append(key)
                continue
            raw_a = left_mapping[name]
            raw_b = right_mapping[name]
            use_cupy = raw_a.__class__.__module__.startswith(
                "cupy"
            ) or raw_b.__class__.__module__.startswith("cupy")
            if use_cupy:
                import cupy as xp  # pragma: no cover - target GPU only
            else:
                xp = np
            a = xp.asarray(raw_a)
            b = xp.asarray(raw_b)
            if a.shape != b.shape or a.dtype != b.dtype:
                categorical.append(key)
                continue
            if a.dtype.kind not in {"f", "c"}:
                if not bool(xp.array_equal(a, b)):
                    categorical.append(key)
                continue
            if not bool(xp.array_equal(xp.isnan(a), xp.isnan(b))):
                floating.append(key)
                continue
            if not bool(xp.array_equal(xp.isposinf(a), xp.isposinf(b))):
                floating.append(key)
                continue
            if not bool(xp.array_equal(xp.isneginf(a), xp.isneginf(b))):
                floating.append(key)
                continue
            finite = xp.isfinite(a) & xp.isfinite(b)
            if bool(xp.any(finite)):
                difference = xp.abs(a[finite] - b[finite])
                local = float(xp.max(difference))
                maximum = max(maximum, local)
                tolerance = float64_atol if a.dtype.itemsize >= 8 else float32_atol
                if not bool(xp.all(difference <= tolerance)):
                    floating.append(key)
    for name in sorted(set(left.scalars) | set(right.scalars)):
        key = f"scalars.{name}"
        if name not in left.scalars or name not in right.scalars:
            categorical.append(key)
            continue
        a, b = left.scalars[name], right.scalars[name]
        if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            difference = abs(float(a) - float(b))
            maximum = max(maximum, difference)
            if difference > float64_atol:
                floating.append(key)
        elif int(a) != int(b):
            categorical.append(key)
    return StateComparison(
        passed=not categorical and not floating,
        categorical_failures=tuple(categorical),
        floating_failures=tuple(floating),
        max_abs_difference=maximum,
    )
