"""JSON-native recursive types and narrowing helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
JSONArray: TypeAlias = list[JSONValue]


def require_object(value: JSONValue, label: str = "value") -> JSONObject:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def require_array(value: JSONValue, label: str = "value") -> JSONArray:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def require_bool(mapping: Mapping[str, JSONValue], key: str) -> bool:
    value = mapping.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a JSON boolean")
    return value


def require_str(mapping: Mapping[str, JSONValue], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a JSON string")
    return value


def require_int(mapping: Mapping[str, JSONValue], key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int:
        raise ValueError(f"{key} must be a JSON integer")
    return value


def require_float(mapping: Mapping[str, JSONValue], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a JSON number")
    return float(value)
