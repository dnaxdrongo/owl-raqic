"""Repository-owned static type contracts."""

from .arrays import (
    ArrayAny,
    BoolArray,
    DeviceArray,
    Float32Array,
    Float64Array,
    Int32Array,
    Int64Array,
)
from .backend import ArrayBackendProtocol, EventLike, StreamLike
from .json import JSONScalar, JSONValue

__all__ = [
    "ArrayAny",
    "ArrayBackendProtocol",
    "BoolArray",
    "DeviceArray",
    "EventLike",
    "Float32Array",
    "Float64Array",
    "Int32Array",
    "Int64Array",
    "JSONScalar",
    "JSONValue",
    "StreamLike",
]
