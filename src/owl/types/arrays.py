"""CPU array aliases and structural device-array contracts."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

ArrayAny: TypeAlias = npt.NDArray[Any]
Float32Array: TypeAlias = npt.NDArray[np.float32]
Float64Array: TypeAlias = npt.NDArray[np.float64]
Int32Array: TypeAlias = npt.NDArray[np.int32]
Int64Array: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
ScalarT = TypeVar("ScalarT", covariant=True)


@runtime_checkable
class DeviceArray(Protocol[ScalarT]):
    shape: tuple[int, ...]
    dtype: Any
    size: int
    nbytes: int
    ndim: int

    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
