from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _dtype_constant(nccl: Any, dtype: Any) -> int:
    import numpy as np

    dt = np.dtype(dtype)
    candidates = {
        np.dtype("float16"): ("NCCL_FLOAT16", "NCCL_HALF"),
        np.dtype("float32"): ("NCCL_FLOAT32", "NCCL_FLOAT"),
        np.dtype("float64"): ("NCCL_FLOAT64", "NCCL_DOUBLE"),
        np.dtype("int8"): ("NCCL_INT8", "NCCL_CHAR"),
        np.dtype("uint8"): ("NCCL_UINT8",),
        np.dtype("int32"): ("NCCL_INT32", "NCCL_INT"),
        np.dtype("uint32"): ("NCCL_UINT32",),
        np.dtype("int64"): ("NCCL_INT64",),
        np.dtype("uint64"): ("NCCL_UINT64",),
        np.dtype("bool"): ("NCCL_UINT8",),
    }
    for name in candidates.get(dt, ()):
        if hasattr(nccl, name):
            return int(getattr(nccl, name))
    raise TypeError(f"unsupported NCCL dtype: {dt}")


@dataclass
class CollectiveRecord:
    sequence: int
    operation: str
    count: int
    dtype: str
    peer_or_root: int
    tick: int
    rank: int
    phase: str
    field_group: str


class NCCLTransport:
    """Thin CuPy NCCL wrapper with a collective-sequence ledger.

    One instance belongs to one process and one CUDA device.  Array methods are
    deliberately synchronous in their Python contract but enqueue work on the
    supplied CUDA stream.
    """

    def __init__(self, communicator: Any, cp: Any, rank: int, world_size: int) -> None:
        self.comm = communicator
        self.cp = cp
        self.nccl = cp.cuda.nccl
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.sequence = 0
        self.ledger: list[CollectiveRecord] = []
        self.closed = False

    @classmethod
    def connect(cls, *, rank: int, world_size: int, unique_id: int, device_id: int) -> Any:
        import cupy as cp

        cp.cuda.Device(int(device_id)).use()
        comm = cp.cuda.nccl.NcclCommunicator(int(world_size), unique_id, int(rank))
        return cls(comm, cp, rank, world_size)

    @staticmethod
    def unique_id() -> Any:
        import cupy as cp

        return cp.cuda.nccl.get_unique_id()

    def _record(
        self,
        operation: Any,
        array: Any,
        peer_or_root: Any,
        tick: int,
        *,
        phase: str = "unspecified",
        field_group: Any = "unspecified",
    ) -> Any:
        self.sequence += 1
        self.ledger.append(
            CollectiveRecord(
                self.sequence,
                str(operation),
                int(array.size),
                str(array.dtype),
                int(peer_or_root),
                int(tick),
                int(self.rank),
                str(phase),
                str(field_group),
            )
        )

    def group_start(self) -> None:
        self.nccl.groupStart()

    def group_end(self) -> None:
        self.nccl.groupEnd()

    def send(
        self,
        array: Any,
        *,
        peer: int,
        stream: Any,
        tick: int = -1,
        phase: str = "unspecified",
        field_group: str = "unspecified",
    ) -> None:
        if not array.flags.c_contiguous:
            raise ValueError("NCCL send source must be a persistent contiguous buffer")
        arr = array
        self._record("send", arr, peer, tick, phase=phase, field_group=field_group)
        self.comm.send(
            int(arr.data.ptr),
            int(arr.size),
            _dtype_constant(self.nccl, arr.dtype),
            int(peer),
            int(stream.ptr),
        )

    def recv(
        self,
        array: Any,
        *,
        peer: int,
        stream: Any,
        tick: int = -1,
        phase: str = "unspecified",
        field_group: str = "unspecified",
    ) -> None:
        if not array.flags.c_contiguous:
            raise ValueError("NCCL receive target must be contiguous")
        self._record("recv", array, peer, tick, phase=phase, field_group=field_group)
        self.comm.recv(
            int(array.data.ptr),
            int(array.size),
            _dtype_constant(self.nccl, array.dtype),
            int(peer),
            int(stream.ptr),
        )

    def all_reduce(
        self,
        send: Any,
        recv: Any,
        *,
        op: str = "sum",
        stream: Any,
        tick: int = -1,
        phase: str = "collective",
        field_group: str = "global",
    ) -> Any:
        if not send.flags.c_contiguous:
            raise ValueError("NCCL all_reduce source must be contiguous")
        if send.size != recv.size or send.dtype != recv.dtype:
            raise ValueError("all_reduce send/recv count and dtype must match")
        op_map = {
            "sum": self.nccl.NCCL_SUM,
            "prod": self.nccl.NCCL_PROD,
            "max": self.nccl.NCCL_MAX,
            "min": self.nccl.NCCL_MIN,
        }
        if op not in op_map:
            raise ValueError(f"unsupported NCCL reduction: {op}")
        self._record(f"all_reduce:{op}", send, -1, tick, phase=phase, field_group=field_group)
        self.comm.allReduce(
            int(send.data.ptr),
            int(recv.data.ptr),
            int(send.size),
            _dtype_constant(self.nccl, send.dtype),
            int(op_map[op]),
            int(stream.ptr),
        )
        return recv

    def broadcast(
        self,
        send: Any,
        recv: Any,
        *,
        root: int,
        stream: Any,
        tick: int = -1,
        phase: str = "collective",
        field_group: str = "global",
    ) -> Any:
        if not send.flags.c_contiguous:
            raise ValueError("NCCL broadcast source must be contiguous")
        if send.size != recv.size or send.dtype != recv.dtype:
            raise ValueError("broadcast send/recv count and dtype must match")
        self._record("broadcast", send, root, tick, phase=phase, field_group=field_group)
        self.comm.broadcast(
            int(send.data.ptr),
            int(recv.data.ptr),
            int(send.size),
            _dtype_constant(self.nccl, send.dtype),
            int(root),
            int(stream.ptr),
        )
        return recv

    def all_gather(
        self,
        send: Any,
        recv: Any,
        *,
        stream: Any,
        tick: int = -1,
        phase: str = "collective",
        field_group: str = "global",
    ) -> Any:
        if not send.flags.c_contiguous:
            raise ValueError("NCCL all_gather source must be contiguous")
        expected = int(send.size) * self.world_size
        if recv.size != expected or recv.dtype != send.dtype:
            raise ValueError(f"all_gather receive must have {expected} elements of {send.dtype}")
        self._record("all_gather", send, -1, tick, phase=phase, field_group=field_group)
        self.comm.allGather(
            int(send.data.ptr),
            int(recv.data.ptr),
            int(send.size),
            _dtype_constant(self.nccl, send.dtype),
            int(stream.ptr),
        )
        return recv

    def ledger_records(self) -> list[dict[str, Any]]:
        return [record.__dict__.copy() for record in self.ledger]

    def ledger_hash(self) -> str:
        payload = self.ledger_records()
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def abort(self) -> None:
        if self.closed:
            return
        abort = getattr(self.comm, "abort", None)
        if callable(abort):
            abort()
        else:
            self.comm.destroy()
        self.closed = True

    def close(self) -> None:
        if self.closed:
            return
        self.comm.destroy()
        self.closed = True
