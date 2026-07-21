from __future__ import annotations

import json
import os
import signal
import socket
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from owl.experiments.progress import atomic_write_json


@dataclass(frozen=True)
class ProcessControlRecord:
    pid: int
    pgid: int
    hostname: str
    started_at: str
    command: tuple[str, ...]
    run_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "hostname": self.hostname,
            "started_at": self.started_at,
            "command": list(self.command),
            "run_root": self.run_root,
        }


class RunLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise RuntimeError(f"experiment lock already exists: {self.path}") from exc
        os.write(self.fd, f"{os.getpid()}\n".encode())
        os.fsync(self.fd)

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()


def write_control_record(run_root: Path, command: list[str]) -> ProcessControlRecord:
    record = ProcessControlRecord(
        pid=os.getpid(),
        pgid=os.getpgrp(),
        hostname=socket.gethostname(),
        started_at=datetime.now(UTC).isoformat(),
        command=tuple(command),
        run_root=str(run_root),
    )
    atomic_write_json(run_root / "control.json", record.to_dict())
    return record


def read_control_record(run_root: str | Path) -> ProcessControlRecord:
    value = json.loads((Path(run_root) / "control.json").read_text(encoding="utf-8"))
    return ProcessControlRecord(
        pid=int(value["pid"]),
        pgid=int(value["pgid"]),
        hostname=str(value["hostname"]),
        started_at=str(value["started_at"]),
        command=tuple(str(item) for item in value["command"]),
        run_root=str(value["run_root"]),
    )


def process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_process_group(run_root: str | Path, timeout: float = 30.0) -> dict[str, Any]:
    record = read_control_record(run_root)
    if not process_alive(record.pid):
        return {"stopped": True, "already_exited": True, "pid": record.pid, "pgid": record.pgid}
    os.killpg(record.pgid, signal.SIGTERM)
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if not process_alive(record.pid):
            return {"stopped": True, "escalated": False, "pid": record.pid, "pgid": record.pgid}
        time.sleep(0.25)
    os.killpg(record.pgid, signal.SIGKILL)
    return {"stopped": True, "escalated": True, "pid": record.pid, "pgid": record.pgid}
