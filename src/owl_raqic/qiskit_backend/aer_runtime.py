"""Bounded Qiskit Aer job execution.

The installed Aer API returns an asynchronous ``AerJob`` backed by a module-
level thread pool.  Long-lived applications should not leave that executor as
an unowned process-global resource.  OWL submits each Aer job through a small
owned executor, waits for the result, and shuts the executor down before
returning.  A lock protects the brief compatibility swap required by Aer
versions whose ``AerBackend.run`` does not forward its documented ``executor``
run option to ``AerJob``.
"""

from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Any

_AER_SUBMIT_LOCK = RLock()


def run_aer_job(simulator: Any, circuits: Any, /, **run_options: Any) -> Any:
    """Run one Aer job and return its completed ``Result``.

    The helper owns and closes the worker executor.  Submission is serialized
    only while Aer captures the executor reference; simulation itself may run
    concurrently when callers use separate higher-level worker tasks.
    """

    import qiskit_aer.jobs.aerjob as aerjob_module

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="owl-aer")
    job = None
    try:
        with _AER_SUBMIT_LOCK:
            previous = aerjob_module.DEFAULT_EXECUTOR
            aerjob_module.DEFAULT_EXECUTOR = executor
            try:
                job = simulator.run(circuits, **run_options)
            finally:
                aerjob_module.DEFAULT_EXECUTOR = previous
        return job.result()
    finally:
        if job is not None:
            with contextlib.suppress(Exception):
                job.cancel()
        executor.shutdown(wait=True, cancel_futures=False)
