from __future__ import annotations

from types import SimpleNamespace

from owl.gpu.run_context import PersistentOWLDeviceRun


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Streams:
    def __init__(self) -> None:
        self.synchronized = False

    def synchronize_all(self) -> None:
        self.synchronized = True


def test_counterfactual_branch_close_has_no_shared_report_side_effects(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    streams = _Streams()
    writer = _Closable()
    observer = _Closable()
    run = object.__new__(PersistentOWLDeviceRun)
    run.closed = False
    run.pending_metric_tickets = []
    run.async_writer = writer
    run.visual_controller = None
    run.counterfactual_source_observer = observer
    run.streams = streams
    run.ds = SimpleNamespace(
        metadata={"counterfactual_suppress_close_reports": True}
    )

    run.close(checkpoint=False)

    assert run.closed is True
    assert streams.synchronized is True
    assert writer.closed is True
    assert observer.closed is True
    assert not (tmp_path / "runs").exists()
