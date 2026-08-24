"""Tests for the per-conversation RSS memory guard."""

import time
from pathlib import Path

import pytest

from openhands.tools.terminal.session_memory_guard import (
    LIMIT_ENV,
    MONITOR_ENV,
    POLL_INTERVAL_ENV,
    SessionMemoryGuard,
    find_marked_pids,
    memory_guard_poll_interval,
    new_session_marker,
    session_memory_limit_bytes,
    session_memory_monitor_enabled,
    sum_vmrss_kb,
)


def _add_proc(root: Path, pid: int, marker: str | None, rss_kb: int) -> None:
    proc = root / str(pid)
    proc.mkdir(parents=True, exist_ok=True)
    if marker is not None:
        (proc / "environ").write_bytes(
            b"PATH=/usr/bin\0" + f"OH_SANDBOX_SESSION_MARK={marker}".encode()
        )
    else:
        (proc / "environ").write_bytes(b"PATH=/usr/bin\0")
    (proc / "status").write_text(f"Name:\tpython\nVmRSS:\t{rss_kb} kB\n")


def test_find_marked_pids_and_sum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "proc"
    marker = new_session_marker()
    _add_proc(root, 100, marker, 512 * 1024)
    _add_proc(root, 101, marker, 128 * 1024)
    _add_proc(root, 102, None, 999999)  # different/other env -> ignored
    _add_proc(root, 103, "othermarker", 5)
    pids = find_marked_pids(marker, root=str(root))
    assert sorted(pids) == [100, 101]
    assert sum_vmrss_kb(pids, root=str(root)) == (512 + 128) * 1024


def test_guard_kills_on_exceed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "proc"
    marker = new_session_marker()
    _add_proc(root, 100, marker, 600 * 1024)
    killed: list[list[int]] = []

    def on_exceed(pids: list[int]) -> None:
        killed.append(pids)

    guard = SessionMemoryGuard(
        marker,
        on_exceed,
        root=str(root),
        limit_bytes=500 * 1024**2,
        interval=0.02,
    )
    guard.start()
    try:
        deadline = time.time() + 5
        while not guard.exceeded and time.time() < deadline:
            time.sleep(0.01)
    finally:
        guard.stop()
        guard.join(timeout=1)
    assert guard.exceeded is True
    assert killed == [[100]]


def test_guard_not_triggered_under_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "proc"
    marker = new_session_marker()
    _add_proc(root, 200, marker, 300 * 1024)
    killed: list[list[int]] = []
    guard = SessionMemoryGuard(
        marker,
        killed.append,
        root=str(root),
        limit_bytes=500 * 1024**2,
        interval=0.02,
    )
    guard.start()
    try:
        time.sleep(0.1)
    finally:
        guard.stop()
        guard.join(timeout=1)
    assert guard.exceeded is False
    assert killed == []


def test_limit_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LIMIT_ENV, raising=False)
    assert session_memory_limit_bytes() == 500 * 1024**2


def test_limit_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LIMIT_ENV, "256M")
    assert session_memory_limit_bytes() == 256 * 1024**2


def test_limit_zero_disables_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LIMIT_ENV, "0")
    assert session_memory_monitor_enabled() is False


def test_monitor_off_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LIMIT_ENV, raising=False)
    monkeypatch.setenv(MONITOR_ENV, "0")
    assert session_memory_monitor_enabled() is False


def test_poll_interval_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(POLL_INTERVAL_ENV, raising=False)
    assert memory_guard_poll_interval() == 1.0
    monkeypatch.setenv(POLL_INTERVAL_ENV, "0.5")
    assert memory_guard_poll_interval() == 0.5
    monkeypatch.setenv(POLL_INTERVAL_ENV, "banana")
    assert memory_guard_poll_interval() == 1.0
