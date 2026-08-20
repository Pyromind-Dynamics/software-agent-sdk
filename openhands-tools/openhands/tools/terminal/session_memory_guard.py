"""Per-conversation memory budget enforced without cgroup delegation.

The kernel only caps memory per-process (``RLIMIT_AS``) or per-cgroup;
without write access to the pod's cgroup tree neither is per-conversation.
This module approximates a hard cap: every process spawned by a sandbox
shell inherits a unique ``OH_SANDBOX_SESSION_MARK``, and a background
watcher sums their ``VmRSS`` from ``/proc``. Once the total crosses
``OH_SANDBOX_MEMORY_LIMIT`` (default 500M) the whole session is SIGKILLed.

Polling means a brief overshoot is possible, but each individual process
is already capped at the same limit by ``RLIMIT_AS``; a single command
therefore cannot blow past roughly one limit, and the watcher bounds the
multi-process sum.
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from collections.abc import Callable

from openhands.sdk.logger import get_logger
from openhands.tools.terminal.sandbox import parse_memory_limit


logger = get_logger(__name__)

MARKER_ENV = "OH_SANDBOX_SESSION_MARK"
LIMIT_ENV = "OH_SANDBOX_MEMORY_LIMIT"
MONITOR_ENV = "OH_SANDBOX_SESSION_MEMORY_MONITOR"
POLL_INTERVAL_ENV = "OH_SANDBOX_SESSION_POLL_INTERVAL"
PROCFS_ROOT_ENV = "OH_SANDBOX_PROCFS_ROOT"

DEFAULT_LIMIT_BYTES = 500 * 1024**2
DEFAULT_POLL_INTERVAL_SECONDS = 1.0

_VMRSS_RE = re.compile(rb"^VmRSS:\s+(\d+) kB", re.MULTILINE)


def session_memory_limit_bytes() -> int:
    """RSS budget for one sandbox session; ``0`` disables the watcher."""
    raw = os.environ.get(LIMIT_ENV, "").strip()
    if not raw:
        return DEFAULT_LIMIT_BYTES
    try:
        return parse_memory_limit(raw)
    except ValueError:
        logger.warning("invalid %s=%r ignored; using default limit", LIMIT_ENV, raw)
        return DEFAULT_LIMIT_BYTES


def session_memory_monitor_enabled() -> bool:
    raw = os.environ.get(MONITOR_ENV, "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return session_memory_limit_bytes() > 0


def memory_guard_poll_interval() -> float:
    raw = os.environ.get(POLL_INTERVAL_ENV, "").strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_POLL_INTERVAL_SECONDS
    return value if value > 0 else DEFAULT_POLL_INTERVAL_SECONDS


def procfs_root() -> str:
    return os.environ.get(PROCFS_ROOT_ENV, "").strip() or "/proc"


def new_session_marker() -> str:
    return uuid.uuid4().hex


def _marker_entry(marker: str) -> bytes:
    return f"{MARKER_ENV}={marker}".encode("utf-8", "ignore")


def find_marked_pids(marker: str, root: str = "/proc") -> list[int]:
    """Return PIDs whose /proc environment carries exactly this marker."""
    expected = _marker_entry(marker)
    pids: list[int] = []
    try:
        entries = os.listdir(root)
    except OSError:
        return pids
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(root, name, "environ"), "rb") as fh:
                env = fh.read()
        except OSError:
            continue
        if expected in env.split(b"\0"):
            pids.append(int(name))
    return pids


def sum_vmrss_kb(pids: list[int], root: str = "/proc") -> int:
    """Sum VmRSS (kB) across ``pids``; unreadable entries are skipped."""
    total = 0
    for pid in pids:
        try:
            with open(os.path.join(root, str(pid), "status"), "rb") as fh:
                status = fh.read()
        except OSError:
            continue
        match = _VMRSS_RE.search(status)
        if match is not None:
            total += int(match.group(1))
    return total


class SessionMemoryGuard(threading.Thread):
    """Background watcher that SIGKILLs a sandbox session over its RSS budget."""

    def __init__(
        self,
        marker: str,
        on_exceed: Callable[[list[int]], None],
        *,
        root: str | None = None,
        limit_bytes: int | None = None,
        interval: float | None = None,
    ) -> None:
        super().__init__(name=f"session-memory-guard-{marker[:8]}", daemon=True)
        self._marker = marker
        self._on_exceed = on_exceed
        self._root = root or procfs_root()
        self._limit_bytes = (
            limit_bytes if limit_bytes is not None else session_memory_limit_bytes()
        )
        self._interval = (
            interval if interval is not None else memory_guard_poll_interval()
        )
        self._stop = threading.Event()
        self.exceeded = False

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            pids = find_marked_pids(self._marker, self._root)
            rss_kb = sum_vmrss_kb(pids, self._root)
            if pids and rss_kb * 1024 > self._limit_bytes:
                logger.warning(
                    "session memory guard: RSS %d kB exceeds %d-byte cap; "
                    "killing %d process(es)",
                    rss_kb,
                    self._limit_bytes,
                    len(pids),
                )
                self.exceeded = True
                try:
                    self._on_exceed(pids)
                except Exception:
                    logger.exception("session memory guard: on_exceed callback failed")
                break
            self._stop.wait(self._interval)
