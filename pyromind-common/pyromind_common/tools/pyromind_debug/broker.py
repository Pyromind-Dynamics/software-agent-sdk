"""In-process registry that lets a blocked tool call wait for an external
debug result and lets a webhook (or the mock platform) deliver that result.

This is intentionally simple: one process, one dict guarded by a lock, one
``threading.Event`` per in-flight debug task. It is not meant to survive an
agent-server restart or to be shared across multiple server instances; see
the "Known trade-offs" section of the debug-loop plan for that limitation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal


DebugStatus = Literal["passed", "failed"]


@dataclass(frozen=True)
class DebugResult:
    """Outcome of a single debug run, as reported by the platform/webhook."""

    status: DebugStatus
    error_log: str | None = None


class DebugResultBroker:
    """Registers waiters for a ``task_id`` and wakes them when a result arrives."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}
        self._results: dict[str, DebugResult] = {}

    def register(self, task_id: str) -> None:
        """Register a new task and create the event a waiter will block on.

        Must be called before :meth:`wait` for the same ``task_id``, and
        before the submitting side can possibly call :meth:`resolve` (i.e.
        register before submitting to the platform/mock).
        """
        with self._lock:
            self._events[task_id] = threading.Event()

    def resolve(
        self, task_id: str, status: DebugStatus, error_log: str | None = None
    ) -> bool:
        """Deliver a result for ``task_id`` and wake any waiter.

        Returns False if no waiter is registered for ``task_id`` (e.g. the
        wait already timed out and cleaned up, or the task_id is unknown).
        Called from the debug callback endpoint (real platform) or from the
        mock platform's background timer thread.
        """
        with self._lock:
            event = self._events.get(task_id)
            if event is None:
                return False
            self._results[task_id] = DebugResult(status=status, error_log=error_log)
        event.set()
        return True

    def wait(self, task_id: str, timeout: float) -> DebugResult | None:
        """Block the calling thread until a result arrives or ``timeout`` elapses.

        Always cleans up the registration for ``task_id`` before returning,
        so a late/duplicate :meth:`resolve` call after a timeout is a no-op
        (and reported as such via its own False return value).

        Returns None on timeout.
        """
        with self._lock:
            event = self._events.get(task_id)
        if event is None:
            return None
        completed = event.wait(timeout)
        with self._lock:
            self._events.pop(task_id, None)
            result = self._results.pop(task_id, None)
        return result if completed else None


_broker_lock = threading.Lock()
_broker: DebugResultBroker | None = None


def get_debug_result_broker() -> DebugResultBroker:
    """Return the process-wide singleton broker."""
    global _broker
    with _broker_lock:
        if _broker is None:
            _broker = DebugResultBroker()
        return _broker
