from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.context import ExecutionTrace


class ExecutionObserver(Protocol):
    def start(self, conversation_id: str, run_id: str) -> ExecutionTrace: ...

    def finish(self, conversation_id: str, run_id: str, status: str) -> None: ...

    def close(self) -> None: ...
