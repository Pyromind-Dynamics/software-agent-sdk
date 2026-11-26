from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import ExternalTaskState


@dataclass(frozen=True)
class TaskCheck:
    conversation_id: str
    task_id: str
    failures: int
    lease: str


@dataclass(frozen=True)
class TaskStatus:
    status: str
    error_summary: str | None = None


class TaskStatusQuery(Protocol):
    async def query(self, task_id: str, context: RequestContext) -> TaskStatus: ...


class TaskMonitorStore(Protocol):
    def register(self, conversation_id: str, task_id: str, due_at: float) -> None: ...

    def claim(self, now: float, lease_seconds: float) -> TaskCheck | None: ...

    def postpone(self, task: TaskCheck, due_at: float, failures: int) -> None: ...

    def remove(self, task: TaskCheck) -> None: ...


class TaskMonitorHost(Protocol):
    def task_monitor_context(self, conversation_id: str) -> RequestContext | None: ...

    def task_monitor_state(
        self, conversation_id: str, task_id: str
    ) -> ExternalTaskState | None: ...

    async def resume_monitored_task(self, conversation_id: str) -> None: ...

    async def deliver_external_task_status(
        self,
        conversation_id: str,
        *,
        task_id: str,
        status: str,
        error_summary: str | None = None,
        auto_run: bool = True,
        trusted_platform: bool = False,
    ) -> object: ...
