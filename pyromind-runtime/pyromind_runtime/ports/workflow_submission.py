from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.task_correlation import TaskCorrelation


class WorkflowSubmissionPort(Protocol):
    async def prepare(
        self, workflow_path: str, context: RequestContext
    ) -> JsonObject: ...

    async def submit(
        self,
        workflow: JsonObject,
        conversation_id: str,
        context: RequestContext,
        *,
        correlation: TaskCorrelation | None = None,
    ) -> str: ...


class ExternalTaskQueryPort(Protocol):
    async def query(self, task_id: str, context: RequestContext) -> JsonObject: ...
