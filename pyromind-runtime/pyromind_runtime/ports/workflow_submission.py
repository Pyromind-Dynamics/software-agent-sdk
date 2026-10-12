from __future__ import annotations

from typing import Protocol

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext


class WorkflowSubmissionPort(Protocol):
    async def prepare(
        self, workflow_path: str, context: RequestContext
    ) -> JsonObject: ...

    async def submit(
        self, workflow: JsonObject, conversation_id: str, context: RequestContext
    ) -> str: ...
