from collections.abc import Callable, Coroutine
from typing import Any, Protocol

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext


type WorkflowCall = Callable[
    [str, str, JsonObject, str, RequestContext, str | None],
    Coroutine[Any, Any, JsonObject],
]
type WorkflowEventSink = Callable[[JsonObject], Coroutine[Any, Any, JsonObject | None]]


class WorkflowBackend(Protocol):
    async def close(self) -> None: ...

    async def invoke(
        self, scope: str, action: str, arguments: JsonObject, request_id: str
    ) -> JsonObject: ...

    async def execute(
        self, scope: str, run_id: str, emit: WorkflowEventSink
    ) -> JsonObject: ...
