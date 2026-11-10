from __future__ import annotations

from dataclasses import dataclass

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.task_correlation import TaskOrigin


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    trace_id: str
    parent_span_id: str


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Ephemeral request credentials; this object must never be persisted."""

    user_id: str
    cookie: str | None = None
    authorization: str | None = None
    x_cluster: str | None = None
    accept_language: str | None = None
    profile_id: str | None = None
    task_origin: TaskOrigin | None = None
    execution_trace: ExecutionTrace | None = None
    input_response_metadata: JsonObject | None = None
