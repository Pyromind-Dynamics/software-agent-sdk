from __future__ import annotations

from dataclasses import dataclass

from pyromind_runtime.domain.task_correlation import TaskOrigin


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
