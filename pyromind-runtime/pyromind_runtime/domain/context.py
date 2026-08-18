from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Ephemeral request credentials; this object must never be persisted."""

    user_id: str
    cookie: str | None = None
    authorization: str | None = None
    x_cluster: str | None = None
    accept_language: str | None = None
