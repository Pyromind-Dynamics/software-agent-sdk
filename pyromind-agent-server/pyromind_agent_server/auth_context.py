from __future__ import annotations

from fastapi import Request
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.dependencies import get_current_user_id
from openhands.agent_server.pyromind_auth import CurrentLoginUser


def request_context(request: Request) -> RequestContext:
    """Copy request-scoped identity and headers without persisting credentials."""
    current_user = getattr(request.state, "current_user", None)
    x_cluster = request.headers.get("x-cluster")
    if isinstance(current_user, CurrentLoginUser) and current_user.x_cluster:
        x_cluster = current_user.x_cluster
    return RequestContext(
        user_id=get_current_user_id(request) or "anonymous",
        cookie=request.headers.get("cookie"),
        authorization=request.headers.get("authorization"),
        x_cluster=x_cluster,
        accept_language=request.headers.get("accept-language"),
    )
