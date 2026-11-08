from __future__ import annotations

from fastapi import Request
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.dependencies import get_current_user_id
from openhands.tools.utils.platform_env import cluster_header


def request_context(request: Request) -> RequestContext:
    """Copy request-scoped identity and headers without persisting credentials."""
    return RequestContext(
        user_id=get_current_user_id(request) or "anonymous",
        cookie=request.headers.get("cookie"),
        authorization=request.headers.get("authorization"),
        x_cluster=cluster_header(request.headers),
        accept_language=request.headers.get("accept-language"),
    )


def evaluation_request_context(request: Request) -> RequestContext:
    """Forward Storage credentials; the optional user label is not verified."""
    return RequestContext(
        user_id=request.headers.get("x-pyromind-debug-user-id", "").strip()
        or "anonymous",
        cookie=request.headers.get("cookie"),
        authorization=request.headers.get("authorization"),
        x_cluster=cluster_header(request.headers),
        accept_language=request.headers.get("accept-language"),
    )
