from __future__ import annotations

from collections.abc import Callable

from fastapi import Request, Response
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import SessionSpec

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.dependencies import load_base_env
from openhands.agent_server.event_service import EventService
from openhands.agent_server.pyromind_auth import (
    CurrentLoginUser,
    get_current_login_user_from_token,
    parse_auth_token_from_cookie_header,
)
from openhands.agent_server.pyromind_router import (
    PyromindCreateConversationRequest,
    PyromindLLMConfig,
    create_pyromind_conversation,
)


class LegacyPyromindSessionFactory:
    """Call the pre-migration Pyromind route implementation unchanged."""

    def __init__(
        self,
        conversation_service_provider: Callable[[], ConversationService],
    ) -> None:
        self._conversation_service_provider = conversation_service_provider

    async def create(
        self,
        spec: SessionSpec,
        context: RequestContext,
    ) -> tuple[str, EventService]:
        request = request_from_context(context)
        response = Response()
        message_parts: list[str] = []
        for block in spec.initial_message:
            if not isinstance(block, TextContent):
                raise ValueError("the legacy OpenHands create route accepts text only")
            message_parts.append(block.text)
        body = PyromindCreateConversationRequest(
            llm=PyromindLLMConfig.model_validate(spec.model_configuration),
            message="\n".join(message_parts) if message_parts else None,
            workflow_xyflow=spec.workflow_xyflow,
            extra=dict(spec.extra),
        )
        service = self._conversation_service_provider()
        info = await create_pyromind_conversation(
            request,
            body,
            response,
            service,
        )
        event_service = await service.get_event_service(
            info.id,
            user_id=_service_user_id(context),
        )
        if event_service is None:
            raise RuntimeError(f"OpenHands event service not found: {info.id}")
        return info.id.hex, event_service


def request_from_context(context: RequestContext) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    for name, value in (
        ("cookie", context.cookie),
        ("authorization", context.authorization),
        ("x-cluster", context.x_cluster),
        ("accept-language", context.accept_language),
    ):
        if value:
            headers.append((name.encode(), value.encode()))
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v2/pyromind/conversations",
            "raw_path": b"/api/v2/pyromind/conversations",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 0),
            "server": ("127.0.0.1", 0),
        }
    )
    current_user = _current_user(context)
    if current_user is not None:
        request.state.current_user = current_user
    load_base_env(request)
    return request


def _current_user(context: RequestContext) -> CurrentLoginUser | None:
    token = parse_auth_token_from_cookie_header(context.cookie)
    current_user = get_current_login_user_from_token(token)
    if current_user is not None:
        return current_user.model_copy(
            update={"cookie": context.cookie, "x_cluster": context.x_cluster}
        )
    try:
        user_id = int(context.user_id)
    except ValueError:
        return None
    return CurrentLoginUser(
        username=f"debug-user-{user_id}",
        email=f"debug-user-{user_id}@example.test",
        user_id=user_id,
        group_id=0,
        cookie=context.cookie,
        x_cluster=context.x_cluster,
    )


def _service_user_id(context: RequestContext) -> str | None:
    return None if context.user_id == "anonymous" else context.user_id
