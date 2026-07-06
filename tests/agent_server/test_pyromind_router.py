from pathlib import Path
from typing import cast

import pytest
from fastapi import Response, status
from starlette.requests import Request

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.models import ConversationInfo
from openhands.agent_server.pyromind_auth import (
    PYROMIND_AUTH_COOKIE_NAME,
    CurrentLoginUser,
    get_debug_current_login_user_by_conversation,
)
from openhands.agent_server.pyromind_constants import (
    PYROMIND_APP_TAG_KEY,
    PYROMIND_APP_TAG_VALUE,
)
from openhands.agent_server.pyromind_router import (
    PyromindCreateConversationRequest,
    PyromindLLMConfig,
    _build_debug_context_headers,
    _build_workflow_validation_tool,
    _get_validation_cookie_header,
    create_pyromind_conversation,
)
from openhands.sdk.conversation.request import StartConversationRequest
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.tools.workflow import DslToXyflowTool, ValidateWorkflowDslTool


_REMOVED_WORKFLOW_TOOL = "publish" + "_workflow"


class _FakeConversationService:
    def __init__(self, conversations_dir: Path) -> None:
        self.conversations_dir = conversations_dir
        self.start_request: StartConversationRequest | None = None

    async def start_conversation(
        self, request: StartConversationRequest
    ) -> tuple[ConversationInfo, bool]:
        self.start_request = request
        assert request.conversation_id is not None
        return (
            ConversationInfo(
                id=request.conversation_id,
                agent=request.agent,
                workspace=request.workspace,
                execution_status=ConversationExecutionStatus.IDLE,
            ),
            True,
        )


def _make_request(headers: dict[str, str] | None = None) -> Request:
    raw_headers = [
        (name.lower().encode(), value.encode())
        for name, value in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/pyromind/conversations",
            "headers": raw_headers,
        }
    )


@pytest.mark.asyncio
async def test_pyromind_conversation_uses_conversation_workspace(tmp_path):
    knowledge_base = tmp_path / "knowledge"
    knowledge_base.mkdir()
    service = _FakeConversationService(tmp_path / "conversations")
    response = Response()
    cookie_header = f"{PYROMIND_AUTH_COOKIE_NAME}=session-token; other=value"

    info = await create_pyromind_conversation(
        _make_request(
            {
                "cookie": cookie_header,
                "x-cluster": "us-west-1#pre",
            }
        ),
        PyromindCreateConversationRequest(
            llm=PyromindLLMConfig(model="gpt-4o", api_key="test-key"),
            extra={
                "knowledge_base_path": str(knowledge_base),
                "skills_path": str(tmp_path / "missing-skills"),
            },
        ),
        response,
        conversation_service=cast(ConversationService, service),
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert service.start_request is not None
    assert service.start_request.conversation_id == info.id
    expected_dir = service.conversations_dir / info.id.hex
    assert Path(service.start_request.workspace.working_dir) == expected_dir
    assert expected_dir.is_dir()

    tool_names = {tool.name for tool in service.start_request.agent.tools}
    assert "grep" in tool_names
    assert "file_editor" in tool_names
    assert DslToXyflowTool.name in tool_names
    assert ValidateWorkflowDslTool.name in tool_names
    assert _REMOVED_WORKFLOW_TOOL not in tool_names
    validation_tool = next(
        tool
        for tool in service.start_request.agent.tools
        if tool.name == ValidateWorkflowDslTool.name
    )
    assert validation_tool.params == {
        "headers": {"x-cluster": "us-west-1#pre"},
        "secret_headers": {"cookie": "PYROMIND_VALIDATE_AUTH_COOKIE"},
    }
    assert "session-token" not in str(validation_tool.params)
    assert (
        service.start_request.secrets["PYROMIND_VALIDATE_AUTH_COOKIE"].get_value()
        == cookie_header
    )
    assert service.start_request.tags == {PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE}


@pytest.mark.asyncio
async def test_pyromind_conversation_binds_login_context(tmp_path):
    knowledge_base = tmp_path / "knowledge"
    knowledge_base.mkdir()
    service = _FakeConversationService(tmp_path / "conversations")
    response = Response()
    request = _make_request()
    request.state.current_user = CurrentLoginUser(
        username="debug-user-42",
        email="debug-user-42@example.test",
        user_id=42,
        cookie="auth_token=context-token",
        x_cluster="context-cluster",
    )

    info = await create_pyromind_conversation(
        request,
        PyromindCreateConversationRequest(
            llm=PyromindLLMConfig(model="gpt-4o", api_key="test-key"),
            extra={
                "knowledge_base_path": str(knowledge_base),
                "skills_path": str(tmp_path / "missing-skills"),
            },
        ),
        response,
        conversation_service=cast(ConversationService, service),
    )

    assert service.start_request is not None
    assert service.start_request.user_id == "42"
    bound_user = get_debug_current_login_user_by_conversation(info.id)
    assert bound_user == request.state.current_user


def test_validation_cookie_header_keeps_full_cookie_in_prod(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")

    cookie_header = f"{PYROMIND_AUTH_COOKIE_NAME}=session-token; other=value"
    request = _make_request({"cookie": cookie_header})

    assert _get_validation_cookie_header(request) == cookie_header


def test_workflow_validation_tool_uses_user_context_headers():
    request = _make_request(
        {
            "cookie": f"{PYROMIND_AUTH_COOKIE_NAME}=request-token",
            "x-cluster": "request-cluster",
        }
    )
    request.state.current_user = CurrentLoginUser(
        username="debug-user-42",
        email="debug-user-42@example.test",
        user_id=42,
        cookie=f"{PYROMIND_AUTH_COOKIE_NAME}=context-token; other=value",
        x_cluster="context-cluster",
    )

    tool, secrets = _build_workflow_validation_tool(request, {})

    assert tool.params == {
        "headers": {"x-cluster": "context-cluster"},
        "secret_headers": {"cookie": "PYROMIND_VALIDATE_AUTH_COOKIE"},
    }
    assert (
        secrets["PYROMIND_VALIDATE_AUTH_COOKIE"].get_value()
        == f"{PYROMIND_AUTH_COOKIE_NAME}=context-token; other=value"
    )


def test_build_debug_context_headers_uses_current_user_context():
    current_user = CurrentLoginUser(
        username="debug-user-42",
        email="debug-user-42@example.test",
        user_id=42,
        cookie="auth_token=session-token; other=value",
        x_cluster="us-west-1#pre",
    )

    assert _build_debug_context_headers(current_user) == {
        "cookie": "auth_token=session-token; other=value",
        "x-cluster": "us-west-1#pre",
    }
