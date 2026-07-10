from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from fastapi import Response, status
from pydantic import ValidationError
from starlette.requests import Request

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.dependencies import load_base_env
from openhands.agent_server.event_service import EventService
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
    PYROMIND_AUTH_TOKEN_SECRET,
    PyromindCreateConversationRequest,
    PyromindLLMConfig,
    PyromindSendMessageRequest,
    _build_debug_context_headers,
    _build_workflow_run_tool,
    _build_pyromind_storage_tools,
    _build_workflow_validation_tool,
    _get_validation_cookie_header,
    _workflow_dsl_from_xyflow,
    apply_pyromind_validation_context,
    create_pyromind_conversation,
    send_pyromind_message,
)
from openhands.sdk.conversation.request import StartConversationRequest
from openhands.sdk.conversation.state import ConversationExecutionStatus

from openhands.tools.workflow import (
    DslToXyflowTool,
    RunWorkflowTool,
    ValidateWorkflowDslTool,
)

from openhands.sdk.llm.message import Message
from openhands.tools.pyromind_dataset import (
    PreviewDatasetTool,
    UploadFileToPyromindTool,
)
from openhands.tools.pyromind_dataset.definition import (
    PYROMIND_STORAGE_AUTH_COOKIE_SECRET,
    PYROMIND_STORAGE_HEADERS_STATE_KEY,
)
from openhands.tools.workflow import DslToXyflowTool, ValidateWorkflowDslTool
from openhands.tools.workflow.validate_workflow_dsl import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)


_REMOVED_WORKFLOW_TOOL = "publish" + "_workflow"


class _FakeConversationService:
    def __init__(self, conversations_dir: Path) -> None:
        self.conversations_dir = conversations_dir
        self.start_request: StartConversationRequest | None = None
        self.event_service = _FakeInitialMessageEventService()
        self.info: ConversationInfo | None = None

    async def start_conversation(
        self, request: StartConversationRequest
    ) -> tuple[ConversationInfo, bool]:
        self.start_request = request
        assert request.conversation_id is not None
        self.info = ConversationInfo(
            id=request.conversation_id,
            agent=request.agent,
            workspace=request.workspace,
            execution_status=ConversationExecutionStatus.IDLE,
        )
        return (self.info, True)

    async def get_event_service(
        self,
        conversation_id: UUID,
        user_id: str | None = None,
    ) -> EventService | None:
        assert self.info is not None
        assert conversation_id == self.info.id
        return cast(EventService, self.event_service)

    async def get_conversation(
        self,
        conversation_id: UUID,
        user_id: str | None = None,
    ) -> ConversationInfo | None:
        assert self.info is not None
        assert conversation_id == self.info.id
        return self.info


class _FakeInitialMessageEventService:
    def __init__(self) -> None:
        self.sent_message: Message | None = None
        self.run: bool | None = None
        self.workflow_dsl_snapshot: str | None = None
        self.workflow_xyflow_snapshot: dict[str, Any] | None = None

    async def send_message(
        self,
        message: Message,
        run: bool = False,
        _from_goal_loop: bool = False,
        extended_content: list[Any] | None = None,
        workflow_dsl_snapshot: str | None = None,
        workflow_xyflow_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.sent_message = message
        self.run = run
        self.workflow_dsl_snapshot = workflow_dsl_snapshot
        self.workflow_xyflow_snapshot = workflow_xyflow_snapshot


class _FakeEventService:
    def __init__(self, tags: dict[str, str]) -> None:
        self.stored = type("FakeStoredConversation", (), {"tags": tags})()
        self.secrets: dict[str, str] = {}
        self.agent_state: dict[str, object] = {}

    async def update_secrets(self, secrets: dict[str, str]) -> None:
        self.secrets.update(secrets)

    async def update_agent_state(self, values: dict[str, object]) -> None:
        self.agent_state.update(values)


class _FakePyromindMessageEventService(_FakeEventService):
    def __init__(self, tags: dict[str, str], working_dir: Path) -> None:
        super().__init__(tags)
        self.sent_message: Message | None = None
        self.run: bool | None = None
        self.workflow_dsl_snapshot: str | None = None
        self.workflow_xyflow_snapshot: dict[str, Any] | None = None
        self.extended_content: list[Any] | None = None
        workspace = type("FakeWorkspace", (), {"working_dir": str(working_dir)})()
        self._conversation = type("FakeConversation", (), {"workspace": workspace})()

    def get_conversation(self):
        return self._conversation

    async def send_message(
        self,
        message: Message,
        run: bool = False,
        _from_goal_loop: bool = False,
        extended_content: list[Any] | None = None,
        workflow_dsl_snapshot: str | None = None,
        workflow_xyflow_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.sent_message = message
        self.run = run
        self.extended_content = extended_content
        self.workflow_dsl_snapshot = workflow_dsl_snapshot
        self.workflow_xyflow_snapshot = workflow_xyflow_snapshot


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

    request = _make_request(
        {
            "cookie": cookie_header,
            "x-cluster": "us-west-1#pre",
        }
    )
    load_base_env(request)

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
    assert RunWorkflowTool.name in tool_names
    assert ValidateWorkflowDslTool.name in tool_names
    assert PreviewDatasetTool.name in tool_names
    assert UploadFileToPyromindTool.name in tool_names
    assert _REMOVED_WORKFLOW_TOOL not in tool_names
    validation_tool = next(
        tool
        for tool in service.start_request.agent.tools
        if tool.name == ValidateWorkflowDslTool.name
    )
    run_tool = next(
        tool
        for tool in service.start_request.agent.tools
        if tool.name == RunWorkflowTool.name
    )
    assert validation_tool.params == {
        "headers": {"x-cluster": "us-west-1#pre"},
        "secret_headers": {"cookie": "PYROMIND_VALIDATE_AUTH_COOKIE"},
    }
    assert run_tool.params == {
        "current_user": None,
        "env": "pre",
        "cluster": "us-west-1",
        "headers": {
            "x-cluster": "us-west-1#pre",
            "request-app": "openhands",
        },
    }
    assert "session-token" not in str(validation_tool.params)
    assert "secret_headers" not in run_tool.params
    assert (
        service.start_request.secrets["PYROMIND_VALIDATE_AUTH_COOKIE"].get_value()
        == cookie_header
    )

    preview_tool = next(
        tool
        for tool in service.start_request.agent.tools
        if tool.name == PreviewDatasetTool.name
    )
    upload_tool = next(
        tool
        for tool in service.start_request.agent.tools
        if tool.name == UploadFileToPyromindTool.name
    )
    assert preview_tool.params == {
        "headers": {"x-cluster": "us-west-1#pre"},
        "secret_headers": {"cookie": "PYROMIND_STORAGE_AUTH_COOKIE"},
    }
    assert upload_tool.params == preview_tool.params
    assert "session-token" not in str(preview_tool.params)
    assert (
        service.start_request.secrets["PYROMIND_STORAGE_AUTH_COOKIE"].get_value()
        == cookie_header
    )
    assert service.start_request.secrets["auth_token"].get_value() == "session-token"
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


@pytest.mark.asyncio
async def test_pyromind_conversation_converts_xyflow_before_seeding_workflow(
    tmp_path, monkeypatch
):
    knowledge_base = tmp_path / "knowledge"
    knowledge_base.mkdir()
    service = _FakeConversationService(tmp_path / "conversations")
    response = Response()
    xyflow = {"name": "Canvas", "nodes": [{"id": "n1"}], "edges": []}
    monkeypatch.setattr(
        "openhands.agent_server.pyromind_router.convert_xyflow_to_dsl",
        lambda workflow: "# workflow: Canvas\nnode = Example()\n",
    )

    info = await create_pyromind_conversation(
        _make_request(),
        PyromindCreateConversationRequest(
            llm=PyromindLLMConfig(model="gpt-4o", api_key="test-key"),
            workflow_xyflow=xyflow,
            extra={
                "knowledge_base_path": str(knowledge_base),
                "skills_path": str(tmp_path / "missing-skills"),
            },
        ),
        response,
        conversation_service=cast(ConversationService, service),
    )

    workflow_path = service.conversations_dir / info.id.hex / "workflow.py"
    assert workflow_path.read_text(encoding="utf-8") == (
        "# workflow: Canvas\nnode = Example()\n"
    )


@pytest.mark.asyncio
async def test_pyromind_conversation_initial_message_saves_workflow_snapshot(
    tmp_path, monkeypatch
):
    knowledge_base = tmp_path / "knowledge"
    knowledge_base.mkdir()
    service = _FakeConversationService(tmp_path / "conversations")
    response = Response()
    xyflow = {"name": "Canvas", "nodes": [{"id": "n1"}], "edges": []}
    monkeypatch.setattr(
        "openhands.agent_server.pyromind_router.convert_xyflow_to_dsl",
        lambda workflow: "# workflow: Canvas\nnode = Example()\n",
    )

    await create_pyromind_conversation(
        _make_request(),
        PyromindCreateConversationRequest(
            llm=PyromindLLMConfig(model="gpt-4o", api_key="test-key"),
            message="帮我继续改工作流",
            workflow_xyflow=xyflow,
            extra={
                "knowledge_base_path": str(knowledge_base),
                "skills_path": str(tmp_path / "missing-skills"),
            },
        ),
        response,
        conversation_service=cast(ConversationService, service),
    )

    assert service.start_request is not None
    assert service.start_request.initial_message is None
    assert service.event_service.sent_message is not None
    assert service.event_service.sent_message.role == "user"
    assert service.event_service.run is True
    assert service.event_service.workflow_dsl_snapshot == (
        "# workflow: Canvas\nnode = Example()\n"
    )
    assert service.event_service.workflow_xyflow_snapshot == xyflow


def test_workflow_dsl_from_xyflow_treats_empty_xyflow_as_empty_canvas():
    assert _workflow_dsl_from_xyflow({"name": "Empty", "nodes": [], "edges": []}) == ""


def test_pyromind_requests_reject_workflow_dsl_field():
    with pytest.raises(ValidationError):
        PyromindCreateConversationRequest.model_validate(
            {
                "llm": {"model": "gpt-4o", "api_key": "test-key"},
                "workflow_dsl": "# workflow: old\n",
            }
        )

    with pytest.raises(ValidationError):
        PyromindSendMessageRequest.model_validate(
            {
                "text": "继续",
                "workflow_dsl": "# workflow: old\n",
            }
        )


@pytest.mark.asyncio
async def test_pyromind_message_refreshes_storage_auth_context(tmp_path):
    service = _FakePyromindMessageEventService(
        {PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE},
        tmp_path,
    )
    http_request = _make_request(
        {
            "cookie": "auth_token=request-token; other=value",
            "x-cluster": "request-cluster",
        }
    )
    http_request.state.current_user = CurrentLoginUser(
        username="debug-user-42",
        email="debug-user-42@example.test",
        user_id=42,
        cookie="auth_token=context-token; other=value",
        x_cluster="context-cluster",
    )

    await send_pyromind_message(
        http_request,
        PyromindSendMessageRequest(text="帮我预览 /start-hook.sh"),
        event_service=cast(EventService, service),
    )

    assert service.secrets == {
        PYROMIND_VALIDATE_AUTH_COOKIE_SECRET: "auth_token=context-token; other=value",
        PYROMIND_STORAGE_AUTH_COOKIE_SECRET: "auth_token=context-token; other=value",
    }
    assert service.agent_state == {
        PYROMIND_VALIDATE_HEADERS_STATE_KEY: {"x-cluster": "context-cluster"},
        PYROMIND_STORAGE_HEADERS_STATE_KEY: {"x-cluster": "context-cluster"},
    }
    assert service.sent_message is not None
    assert service.run is True


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


def test_pyromind_storage_tools_use_user_context_headers():
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

    tools, secrets = _build_pyromind_storage_tools(request, {})

    assert [tool.name for tool in tools] == [
        PreviewDatasetTool.name,
        UploadFileToPyromindTool.name,
    ]
    assert tools[0].params == {
        "headers": {"x-cluster": "context-cluster"},
        "secret_headers": {"cookie": "PYROMIND_STORAGE_AUTH_COOKIE"},
    }
    assert tools[1].params == tools[0].params
    assert (
        secrets["PYROMIND_STORAGE_AUTH_COOKIE"].get_value()
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


@pytest.mark.asyncio
async def test_pyromind_validation_context_uses_websocket_user_headers():
    service = _FakeEventService({PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE})
    current_user = CurrentLoginUser(
        username="debug-user-42",
        email="debug-user-42@example.test",
        user_id=42,
        cookie="auth_token=websocket-token; other=value",
        x_cluster="websocket-cluster",
    )

    await apply_pyromind_validation_context(cast(EventService, service), current_user)

    assert service.secrets == {
        PYROMIND_VALIDATE_AUTH_COOKIE_SECRET: "auth_token=websocket-token; other=value",
        PYROMIND_AUTH_TOKEN_SECRET: "websocket-token",
        PYROMIND_STORAGE_AUTH_COOKIE_SECRET: "auth_token=websocket-token; other=value",
    }
    assert service.agent_state == {
        PYROMIND_VALIDATE_HEADERS_STATE_KEY: {"x-cluster": "websocket-cluster"},
        PYROMIND_STORAGE_HEADERS_STATE_KEY: {"x-cluster": "websocket-cluster"},
    }











def test_build_workflow_run_tool_wires_env_headers_and_auth_token():
    request = _make_request(
        {
            "cookie": f"{PYROMIND_AUTH_COOKIE_NAME}=jwt-token",
            "x-cluster": "us-west-1#pre",
            "accept-language": "en-US",
        }
    )
    load_base_env(request)

    tool, secrets = _build_workflow_run_tool(request, {})

    assert tool.name == RunWorkflowTool.name
    assert tool.params == {
        "current_user": None,
        "env": "pre",
        "cluster": "us-west-1",
        "headers": {
            "x-cluster": "us-west-1#pre",
            "accept-language": "en-US",
            "request-app": "openhands",
        },
    }
    assert secrets["auth_token"].get_value() == "jwt-token"


def test_parse_auth_token_from_cookie_header_extracts_jwt():
    from openhands.agent_server.pyromind_auth import parse_auth_token_from_cookie_header

    assert (
        parse_auth_token_from_cookie_header("auth_token=jwt-token; other=value")
        == "jwt-token"
    )
    assert parse_auth_token_from_cookie_header(None) is None


@pytest.mark.asyncio
async def test_pyromind_validation_context_ignores_non_pyromind_conversations():
    service = _FakeEventService({})
    current_user = CurrentLoginUser(
        username="debug-user-42",
        email="debug-user-42@example.test",
        user_id=42,
        cookie="auth_token=websocket-token",
        x_cluster="websocket-cluster",
    )

    await apply_pyromind_validation_context(cast(EventService, service), current_user)

    assert service.secrets == {}
    assert service.agent_state == {}
