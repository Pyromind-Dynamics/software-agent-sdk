from pathlib import Path

import pytest
from fastapi import Response, status

from openhands.agent_server.models import ConversationInfo
from openhands.agent_server.pyromind_constants import (
    PYROMIND_APP_TAG_KEY,
    PYROMIND_APP_TAG_VALUE,
)
from openhands.agent_server.pyromind_router import (
    PyromindCreateConversationRequest,
    PyromindLLMConfig,
    create_pyromind_conversation,
)
from openhands.sdk.conversation.request import StartConversationRequest
from openhands.sdk.conversation.state import ConversationExecutionStatus


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


@pytest.mark.asyncio
async def test_pyromind_conversation_uses_conversation_workspace(tmp_path):
    knowledge_base = tmp_path / "knowledge"
    knowledge_base.mkdir()
    service = _FakeConversationService(tmp_path / "conversations")
    response = Response()

    info = await create_pyromind_conversation(
        PyromindCreateConversationRequest(
            llm=PyromindLLMConfig(model="gpt-4o", api_key="test-key"),
            extra={
                "knowledge_base_path": str(knowledge_base),
                "skills_path": str(tmp_path / "missing-skills"),
            },
        ),
        response,
        conversation_service=service,
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
    assert _REMOVED_WORKFLOW_TOOL not in tool_names
    assert service.start_request.tags == {
        PYROMIND_APP_TAG_KEY: PYROMIND_APP_TAG_VALUE
    }
