from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.run_workflow_callback import deliver_run_workflow_status
from openhands.sdk.llm import Message, TextContent


class _FakeEventService:
    def __init__(self) -> None:
        self.message: Message | None = None
        self.run: bool | None = None
        self.extended_content: list[TextContent] | None = None

    async def send_message(
        self,
        message: Message,
        run: bool = False,
        extended_content: list[TextContent] | None = None,
    ) -> None:
        self.message = message
        self.run = run
        self.extended_content = extended_content


class _FakeConversationService:
    def __init__(self, conversations_dir: Path) -> None:
        self.conversations_dir = conversations_dir
        self.event_service = _FakeEventService()
        self.requested_conversation_id: UUID | None = None

    async def get_event_service(self, conversation_id: UUID):
        self.requested_conversation_id = conversation_id
        return self.event_service


@pytest.mark.asyncio
async def test_generic_callback_silently_resumes_conversation(tmp_path):
    conversation_id = uuid4()
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_id=str(conversation_id),
        auto_run=False,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert result.normalized_status == "Succeeded"
    assert result.conversation_id == str(conversation_id)
    assert service.requested_conversation_id == conversation_id
    assert service.event_service.run is False
    assert service.event_service.message is not None
    assert service.event_service.message.content == []
    assert service.event_service.extended_content is not None
    reminder = service.event_service.extended_content[0].text
    assert task_id in reminder
    assert "tool invocation associated with this task" in reminder
    assert "most recent non-empty visible message" in reminder
    assert "stats.json" not in reminder

    duplicate = await deliver_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_id=str(conversation_id),
        auto_run=False,
        conversation_service=cast(ConversationService, service),
    )
    assert duplicate.outcome == "duplicate_terminal"


@pytest.mark.asyncio
async def test_generic_callback_without_conversation_is_unknown_task(tmp_path):
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "unknown_task"
    assert service.requested_conversation_id is None
    assert service.event_service.message is None
