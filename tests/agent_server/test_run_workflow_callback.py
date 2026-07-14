from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.dataset_cleaning_callback import (
    dispatch_run_workflow_status,
)
from openhands.agent_server.run_workflow_callback import deliver_run_workflow_status
from openhands.sdk.llm import Message, TextContent
from openhands.tools.pyromind_cleaning.task_store import (
    DatasetCleaningTaskAssociation,
    task_store_for_conversations_dir,
)


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
async def test_generic_callback_keeps_default_terminal_delivery(tmp_path):
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
    assert service.event_service.message is not None
    message_content = service.event_service.message.content[0]
    assert isinstance(message_content, TextContent)
    assert "workflow run finished" in message_content.text
    assert service.event_service.extended_content is not None
    assert "stats.json" not in service.event_service.extended_content[0].text


@pytest.mark.asyncio
async def test_cleaning_callback_resolves_persisted_task_association(tmp_path):
    conversation_id = uuid4()
    task_id = f"clean-task-{uuid4()}"
    run_id = str(uuid4())
    output_dir = f"/agentTest/data_cleaning/{run_id}"
    service = _FakeConversationService(tmp_path / "conversations")
    store = task_store_for_conversations_dir(service.conversations_dir)
    store.save(
        DatasetCleaningTaskAssociation(
            task_id=task_id,
            conversation_id=str(conversation_id),
            run_id=run_id,
            output_dir=output_dir,
            input_path="/datasets/source.jsonl",
            script_path="/agentTest/clean.py",
        )
    )

    result = await dispatch_run_workflow_status(
        task_id=task_id,
        status="succeeded",
        conversation_id=None,
        auto_run=False,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert result.normalized_status == "Succeeded"
    assert result.conversation_id == str(conversation_id)
    assert service.requested_conversation_id == conversation_id
    assert service.event_service.run is False
    assert service.event_service.message is not None
    message_content = service.event_service.message.content[0]
    assert isinstance(message_content, TextContent)
    assert task_id in message_content.text
    assert service.event_service.extended_content is not None
    reminder = service.event_service.extended_content[0].text
    assert run_id in reminder
    assert output_dir in reminder
    assert "stats.json" in reminder
    persisted = store.get(task_id)
    assert persisted is not None
    assert persisted.status == "Succeeded"

    duplicate = await dispatch_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_service=cast(ConversationService, service),
    )
    assert duplicate.outcome == "duplicate_terminal"


@pytest.mark.asyncio
async def test_unknown_callback_can_retry_after_task_association_is_saved(tmp_path):
    conversation_id = uuid4()
    task_id = f"clean-task-{uuid4()}"
    run_id = str(uuid4())
    service = _FakeConversationService(tmp_path / "conversations")

    first_result = await dispatch_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_service=cast(ConversationService, service),
    )
    assert first_result.outcome == "unknown_task"

    task_store_for_conversations_dir(service.conversations_dir).save(
        DatasetCleaningTaskAssociation(
            task_id=task_id,
            conversation_id=str(conversation_id),
            run_id=run_id,
            output_dir=f"/agentTest/data_cleaning/{run_id}",
            input_path="/datasets/source.jsonl",
            script_path="/agentTest/clean.py",
        )
    )
    retry_result = await dispatch_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_service=cast(ConversationService, service),
    )

    assert retry_result.outcome == "delivered_async"
    assert retry_result.conversation_id == str(conversation_id)
