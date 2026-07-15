from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.run_workflow_callback import (
    build_run_workflow_terminal_reminder,
    deliver_run_workflow_status,
)
from openhands.sdk.llm import TextContent


class _FakeEventService:
    def __init__(self) -> None:
        self.run: bool | None = None
        self.internal_context: list[TextContent] | None = None

    async def send_internal_context(
        self,
        content: list[TextContent],
        run: bool = False,
    ) -> str:
        self.run = run
        self.internal_context = content
        return "internal-event"


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
    assert service.event_service.internal_context is not None
    reminder = service.event_service.internal_context[0].text
    assert task_id in reminder
    assert "Resume the tool invocation associated with this task" in reminder
    assert "most recent non-empty visible message" in reminder
    assert "workflow_debug" not in reminder
    assert "Review the outcome and continue helping the user" not in reminder
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
async def test_workflow_debug_callback_success_uses_debug_guidance(tmp_path):
    conversation_id = uuid4()
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Succeeded",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert service.event_service.internal_context is not None
    reminder = service.event_service.internal_context[0].text
    assert "workflow_debug (test) run that passed" in reminder
    assert "wait for their next message" in reminder
    assert "Resume the tool invocation associated with this task" not in reminder


@pytest.mark.asyncio
async def test_workflow_debug_callback_failure_uses_debug_guidance(tmp_path):
    conversation_id = uuid4()
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Failed",
        error_log="node X failed",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert service.event_service.internal_context is not None
    reminder = service.event_service.internal_context[0].text
    assert "workflow_debug (test) run that failed" in reminder
    assert "call workflow_debug again" in reminder
    assert "node X failed" in reminder
    assert "Resume the tool invocation associated with this task" not in reminder


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
    assert service.event_service.internal_context is None


def test_build_reminder_production_ignores_debug_status_semantics():
    reminder = build_run_workflow_terminal_reminder(
        task_id="t1",
        status="Succeeded",
        from_workflow_debug=False,
    )
    assert "Resume the tool invocation associated with this task" in reminder
    assert "workflow_debug" not in reminder


def test_build_reminder_debug_terminated_guidance():
    reminder = build_run_workflow_terminal_reminder(
        task_id="t1",
        status="Terminated",
        error_log="cancelled by user",
        from_workflow_debug=True,
    )
    assert "was terminated" in reminder
    assert "cancelled by user" in reminder
    assert "Resume the tool invocation associated with this task" not in reminder


def test_build_reminder_debug_error_matches_failure_guidance():
    reminder = build_run_workflow_terminal_reminder(
        task_id="t1",
        status="Error",
        error_log="boom",
        from_workflow_debug=True,
    )
    assert "workflow_debug (test) run that failed" in reminder
    assert "call workflow_debug again" in reminder
    assert "boom" in reminder


@pytest.mark.asyncio
async def test_workflow_debug_callback_terminated_uses_debug_guidance(tmp_path):
    conversation_id = uuid4()
    task_id = f"workflow-task-{uuid4()}"
    service = _FakeConversationService(tmp_path / "conversations")

    result = await deliver_run_workflow_status(
        task_id=task_id,
        status="Terminated",
        conversation_id=str(conversation_id),
        auto_run=False,
        from_workflow_debug=True,
        conversation_service=cast(ConversationService, service),
    )

    assert result.outcome == "delivered_async"
    assert service.event_service.internal_context is not None
    reminder = service.event_service.internal_context[0].text
    assert "was terminated" in reminder
    assert "Resume the tool invocation associated with this task" not in reminder
