from typing import Any

import pytest

from openhands.agent_server.kafka_bus.handlers import (
    studio_workflow_notify_handler as handler_module,
)
from openhands.agent_server.kafka_bus.handlers.studio_workflow_notify_handler import (
    StudioWorkflowNotifyHandler,
    TrainingTaskEventType,
)
from openhands.agent_server.kafka_bus.message_event import MessageEvent
from openhands.agent_server.run_workflow_callback import RunWorkflowCallbackResult


@pytest.mark.asyncio
async def test_terminal_notification_uses_generic_workflow_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_deliver(**kwargs: Any) -> RunWorkflowCallbackResult:
        captured.update(kwargs)
        return RunWorkflowCallbackResult(
            outcome="delivered_async",
            task_id="clean-task-1",
            normalized_status="Succeeded",
            conversation_id="550e8400-e29b-41d4-a716-446655440000",
        )

    monkeypatch.setattr(
        handler_module,
        "deliver_run_workflow_status",
        fake_deliver,
    )

    await StudioWorkflowNotifyHandler().handle(
        MessageEvent(
            event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED,
            message_id="message-1",
            data={
                "task_id": "clean-task-1",
                "status": "Succeeded",
                "error_msg": None,
                "out_id": "agent1#550e8400-e29b-41d4-a716-446655440000",
            },
        )
    )

    assert captured == {
        "task_id": "clean-task-1",
        "status": "Succeeded",
        "error_log": None,
        "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
        "auto_run": True,
    }


@pytest.mark.asyncio
async def test_non_terminal_notification_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def fake_deliver(**kwargs: Any) -> RunWorkflowCallbackResult:
        nonlocal called
        called = True
        raise AssertionError("non-terminal status must not be dispatched")

    monkeypatch.setattr(
        handler_module,
        "deliver_run_workflow_status",
        fake_deliver,
    )

    await StudioWorkflowNotifyHandler().handle(
        MessageEvent(
            event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED,
            data={"task_id": "task-1", "status": "Running"},
        )
    )

    assert called is False
