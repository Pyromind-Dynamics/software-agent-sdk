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
        "from_workflow_debug": False,
    }


@pytest.mark.asyncio
async def test_debug_out_id_passes_from_workflow_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_deliver(**kwargs: Any) -> RunWorkflowCallbackResult:
        captured.update(kwargs)
        return RunWorkflowCallbackResult(
            outcome="delivered_async",
            task_id="debug-task-1",
            normalized_status="Failed",
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
            message_id="message-debug-1",
            data={
                "task_id": "debug-task-1",
                "status": "Failed",
                "error_msg": "node boom",
                "out_id": "agent1#debug#550e8400-e29b-41d4-a716-446655440000",
            },
        )
    )

    assert captured == {
        "task_id": "debug-task-1",
        "status": "Failed",
        "error_log": "node boom",
        "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
        "auto_run": True,
        "from_workflow_debug": True,
    }


@pytest.mark.asyncio
async def test_missing_or_foreign_out_id_is_discarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def fake_deliver(**kwargs: Any) -> RunWorkflowCallbackResult:
        nonlocal called
        called = True
        raise AssertionError("invalid out_id must not be delivered")

    monkeypatch.setattr(
        handler_module,
        "deliver_run_workflow_status",
        fake_deliver,
    )

    handler = StudioWorkflowNotifyHandler()
    for out_id in (None, "", "other#cid", "bare-uuid"):
        await handler.handle(
            MessageEvent(
                event_type=TrainingTaskEventType.TRAINING_TASK_STATUS_CHANGED,
                data={
                    "task_id": "task-1",
                    "status": "Succeeded",
                    "out_id": out_id,
                },
            )
        )

    assert called is False


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
