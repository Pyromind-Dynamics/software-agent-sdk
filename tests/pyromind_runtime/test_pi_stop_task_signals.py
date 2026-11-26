"""Stop-request acceptance does not synthesize a task termination."""

from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from pyromind_common.tools.data_preparation.stop_task import DfStopTaskObservation
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import CancelCommand
from pyromind_runtime.domain.snapshot import ConversationSnapshot, ExternalTaskState


@pytest.mark.parametrize(
    ("platform_status", "terminal_confirmed", "expected_status"),
    [
        ("Running", False, None),
        ("Stopping", False, None),
        ("Stopped", False, None),
        ("Stopped", True, "stopped"),
        ("Terminated", True, "terminated"),
        ("Cancelled", True, "stopped"),
        ("Canceled", True, "stopped"),
        ("Succeeded", True, None),
    ],
)
@pytest.mark.parametrize("entry", ["tool", "cancel"])
async def test_stop_paths_only_emit_confirmed_stop_status(
    pi_stop_session,
    monkeypatch,
    platform_status,
    terminal_confirmed,
    expected_status,
    entry,
) -> None:
    adapter, session = pi_stop_session
    session.active_external_tasks["task-1"] = "data_preparation"
    observation = DfStopTaskObservation.from_text(
        text="Stop request accepted",
        task_id="task-1",
        stopped=True,
        status="Stopped",
        platform_status=platform_status,
        terminal_confirmed=terminal_confirmed,
    )

    receipt = {
        "is_error": observation.is_error,
        "details": observation.model_dump(mode="json"),
    }
    if entry == "tool":
        adapter._emit_tool_signals(session, "df_stop_task", {**receipt, "signals": []})
    else:
        adapter._sessions[session.session_id] = session
        native_cancel = AsyncMock()
        monkeypatch.setattr(
            session, "runner", SimpleNamespace(running=True, request=native_cancel)
        )
        stop = AsyncMock(return_value=receipt)
        monkeypatch.setattr(adapter._business_tools, "stop_platform_task", stop)
        result = await adapter.send(
            adapter._handle(session),
            CancelCommand(command_id="cancel"),
            session.context,
        )
        assert result["cancelled"] is True
        native_cancel.assert_awaited_once_with("cancel", {})
        assert stop.await_args is not None
        assert stop.await_args.args[0] == "task-1"

    if expected_status is not None:
        event = session.queue.get_nowait()
        assert event is not None
        assert event.type == "external_task.completed"
        assert event.payload["task_id"] == "task-1"
        assert event.payload["status"] == expected_status
        assert "task-1" not in session.active_external_tasks
    else:
        assert session.queue.empty()
        assert session.active_external_tasks == {"task-1": "data_preparation"}


async def test_cancel_restores_pending_business_tasks_from_snapshot(
    pi_stop_session, monkeypatch
) -> None:
    adapter, session = pi_stop_session
    states: tuple[tuple[str, Literal["pending", "succeeded"]], ...] = (
        ("plan-task", "pending"),
        ("done-task", "succeeded"),
    )
    snapshot = ConversationSnapshot(
        conversation_id=session.session_id,
        capabilities=HarnessCapabilities(),
        external_tasks=tuple(
            ExternalTaskState(
                task_id=task_id,
                kind="data_preparation",
                status=status,
                submitted_at="2026-11-23T00:00:00+00:00",
                updated_at="2026-11-23T00:00:00+00:00",
            )
            for task_id, status in states
        ),
    )
    product = session.workspace_root / "product"
    product.mkdir()
    (product / "snapshot.json").write_text(snapshot.model_dump_json())
    adapter._sessions[session.session_id] = session
    stop = AsyncMock(
        return_value={
            "is_error": False,
            "details": DfStopTaskObservation.from_text(
                text="Stop request accepted",
                task_id="plan-task",
                stopped=True,
                status="Stopped",
                platform_status="Running",
                terminal_confirmed=False,
            ).model_dump(mode="json"),
        }
    )
    monkeypatch.setattr(adapter._business_tools, "stop_platform_task", stop)
    assert session.active_external_tasks == {}

    result = await adapter.send(
        adapter._handle(session), CancelCommand(command_id="cancel"), session.context
    )

    assert result["cancelled"] is True
    stop.assert_awaited_once()
    assert stop.await_args is not None
    assert stop.await_args.args[0] == "plan-task"
    assert session.active_external_tasks == {"plan-task": "data_preparation"}
    assert session.queue.empty()
