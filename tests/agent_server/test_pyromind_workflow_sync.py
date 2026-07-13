"""Tests for the workflow<->canvas sync helper and the debug callback route.

Covers the "工作流同步链路" cases from the debug-loop plan: from-scratch
(no-op), canvas edited (overwrite + reminder), canvas cleared (remove file +
reminder), already-in-sync (no-op), and no canvas attached (no-op).
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from openhands.agent_server.pyromind_router import (
    PyromindDebugCallbackRequest,
    PyromindWorkflowCallbackRequest,
    _sync_workflow_with_canvas,
    pyromind_debug_callback,
    pyromind_workflow_callback,
)
from openhands.agent_server.run_workflow_callback import RunWorkflowCallbackResult
from openhands.tools.pyromind_debug.broker import get_debug_result_broker


def test_no_workflow_dsl_is_a_noop(tmp_path):
    assert _sync_workflow_with_canvas(tmp_path, None) is None
    assert not (tmp_path / "workflow.py").exists()


def test_from_scratch_both_empty_is_a_noop(tmp_path):
    assert _sync_workflow_with_canvas(tmp_path, "") is None
    assert not (tmp_path / "workflow.py").exists()


def test_already_in_sync_is_a_noop(tmp_path):
    (tmp_path / "workflow.py").write_text("# workflow: demo\nx = 1\n", encoding="utf-8")

    reminder = _sync_workflow_with_canvas(tmp_path, "# workflow: demo\nx = 1\n")

    assert reminder is None
    assert (tmp_path / "workflow.py").read_text(encoding="utf-8") == (
        "# workflow: demo\nx = 1\n"
    )


def test_canvas_edited_overwrites_and_reminds(tmp_path):
    (tmp_path / "workflow.py").write_text("# workflow: old\nx = 1\n", encoding="utf-8")

    reminder = _sync_workflow_with_canvas(tmp_path, "# workflow: new\nx = 2\n")

    assert reminder is not None
    assert "system_reminder" in reminder.text
    assert "modified the workflow on the canvas" in reminder.text
    assert (tmp_path / "workflow.py").read_text(encoding="utf-8") == (
        "# workflow: new\nx = 2\n"
    )


def test_canvas_seeds_missing_workflow_file(tmp_path):
    reminder = _sync_workflow_with_canvas(tmp_path, "# workflow: from-canvas\n")

    assert reminder is not None
    assert "already had a workflow on the canvas" in reminder.text
    assert (tmp_path / "workflow.py").read_text(encoding="utf-8") == (
        "# workflow: from-canvas\n"
    )


def test_canvas_cleared_removes_file_and_reminds(tmp_path):
    (tmp_path / "workflow.py").write_text("# workflow: old\nx = 1\n", encoding="utf-8")

    reminder = _sync_workflow_with_canvas(tmp_path, "")

    assert reminder is not None
    assert "cleared the workflow on the canvas" in reminder.text
    assert not (tmp_path / "workflow.py").exists()


def test_whitespace_only_diff_is_a_noop(tmp_path):
    (tmp_path / "workflow.py").write_text("# workflow: demo\nx = 1\n", encoding="utf-8")

    reminder = _sync_workflow_with_canvas(tmp_path, "# workflow: demo\nx = 1\n\n\n  ")

    assert reminder is None


@pytest.mark.asyncio
async def test_debug_callback_resolves_broker():
    broker = get_debug_result_broker()
    broker.register("task-abc")

    result_holder: dict[str, object] = {}

    def _wait():
        result_holder["result"] = broker.wait("task-abc", timeout=5)

    import threading

    waiter = threading.Thread(target=_wait)
    waiter.start()

    success = await pyromind_debug_callback(
        PyromindDebugCallbackRequest(
            task_id="task-abc", status="failed", error_log="boom"
        )
    )

    waiter.join(timeout=5)
    assert success.success is True
    assert result_holder["result"] is not None
    assert result_holder["result"].status == "failed"  # type: ignore[union-attr]
    assert result_holder["result"].error_log == "boom"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_debug_callback_unknown_task_returns_404():
    with pytest.raises(HTTPException) as exc_info:
        await pyromind_debug_callback(
            PyromindDebugCallbackRequest(task_id="does-not-exist", status="passed")
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_workflow_callback_delegates_to_deliver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_deliver(**kwargs: object) -> RunWorkflowCallbackResult:
        captured.update(kwargs)
        return RunWorkflowCallbackResult(
            outcome="delivered_async",
            task_id="task-1",
            normalized_status="Succeeded",
            conversation_id="conv-1",
        )

    monkeypatch.setattr(
        "openhands.agent_server.pyromind_router.deliver_run_workflow_status",
        fake_deliver,
    )

    response = await pyromind_workflow_callback(
        PyromindWorkflowCallbackRequest(
            task_id="task-1",
            status="Succeeded",
            conversation_id="550e8400-e29b-41d4-a716-446655440000",
            auto_run=False,
        )
    )

    assert response.success is True
    assert response.outcome == "delivered_async"
    assert captured == {
        "task_id": "task-1",
        "status": "Succeeded",
        "error_log": None,
        "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
        "auto_run": False,
    }


@pytest.mark.asyncio
async def test_workflow_callback_unknown_conversation_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_deliver(**kwargs: object) -> RunWorkflowCallbackResult:
        return RunWorkflowCallbackResult(
            outcome="unknown_conversation",
            task_id="task-1",
            normalized_status="Succeeded",
            conversation_id="missing",
        )

    monkeypatch.setattr(
        "openhands.agent_server.pyromind_router.deliver_run_workflow_status",
        fake_deliver,
    )

    with pytest.raises(HTTPException) as exc_info:
        await pyromind_workflow_callback(
            PyromindWorkflowCallbackRequest(
                task_id="task-1",
                status="Succeeded",
                conversation_id="missing",
            )
        )

    assert exc_info.value.status_code == 404
