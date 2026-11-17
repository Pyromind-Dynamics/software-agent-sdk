"""验证 Pi 复用的平台调试与工作流状态回调。"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from pyromind_agent_server.pyromind_router import (
    PyromindDebugCallbackRequest,
    PyromindWorkflowCallbackRequest,
    pyromind_debug_callback,
    pyromind_workflow_callback,
)
from pyromind_agent_server.workflow_status import WorkflowCallbackResult
from pyromind_common.tools.pyromind_debug.broker import get_debug_result_broker


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

    async def fake_deliver(**kwargs: object) -> WorkflowCallbackResult:
        captured.update(kwargs)
        return WorkflowCallbackResult(
            outcome="delivered_async",
            task_id="task-1",
            normalized_status="Succeeded",
            conversation_id="conv-1",
        )

    monkeypatch.setattr(
        "pyromind_agent_server.pyromind_router.deliver_workflow_status",
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
        "from_workflow_debug": False,
    }


@pytest.mark.asyncio
async def test_workflow_callback_passes_from_workflow_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_deliver(**kwargs: object) -> WorkflowCallbackResult:
        captured.update(kwargs)
        return WorkflowCallbackResult(
            outcome="delivered_async",
            task_id="task-debug-1",
            normalized_status="Succeeded",
            conversation_id="conv-1",
        )

    monkeypatch.setattr(
        "pyromind_agent_server.pyromind_router.deliver_workflow_status",
        fake_deliver,
    )

    response = await pyromind_workflow_callback(
        PyromindWorkflowCallbackRequest(
            task_id="task-debug-1",
            status="Succeeded",
            conversation_id="550e8400-e29b-41d4-a716-446655440000",
            auto_run=True,
            from_workflow_debug=True,
        )
    )

    assert response.success is True
    assert captured["from_workflow_debug"] is True
    assert captured["auto_run"] is True


@pytest.mark.asyncio
async def test_workflow_callback_allows_omitted_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_deliver(**kwargs: object) -> WorkflowCallbackResult:
        captured.update(kwargs)
        return WorkflowCallbackResult(
            outcome="delivered_async",
            task_id="clean-task-1",
            normalized_status="Succeeded",
            conversation_id="550e8400-e29b-41d4-a716-446655440000",
        )

    monkeypatch.setattr(
        "pyromind_agent_server.pyromind_router.deliver_workflow_status",
        fake_deliver,
    )

    response = await pyromind_workflow_callback(
        PyromindWorkflowCallbackRequest(
            task_id="clean-task-1",
            status="Succeeded",
        )
    )

    assert response.success is True
    assert captured["conversation_id"] is None


@pytest.mark.asyncio
async def test_workflow_callback_unknown_conversation_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_deliver(**kwargs: object) -> WorkflowCallbackResult:
        return WorkflowCallbackResult(
            outcome="unknown_conversation",
            task_id="task-1",
            normalized_status="Succeeded",
            conversation_id="missing",
        )

    monkeypatch.setattr(
        "pyromind_agent_server.pyromind_router.deliver_workflow_status",
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
