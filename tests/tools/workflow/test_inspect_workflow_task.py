"""Tests for the ``inspect_workflow_task`` tool."""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.tools.workflow import (
    InspectWorkflowTaskAction,
    InspectWorkflowTaskExecutor,
    InspectWorkflowTaskTool,
)
from openhands.tools.workflow.inspect_workflow_task import (
    PROD_API_BASE,
    workflow_execution_state,
)


class _Response:
    def __init__(self, status_code: int, payload: Any, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        return self._payload


def _task_payload() -> dict[str, Any]:
    statuses = [
        ("1", "Load Dataset", "Succeeded"),
        ("2", "Train", "Running"),
        ("3", "Evaluate", "Pending"),
        ("4", "Publish", "Cancelled"),
        ("5", "Cleanup", "Skipped"),
        ("6", "Unknown Node", "platform-specific-state"),
        ("7", "Timed Out", "Timeout"),
    ]
    return {
        "task_status": "Running",
        "workflow": {
            "nodes": [
                {
                    "id": node_id,
                    "data": {"nodeType": f"Node{node_id}", "display_name": name},
                    "properties": {"dystatus": status},
                }
                for node_id, name, status in statuses
            ]
        },
    }


def _install_router(monkeypatch) -> list[tuple[str, dict[str, str]]]:
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_get(url, *, params, headers, timeout):
        calls.append((url, params))
        if "task_workflow_result" in url:
            return _Response(200, _task_payload())
        if "logs/node/raw" in url:
            return _Response(200, {"entries": [{"m": "line 1\n"}, {"m": "line 2"}]})
        return _Response(404, {}, "not found")

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def test_lists_all_node_states_without_fetching_logs(monkeypatch):
    calls = _install_router(monkeypatch)

    observation = InspectWorkflowTaskExecutor()(
        InspectWorkflowTaskAction(task_id="10077")
    )

    assert not observation.is_error
    assert observation.task_state == "running"
    assert [node.state for node in observation.nodes] == [
        "succeeded",
        "running",
        "pending",
        "cancelled",
        "skipped",
        "unknown",
        "failed",
    ]
    assert [node.node_id for node in observation.failed_nodes] == ["7"]
    assert observation.selected_node is None
    assert observation.log is None
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("route", "base"),
    [
        (None, PROD_API_BASE),
        ("us-west-2#pre", "https://pre-api-portal.pyromind.ai/std2/studio_api/"),
        ("us-west-2#pre2", "https://pre2-api-portal.pyromind.ai/std2/studio_api/"),
    ],
)
def test_api_base_uses_request_route(monkeypatch, route, base):
    monkeypatch.delenv("PYROMIND_API_BASE", raising=False)
    monkeypatch.setenv("APP_ENV", "pre")
    calls = _install_router(monkeypatch)

    observation = InspectWorkflowTaskExecutor(
        headers={"x-cluster": route} if route else {}
    )(InspectWorkflowTaskAction(task_id="10077"))

    assert not observation.is_error
    assert calls[0][0] == f"{base}api/task_workflow_result"


def test_api_base_preserves_custom_endpoint(monkeypatch):
    calls = _install_router(monkeypatch)

    observation = InspectWorkflowTaskExecutor(
        api_base="https://custom.test/studio/",
        headers={"x-cluster": "us-west-2#pre"},
    )(InspectWorkflowTaskAction(task_id="10077"))

    assert not observation.is_error
    assert calls[0][0] == "https://custom.test/studio/api/task_workflow_result"


def test_fetches_log_for_successful_node(monkeypatch):
    calls = _install_router(monkeypatch)

    observation = InspectWorkflowTaskExecutor()(
        InspectWorkflowTaskAction(task_id="10077", node_id="1", include_log=True)
    )

    assert not observation.is_error
    assert observation.selected_node is not None
    assert observation.selected_node.state == "succeeded"
    assert observation.log == "line 1\nline 2"
    assert observation.node_log_available
    assert [node.node_id for node in observation.failed_nodes] == ["7"]
    assert calls[1][1] == {"nodeId": "1", "taskId": "10077"}


def test_selects_node_without_fetching_log(monkeypatch):
    calls = _install_router(monkeypatch)

    observation = InspectWorkflowTaskExecutor()(
        InspectWorkflowTaskAction(task_id="10077", node_id="2")
    )

    assert not observation.is_error
    assert observation.selected_node is not None
    assert observation.selected_node.state == "running"
    assert observation.log is None
    assert len(calls) == 1


def test_unknown_node_returns_available_nodes(monkeypatch):
    calls = _install_router(monkeypatch)

    observation = InspectWorkflowTaskExecutor()(
        InspectWorkflowTaskAction(task_id="10077", node_id="missing")
    )

    assert observation.is_error
    assert observation.error_message is not None
    assert "Available nodes: 1, 2, 3, 4, 5, 6, 7" in observation.error_message
    assert len(calls) == 1


def test_include_log_requires_node_id():
    with pytest.raises(ValidationError, match="node_id is required"):
        InspectWorkflowTaskAction(task_id="10077", include_log=True)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (None, "unknown"),
        ("not started", "pending"),
        ("completed", "succeeded"),
        ("aborted by user", "cancelled"),
        ("operator-specific", "unknown"),
    ],
)
def test_workflow_execution_state(status, expected):
    assert workflow_execution_state(status) == expected


def test_tool_is_registered_and_accepts_auth_params():
    resolved = resolve_tool(Tool(name=InspectWorkflowTaskTool.name), cast(Any, None))
    assert isinstance(resolved[0], InspectWorkflowTaskTool)

    instances = InspectWorkflowTaskTool.create(headers={"x-cluster": "us-west-1"})
    assert isinstance(instances[0].executor, InspectWorkflowTaskExecutor)
