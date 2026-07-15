"""Tests for the workflow_debug tool wrapper."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

from pyromind_sdk.client.models import TrainingTaskCreateResponse

from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.tools.workflow import RunWorkflowObservation
from openhands.tools.workflow.run_workflow import RunWorkflowExecutor
from openhands.tools.workflow_debug import (
    WorkflowDebugAction,
    WorkflowDebugExecutor,
    WorkflowDebugObservation,
    WorkflowDebugTool,
)


_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000456")


def _executor_kwargs(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "env": "pre",
        "cluster": "us-west-1",
        "current_user": object(),
        "headers": {"x-cluster": "us-west-1#pre", "request-app": "openhands"},
    }
    defaults.update(overrides)
    return defaults


def _fake_conversation(
    tmp_path: Path,
    *,
    secret_registry: SecretRegistry | None = None,
) -> LocalConversation:
    return cast(
        LocalConversation,
        SimpleNamespace(
            id=_CONVERSATION_ID,
            workspace=SimpleNamespace(working_dir=str(tmp_path)),
            state=SimpleNamespace(
                secret_registry=secret_registry or SecretRegistry(),
            ),
        ),
    )


def test_workflow_debug_tool_create_and_resolve() -> None:
    params = _executor_kwargs()
    tool = WorkflowDebugTool.create(**params)[0]

    assert tool.name == "workflow_debug"
    assert tool.annotations is not None
    assert tool.annotations.title == "workflow_debug"
    assert {"note", "dsl", "name"} <= set(tool.action_type.model_fields)
    assert "test_mode" not in tool.action_type.model_fields
    assert "keep_ui_lock" in tool.observation_type.model_fields  # type: ignore[union-attr]
    assert "keep_ui_lock" not in RunWorkflowObservation.model_fields

    resolved = resolve_tool(
        Tool(name="workflow_debug", params=params),
        cast(Any, None),
    )
    assert isinstance(resolved[0], WorkflowDebugTool)


def test_workflow_debug_submits_with_test_mode(
    tmp_path: Path, monkeypatch: Any
) -> None:
    registry = SecretRegistry()
    registry.update_secrets({"auth_token": "jwt-token"})

    mock_client = MagicMock()
    mock_client.studio.create.return_value = TrainingTaskCreateResponse(
        task_id="task-debug-1",
        name="demo",
        status="Pending",
    )

    monkeypatch.setattr(
        "openhands.tools.workflow.task_submission.get_api_key",
        lambda **kwargs: "access-key-1",
    )
    monkeypatch.setattr(
        "openhands.tools.workflow.task_submission.get_pyromind_api_client",
        lambda **kwargs: mock_client,
    )
    monkeypatch.setattr(
        RunWorkflowExecutor,
        "_convert_dsl_to_xyflow",
        lambda self, *, dsl, name: SimpleNamespace(
            is_error=False,
            xyflow={"name": name, "nodes": [], "edges": []},
            text="",
        ),
    )

    observation = WorkflowDebugExecutor(**_executor_kwargs())(
        WorkflowDebugAction(dsl="# workflow: demo", name="demo"),
        conversation=_fake_conversation(tmp_path, secret_registry=registry),
    )

    assert isinstance(observation, WorkflowDebugObservation)
    assert not observation.is_error
    assert observation.status == "Pending"
    assert observation.task_id == "task-debug-1"
    assert observation.keep_ui_lock is True
    mock_client.studio.create.assert_called_once()
    request = mock_client.studio.create.call_args.args[0]
    assert request.workflow["execution_argos"] == [{"execution_mode": "test"}]
    assert request.out_id == f"agent1#debug#{_CONVERSATION_ID}"


def test_workflow_debug_forces_keep_ui_lock_on_success() -> None:
    """Async submit must return keep_ui_lock=True for the frontend lock."""
    run_executor = MagicMock()
    run_executor.return_value = RunWorkflowObservation.from_text(
        text="submitted",
        status="Pending",
        task_id="task-1",
        attempt=1,
        max_attempts=10,
        is_error=False,
    )
    observation = WorkflowDebugExecutor(run_executor=run_executor)(
        WorkflowDebugAction(dsl="# workflow: demo"),
        conversation=None,
    )
    assert isinstance(observation, WorkflowDebugObservation)
    assert observation.keep_ui_lock is True
    assert observation.task_id == "task-1"
    assert run_executor.call_args.kwargs.get("test_mode") is True
    call_action = run_executor.call_args.args[0]
    assert "test_mode" not in type(call_action).model_fields
    assert call_action.dsl == "# workflow: demo"


def test_workflow_debug_does_not_lock_on_error() -> None:
    run_executor = MagicMock()
    run_executor.return_value = RunWorkflowObservation.from_text(
        text="failed",
        status="Failed",
        attempt=1,
        max_attempts=10,
        is_error=True,
    )
    observation = WorkflowDebugExecutor(run_executor=run_executor)(
        WorkflowDebugAction(dsl="# workflow: demo"),
        conversation=None,
    )
    assert isinstance(observation, WorkflowDebugObservation)
    assert observation.is_error is True
    assert observation.keep_ui_lock is False
