from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pyromind_sdk.client.models import TrainingTaskCreateResponse

from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.tools.workflow import (
    RunWorkflowAction,
    RunWorkflowExecutor,
    RunWorkflowObservation,
    RunWorkflowTool,
)
from openhands.tools.workflow.run_workflow import DEFAULT_MAX_ATTEMPTS


_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000123")


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


def test_run_workflow_tool_create_and_resolve() -> None:
    params = {
        "env": "pre",
        "cluster": "us-west-1",
        "current_user": object(),
        "headers": {"x-cluster": "us-west-1#pre", "request-app": "openhands"},
    }
    tool = RunWorkflowTool.create(**params)[0]

    assert tool.name == "run_workflow"
    assert tool.annotations is not None
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.openWorldHint is True
    assert {"note", "dsl", "name", "test_mode"} <= set(tool.action_type.model_fields)

    resolved = resolve_tool(
        Tool(name="run_workflow", params=params),
        cast(Any, None),
    )
    assert isinstance(resolved[0], RunWorkflowTool)


def test_run_workflow_requires_conversation_context() -> None:
    observation = RunWorkflowExecutor(**_executor_kwargs())(
        RunWorkflowAction(dsl="# workflow"),
        conversation=None,
    )

    assert isinstance(observation, RunWorkflowObservation)
    assert observation.is_error
    assert observation.status == "Error"
    assert "requires a local conversation context" in observation.text


def test_run_workflow_reports_blank_env(tmp_path: Path) -> None:
    observation = RunWorkflowExecutor(**_executor_kwargs(env=None))(
        RunWorkflowAction(dsl="# workflow"),
        conversation=_fake_conversation(tmp_path),
    )

    assert observation.is_error
    assert observation.status == "Failed"
    assert observation.error_log == "param env is blank"


def test_run_workflow_reports_missing_dsl(tmp_path: Path) -> None:
    observation = RunWorkflowExecutor(**_executor_kwargs())(
        RunWorkflowAction(),
        conversation=_fake_conversation(tmp_path),
    )

    assert observation.is_error
    assert observation.status == "Failed"
    assert "Workflow DSL source code is required" in observation.text


def test_run_workflow_reports_dsl_conversion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        RunWorkflowExecutor,
        "_convert_dsl_to_xyflow",
        lambda self, *, dsl, name: SimpleNamespace(
            is_error=True,
            text="DslConverter unavailable",
            xyflow=None,
        ),
    )

    observation = RunWorkflowExecutor(**_executor_kwargs())(
        RunWorkflowAction(dsl="# workflow: demo"),
        conversation=_fake_conversation(tmp_path),
    )

    assert observation.is_error
    assert observation.status == "Failed"
    assert "DslConverter unavailable" in observation.text


def test_run_workflow_reports_missing_auth_token(tmp_path: Path) -> None:
    observation = RunWorkflowExecutor(**_executor_kwargs())(
        RunWorkflowAction(dsl="# workflow: demo"),
        conversation=_fake_conversation(tmp_path),
    )

    assert observation.is_error
    assert observation.status == "Failed"
    assert observation.error_log == "API key is required."


def test_run_workflow_submits_task_successfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = SecretRegistry()
    registry.update_secrets({"auth_token": "jwt-token"})

    mock_client = MagicMock()
    mock_client.studio.create.return_value = TrainingTaskCreateResponse(
        task_id="task-123",
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

    observation = RunWorkflowExecutor(**_executor_kwargs())(
        RunWorkflowAction(dsl="# workflow: demo", name="demo"),
        conversation=_fake_conversation(tmp_path, secret_registry=registry),
    )

    assert not observation.is_error
    assert observation.status == "Pending"
    assert observation.task_id == "task-123"
    assert observation.attempt == 1
    assert "task-123" in observation.text
    mock_client.studio.create.assert_called_once()
    request = mock_client.studio.create.call_args.args[0]
    assert request.out_id == f"agent1#{_CONVERSATION_ID}"
    assert request.workflow == {"name": "demo", "nodes": [], "edges": []}


def test_run_workflow_applies_test_mode_to_xyflow() -> None:
    executor = RunWorkflowExecutor(**_executor_kwargs())
    workflow = {"name": "demo", "nodes": [], "edges": []}

    updated = executor._resolve_add_test_mode(workflow, test_mode=True)

    assert updated["execution_argos"] == [{"execution_mode": "test"}]


def test_run_workflow_enforces_max_attempts(tmp_path: Path) -> None:
    executor = RunWorkflowExecutor(**_executor_kwargs())
    executor._attempt = DEFAULT_MAX_ATTEMPTS
    conversation = _fake_conversation(tmp_path)

    observation = executor(
        RunWorkflowAction(dsl="# workflow"),
        conversation=conversation,
    )

    assert observation.is_error
    assert observation.status == "Error"
    assert observation.attempt == DEFAULT_MAX_ATTEMPTS
    assert "Reached the maximum" in observation.text
