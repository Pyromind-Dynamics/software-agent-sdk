from unittest.mock import Mock

import pytest

from pyromind_common.sdk.tool import ToolExecutor
from pyromind_common.tools.workflow.task_submission import workflow_submission_scope
from pyromind_common.tools.workflow_debug import execution
from pyromind_common.tools.workflow_debug.definition import (
    WorkflowDebugAction,
    WorkflowDebugExecutor,
    WorkflowDebugTool,
    create_workflow_debug_runner,
)
from pyromind_common.tools.workflow_debug.execution import (
    WORKFLOW_ATTEMPT_STATE_KEY,
    WorkflowDebugSubmissionExecutor,
    WorkflowDebugSubmissionRunner,
)
from pyromind_runtime.domain.task_correlation import TaskOrigin, parse_task_correlation


def test_plain_debug_runner_reads_workspace_and_submits_sdk_test_without_tools(
    monkeypatch, debug_conversation, debug_client
):
    _, create = debug_client
    legacy = Mock(side_effect=AssertionError("legacy Tool must not execute"))
    monkeypatch.setattr(WorkflowDebugTool, "create", legacy)
    monkeypatch.setattr(WorkflowDebugExecutor, "__call__", legacy)
    monkeypatch.setattr(WorkflowDebugSubmissionExecutor, "__call__", legacy)
    runner = create_workflow_debug_runner(
        env="pre2", cluster="cn-east-1", current_user={"id": "42"}
    )

    with workflow_submission_scope(TaskOrigin()) as scope:
        result = runner.run(WorkflowDebugAction(name="debug"), debug_conversation)

    assert not isinstance(runner, ToolExecutor)
    assert not issubclass(WorkflowDebugSubmissionRunner, ToolExecutor)
    assert result.task_id == "debug-task"
    assert result.keep_ui_lock is True
    assert result.attempt == 1
    assert debug_conversation.state.agent_state[WORKFLOW_ATTEMPT_STATE_KEY] == 1
    assert scope.submissions["debug-task"].task_kind == "workflow_debug"
    factory = execution.create_workflow_api_client
    assert isinstance(factory, Mock)
    factory.assert_called_once_with(
        env="pre2", cluster="cn-east-1", auth_token="request-token", headers={}
    )
    converter = execution.convert_dsl_to_xyflow
    assert isinstance(converter, Mock)
    converter.assert_called_once_with("workflow_dsl = 'test'", name="debug")
    request = create.call_args.args[0]
    assert request.workflow["execution_argos"] == [{"execution_mode": "test"}]
    correlation = parse_task_correlation(request.out_id)
    assert correlation is not None
    assert correlation.conversation_id == debug_conversation.id
    assert correlation.task_kind == "workflow_debug"
    legacy.assert_not_called()


def test_debug_budget_stops_at_ten_and_reset_uses_the_same_state_key(
    debug_conversation, debug_client
):
    _, create = debug_client
    debug_conversation.state.agent_state[WORKFLOW_ATTEMPT_STATE_KEY] = 9
    runner = create_workflow_debug_runner(
        env="prod", cluster="us-west-1", current_user={"id": "42"}
    )

    accepted = runner.run(WorkflowDebugAction(), debug_conversation)
    blocked = runner.run(WorkflowDebugAction(), debug_conversation)

    assert accepted.attempt == 10
    assert blocked.is_error
    assert blocked.keep_ui_lock is False
    assert blocked.attempt == 10
    assert create.call_count == 1

    reset = runner.run(WorkflowDebugAction(reset_round=True), debug_conversation)

    assert reset.attempt == 1
    assert debug_conversation.state.agent_state[WORKFLOW_ATTEMPT_STATE_KEY] == 1
    assert create.call_count == 2


@pytest.mark.parametrize("failure", ["conversion", "submission"])
def test_debug_failures_do_not_consume_budget_or_retry_create(
    monkeypatch, debug_conversation, debug_client, failure
):
    _, create = debug_client
    debug_conversation.state.agent_state[WORKFLOW_ATTEMPT_STATE_KEY] = 3
    if failure == "conversion":
        monkeypatch.setattr(
            execution,
            "convert_dsl_to_xyflow",
            Mock(side_effect=ValueError("invalid DSL")),
        )
    else:
        create.side_effect = OSError("connection lost")
    runner = create_workflow_debug_runner(
        env="prod", cluster="us-west-1", current_user={"id": "42"}
    )

    result = runner.run(WorkflowDebugAction(), debug_conversation)

    assert result.is_error
    assert result.status == "Failed"
    assert result.attempt == 4
    assert result.keep_ui_lock is False
    assert debug_conversation.state.agent_state[WORKFLOW_ATTEMPT_STATE_KEY] == 3
    assert create.call_count == (1 if failure == "submission" else 0)


def test_debug_workspace_errors_do_not_submit(debug_conversation, debug_client):
    _, create = debug_client
    runner = create_workflow_debug_runner()

    result = runner.run(
        WorkflowDebugAction(dsl_path="../outside.py"), debug_conversation
    )

    assert result.is_error
    assert result.attempt == 0
    assert "inside the workspace" in result.text
    assert runner.run(WorkflowDebugAction()).is_error
    create.assert_not_called()


def test_legacy_debug_executor_delegates_to_the_plain_submission_runner(
    debug_conversation, debug_client
):
    executor = WorkflowDebugExecutor(
        run_executor=WorkflowDebugSubmissionExecutor(
            env="prod", cluster="us-west-1", current_user={"id": "42"}
        )
    )

    result = executor(WorkflowDebugAction(), debug_conversation)

    assert result.task_id == "debug-task"
    assert result.keep_ui_lock is True
