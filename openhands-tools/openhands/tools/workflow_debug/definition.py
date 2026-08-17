"""Definition of the ``workflow_debug`` tool.

Thin wrapper around :class:`~openhands.tools.workflow.run_workflow.RunWorkflowTool`
that always submits with ``test_mode=True`` (platform ``execution_mode=test``).

测试 / 调试 / 试跑统一走本工具，不再让 Agent 直接调用
``run_workflow(test_mode=true)``。异步提交成功时返回 ``keep_ui_lock=True``
（``run_workflow`` 正式执行不返回该字段）。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.workflow.definition import WORKFLOW_RELATIVE_PATH
from openhands.tools.workflow.run_workflow import (
    DEFAULT_MAX_ATTEMPTS,
    RunWorkflowAction,
    RunWorkflowExecutor,
    RunWorkflowObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


class WorkflowDebugAction(Action):
    """Trigger a platform test/debug run of the current Pyromind workflow DSL.

    触发当前 Pyromind 工作流 DSL 的平台测试/调试运行。
    """

    note: str | None = Field(
        default=None,
        description=(
            "Optional short note on what changed since the previous debug "
            "attempt. Not sent to the platform; purely for conversation history."
        ),
    )
    dsl_path: str | None = Field(
        default=None,
        description=(
            "Conversation-relative path to the workflow DSL file to debug. Pass "
            "the path of the declarative workflow script the agent generated "
            "or edited. If omitted, reads "
            "public_data/workflow_canvas/workflow.py from the workspace."
        ),
    )
    name: str = Field(
        default="workflow",
        description="Workflow name passed to the platform when submitting the run.",
    )
    reset_round: bool = Field(
        default=False,
        description=(
            "Start a new debug round by clearing the current round's submission "
            "count. Only use this when the tool reports the round attempt limit "
            "was reached AND the user explicitly asks to continue debugging."
        ),
    )


class WorkflowDebugObservation(Observation):
    """Result of one workflow debug/test attempt.

    Includes ``keep_ui_lock`` for the frontend; production ``run_workflow``
    observations do not carry this field.
    """

    status: Literal[
        "Succeeded", "Pending", "Running", "Failed", "Error", "Terminated"
    ] = Field(
        description=(
            "'Succeeded': 工作流执行完成; "
            "'Pending': 工作流提交成功，等待系统调度中; "
            "'Running': 工作流运行中; "
            "'Failed': 工作流提交失败; "
            "'Error': 工作流运行异常; "
            "'Terminated': 工作流被主动停止"
        )
    )
    error_log: str | None = Field(
        default=None,
        description=("Runtime error output when status is 'Failed' or 'Error'."),
    )
    task_id: str | None = Field(
        default=None,
        description="工作流提交成功，返回的任务 ID",
    )
    attempt: int = Field(description="This debug attempt number for the current round.")
    max_attempts: int = Field(
        description="The maximum number of debug attempts allowed."
    )
    keep_ui_lock: bool = Field(
        default=False,
        description=(
            "When true, the Web UI should stay locked until the platform "
            "callback delivers the terminal workflow status."
        ),
    )


TOOL_DESCRIPTION = """Test-run (debug) the current Pyromind workflow on the platform.

Use this tool whenever the user asks to test, debug, 测试, 调试, or 试跑 a workflow.
This is the only tool for debug/test runs — do **not** use `run_workflow` for
测试/调试/试跑 (`run_workflow` has no agent-facing `test_mode` parameter).

`public_data/workflow_canvas/workflow.py` is a declarative DSL, not a runnable Python script. Pass its
conversation-relative path as `dsl_path` (default `public_data/workflow_canvas/workflow.py`); the tool reads the file itself. Do not execute
it locally with bash or Python.

This tool delegates to `run_workflow`'s executor with internal `test_mode=true`
(platform `execution_mode=test`). `test_mode` is **not** an agent-facing
`run_workflow` action field — Agents must call this tool for test/debug runs.
Submission is **asynchronous**: it returns a task ID after the platform accepts
the run and sets `keep_ui_lock=true` so the Web UI stays locked until the
Kafka/platform callback delivers the terminal status. That callback injects a
`<system_reminder>` and **auto-continues** this conversation (`auto_run=true`).

Debug submissions use out_id `agent1#debug#<conversation_id>` so Kafka can tell
them apart from production `run_workflow` tasks (`agent1#<conversation_id>`).
Only debug-tagged terminals get the success/fail guidance below.

How to use the result:
- Submission Pending/Running with `keep_ui_lock=true`: tell the user the test was
  submitted; wait for the callback (do not poll).
- Terminal Succeeded (via callback, debug-tagged only): briefly tell the user this
  test workflow succeeded, then **wait for their next input**. Do not call
  workflow_debug again unless they ask.
- Terminal Failed/Error (debug-tagged): the DSL may be wrong. Read `error_log`,
  regenerate or fix `public_data/workflow_canvas/workflow.py` from the error, validate if needed, then call
  `workflow_debug` again to continue testing.
- Round limit reached: if the tool reports the round attempt limit (10 accepted
  submissions) was reached and the user explicitly asks to keep debugging, call
  `workflow_debug` again with `reset_round=true` to start a fresh round.
- Do not use this tool for production run/publish — use `run_workflow` instead.
  Production `run_workflow` does not return `keep_ui_lock`.
"""  # noqa: E501


def _to_debug_observation(
    observation: RunWorkflowObservation,
) -> WorkflowDebugObservation:
    """Map a run_workflow observation and set keep_ui_lock for async debug."""
    return WorkflowDebugObservation.from_text(
        text=observation.text,
        is_error=observation.is_error,
        status=observation.status,
        error_log=observation.error_log,
        task_id=observation.task_id,
        attempt=observation.attempt,
        max_attempts=observation.max_attempts,
        keep_ui_lock=not observation.is_error,
    )


class WorkflowDebugExecutor(
    ToolExecutor[WorkflowDebugAction, WorkflowDebugObservation]
):
    """Delegate to :class:`RunWorkflowExecutor` with internal ``test_mode=True``.

    Successful async submissions always surface ``keep_ui_lock=True`` so the
    frontend can lock the UI until the platform callback completes.
    """

    def __init__(
        self,
        *,
        cluster: str | None = None,
        env: str | None = None,
        current_user: object | None = None,
        headers: dict[str, str] | None = None,
        run_executor: RunWorkflowExecutor | None = None,
    ) -> None:
        self._run_executor = run_executor or RunWorkflowExecutor(
            cluster=cluster,
            env=env,
            current_user=current_user,
            headers=headers,
        )

    def _resolve_dsl_source(
        self,
        action: WorkflowDebugAction,
        conversation: BaseConversation | None,
    ) -> str:
        """Read the workflow DSL file referenced by ``action.dsl_path``.

        ``dsl_path`` is a conversation-relative path resolved against the
        conversation's working dir; it defaults to
        ``public_data/workflow_canvas/workflow.py``. Reads are confined to the
        workspace so a stray path cannot escape it.

        ``dsl_path`` 为相对工作区的路径，基于 conversation 的 working dir 解析，
        缺省为 ``public_data/workflow_canvas/workflow.py``；读取被限制在工作区内。
        """
        dsl_path = action.dsl_path or WORKFLOW_RELATIVE_PATH
        if conversation is None:
            raise ValueError(
                "Cannot read the workflow DSL file without an active conversation."
            )
        workspace = cast(Any, conversation).workspace
        base_dir = Path(workspace.working_dir).resolve()
        workflow_path = (base_dir / dsl_path).resolve()
        if not workflow_path.is_relative_to(base_dir):
            raise ValueError(
                f"Workflow DSL path must stay inside the workspace: {dsl_path!r}"
            )
        if not workflow_path.is_file():
            raise ValueError(
                f"Cannot read workflow DSL file: {dsl_path!r} does not exist."
            )
        return workflow_path.read_text(encoding="utf-8")

    def __call__(
        self,
        action: WorkflowDebugAction,
        conversation: BaseConversation | None = None,
    ) -> WorkflowDebugObservation:
        try:
            dsl_source = self._resolve_dsl_source(action, conversation)
        except ValueError as exc:
            return WorkflowDebugObservation.from_text(
                text=str(exc),
                status="Failed",
                attempt=0,
                max_attempts=DEFAULT_MAX_ATTEMPTS,
                is_error=True,
                error_log=str(exc),
            )
        observation = self._run_executor(
            RunWorkflowAction(
                note=action.note,
                dsl=dsl_source,
                name=action.name,
                reset_round=action.reset_round,
            ),
            conversation,
            test_mode=True,
        )
        return _to_debug_observation(observation)


class WorkflowDebugTool(ToolDefinition[WorkflowDebugAction, WorkflowDebugObservation]):
    """Tool that submits a Pyromind workflow test/debug run asynchronously."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        env = params.get("env", None)
        cluster = params.get("cluster", None)
        current_user = params.get("current_user", None)
        headers = params.get("headers", {})

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=WorkflowDebugAction,
                observation_type=WorkflowDebugObservation,
                executor=WorkflowDebugExecutor(
                    cluster=cluster,
                    env=env,
                    current_user=current_user,
                    headers=headers,
                ),
                annotations=ToolAnnotations(
                    title="workflow_debug",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(WorkflowDebugTool.name, WorkflowDebugTool)
