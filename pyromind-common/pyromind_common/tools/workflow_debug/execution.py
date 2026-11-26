"""workflow_debug 内部提交能力，始终以测试模式提交。"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Literal, cast

from pydantic import Field
from pyromind_sdk import PyroMindAPIClient

from pyromind_common.platform.workflows import (
    WORKFLOW_ATTEMPT_STATE_KEY as WORKFLOW_ATTEMPT_STATE_KEY,
    prepare_debug_workflow,
)
from pyromind_common.sdk.logger import get_logger
from pyromind_common.sdk.tool.tool import (
    Action,
    Observation,
    ToolExecutor,
)
from pyromind_common.tools.workflow.dsl_to_xyflow import (
    convert_dsl_to_xyflow,
)
from pyromind_common.tools.workflow.task_submission import (
    PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET,
    create_workflow_api_client,
    submit_workflow_task,
)


logger = get_logger(__name__)
if TYPE_CHECKING:
    from pyromind_common.sdk.conversation.base import ToolConversation
    from pyromind_common.sdk.conversation.state import ToolState
DEFAULT_MAX_ATTEMPTS = 10


class WorkflowDebugSubmissionAction(Action):
    note: str | None = Field(
        default=None,
        description=(
            "Optional short note on what changed since the previous"
            " run attempt. Not sent to the platform; purely for "
            "conversation history."
        ),
    )
    dsl: str | None = Field(
        default=None,
        description=(
            "Pyromind workflow Python DSL source code to run (not a"
            " file path). Required: pass the declarative workflow "
            "script text the agent generated or edited."
        ),
    )
    name: str = Field(
        default="workflow",
        description="Workflow name passed to the platform when submitting the run.",
    )
    reset_round: bool = Field(
        default=False,
        description=(
            "Start a new round by clearing the current round's "
            "submission count. Only use this when the tool reports "
            "the round attempt limit was reached AND the user "
            "explicitly asks to continue debugging."
        ),
    )


class WorkflowDebugSubmissionObservation(Observation):
    status: Literal[
        "Succeeded", "Pending", "Running", "Failed", "Error", "Terminated"
    ] = Field(
        description=(
            "'Succeeded': 工作流执行完成; 'Pending': 工作流提交成功，等待系统调度中; "
            "'Running': 工作流运行中; 'Failed': 工作流提交失败; 'Error': "
            "工作流运行异常; 'Terminated': 工作流被主动停止"
        )
    )
    error_log: str | None = Field(
        default=None,
        description="Runtime error output when status is 'Failed' or 'Error'.",
    )
    task_id: str | None = Field(
        default=None, description="工作流提交成功，返回的任务 ID"
    )
    attempt: int = Field(description="This run attempt number for the current round.")
    max_attempts: int = Field(description="The maximum number of run attempts allowed.")


class WorkflowDebugSubmissionRunner:
    def __init__(
        self,
        *,
        cluster: str | None = None,
        env: str | None = None,
        current_user: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.cluster = cluster
        self.env = env
        self.current_user = current_user
        self._max_attempts = DEFAULT_MAX_ATTEMPTS
        self.headers = headers or {}
        self.submission_handler = None

    def __call__(
        self,
        action: WorkflowDebugSubmissionAction,
        conversation: ToolConversation | None = None,
    ) -> WorkflowDebugSubmissionObservation:
        return self.run(action, conversation)

    def run(
        self,
        action: WorkflowDebugSubmissionAction,
        conversation: ToolConversation | None = None,
    ) -> WorkflowDebugSubmissionObservation:
        if conversation is None:
            return WorkflowDebugSubmissionObservation.from_text(
                text="run_workflow requires a local conversation context.",
                status="Error",
                attempt=0,
                max_attempts=self._max_attempts,
                is_error=True,
            )
        attempts = self._read_attempts(conversation)
        if action.reset_round:
            attempts = 0
            self._write_attempts(conversation, 0)
        if attempts >= self._max_attempts:
            return WorkflowDebugSubmissionObservation.from_text(
                text=(
                    f"Reached the maximum of {self._max_attempts} accepted "
                    "submissions for this round. If the user explicitly asks "
                    "to continue, call again with reset_round=true to start a "
                    "new round; otherwise report the remaining failure to the user."
                ),
                status="Error",
                attempt=attempts,
                max_attempts=self._max_attempts,
                is_error=True,
                error_log="Reached the maximum number of run attempts.",
            )
        attempt_number = attempts + 1
        call_t0 = time.monotonic()
        try:
            if not self.env:
                raise ValueError("param env is blank")
            if not self.cluster:
                raise ValueError("param cluster is blank")
            if self.current_user is None:
                raise ValueError("param current_user is None")
            if conversation.id is None:
                raise ValueError("conversation.id is blank")
            dsl_t0 = time.monotonic()
            workflow_json = self._resolve_dsl(action)
            dsl_ms = (time.monotonic() - dsl_t0) * 1000
            state = cast("ToolState", conversation.state)
            auth_token = state.secret_registry.get_secret_value(
                PYROMIND_WORKFLOW_AUTH_TOKEN_SECRET
            )
            client = create_workflow_api_client(
                env=self.env,
                cluster=self.cluster,
                auth_token=auth_token,
                headers=self.headers,
            )
            submit_t0 = time.monotonic()
            observation = self._execute_run(
                client=client,
                workflow_json=workflow_json,
                workflow_name=action.name,
                attempt=attempt_number,
                conversation_id=str(conversation.id),
                source_code=action.dsl,
            )
            submit_ms = (time.monotonic() - submit_t0) * 1000
            logger.info(
                (
                    "[perf] run_workflow conversation_id=%s dsl_ms=%.1f "
                    "submit_ms=%.1f total_ms=%.1f test_mode=%s status=%s"
                ),
                conversation.id,
                dsl_ms,
                submit_ms,
                (time.monotonic() - call_t0) * 1000,
                True,
                observation.status,
            )
            if observation.status not in ("Failed", "Error"):
                self._write_attempts(conversation, attempt_number)
            return observation
        except Exception as exc:
            logger.info(
                "[perf] run_workflow total_ms=%.1f outcome=error",
                (time.monotonic() - call_t0) * 1000,
            )
            return WorkflowDebugSubmissionObservation.from_text(
                text=str(exc),
                status="Failed",
                attempt=attempt_number,
                max_attempts=self._max_attempts,
                is_error=True,
                error_log=str(exc),
            )

    def _read_attempts(self, conversation: ToolConversation) -> int:
        state = cast("ToolState", conversation.state)
        return int(state.agent_state.get(WORKFLOW_ATTEMPT_STATE_KEY, 0))

    def _write_attempts(self, conversation: ToolConversation, value: int) -> None:
        state = cast("ToolState", conversation.state)
        state.agent_state = {**state.agent_state, WORKFLOW_ATTEMPT_STATE_KEY: value}

    def _execute_run(
        self,
        *,
        client: PyroMindAPIClient,
        workflow_json: dict,
        workflow_name: str,
        attempt: int,
        conversation_id: str,
        source_code: str | None = None,
    ) -> WorkflowDebugSubmissionObservation:
        if workflow_json is None:
            return WorkflowDebugSubmissionObservation.from_text(
                text="Workflow xyflow is empty.",
                status="Failed",
                attempt=attempt,
                max_attempts=self._max_attempts,
                is_error=True,
                error_log="Workflow xyflow is empty.",
            )
        if not client:
            return WorkflowDebugSubmissionObservation.from_text(
                text="PyroMindAPIClient can not init.",
                status="Failed",
                attempt=attempt,
                max_attempts=self._max_attempts,
                is_error=True,
                error_log="PyroMindAPIClient can not init.",
            )
        try:
            workflow_xyflow = self._resolve_add_test_mode(workflow_json)
            _workflow_name = str(workflow_xyflow.get("name", workflow_name))
            response = submit_workflow_task(
                client=client,
                workflow=workflow_xyflow,
                name=_workflow_name,
                conversation_id=conversation_id,
                test_mode=True,
                retry_create=False,
                submission_handler=self.submission_handler,
                source_code=source_code,
            )
            user_text = (
                "The test workflow task has been submitted. Please wait "
                f"patiently. task id: {response.task_id}"
            )
            logger.info(
                "Run workflow successfully, msg = %s, mode = %s", user_text, "test"
            )
            return WorkflowDebugSubmissionObservation.from_text(
                text=user_text,
                status=response.status,
                task_id=response.task_id,
                attempt=attempt,
                max_attempts=self._max_attempts,
                is_error=False,
            )
        except Exception as exc:
            return WorkflowDebugSubmissionObservation.from_text(
                text=str(exc),
                status="Failed",
                attempt=attempt,
                max_attempts=self._max_attempts,
                is_error=True,
                error_log=str(exc),
            )

    def _resolve_add_test_mode(self, workflow_json: dict) -> dict:
        if not isinstance(workflow_json, dict):
            raise ValueError("Workflow JSON must be a JSON object.")
        return prepare_debug_workflow(workflow_json)

    def _resolve_workflow_dsl_source(
        self, action: WorkflowDebugSubmissionAction
    ) -> str:
        if action.dsl is None:
            raise ValueError(
                "Workflow DSL source code is required. Pass the agent-"
                "generated Python DSL in `dsl`."
            )
        dsl_source = action.dsl.strip()
        if not dsl_source:
            raise ValueError("Workflow DSL source code must be a non-empty string.")
        return dsl_source

    def _resolve_dsl(self, action: WorkflowDebugSubmissionAction) -> dict:
        dsl_source = self._resolve_workflow_dsl_source(action)
        if not action.name.strip():
            raise ValueError("Workflow name must be a non-empty string.")
        try:
            workflow = convert_dsl_to_xyflow(dsl_source, name=action.name)
        except Exception as exc:
            raise ValueError(
                f"Failed to convert workflow DSL to xyflow: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            json.dumps(workflow)
        except TypeError as exc:
            raise ValueError(
                f"Workflow DSL converter returned non-JSON data: {exc}"
            ) from exc
        return workflow


class WorkflowDebugSubmissionExecutor(
    WorkflowDebugSubmissionRunner,
    ToolExecutor[WorkflowDebugSubmissionAction, WorkflowDebugSubmissionObservation],
):
    def __call__(
        self,
        action: WorkflowDebugSubmissionAction,
        conversation: ToolConversation | None = None,
    ) -> WorkflowDebugSubmissionObservation:
        return self.run(action, conversation)
