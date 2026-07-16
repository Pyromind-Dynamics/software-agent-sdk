"""Executor for the debug_workflow tool.

Blocks the calling (tool-executor) thread for the duration of one debug run.
This is safe because ``ToolExecutor.__call__`` already runs off the agent's
main loop thread (see ``ParallelToolExecutor``); a 30s-2min block here only
delays this one tool call's result, the same way the built-in browser tool
blocks for each action.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolExecutor
from openhands.tools.pyromind_debug.broker import get_debug_result_broker
from openhands.tools.pyromind_debug.definition import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_SECONDS,
    DebugWorkflowAction,
    DebugWorkflowObservation,
)
from openhands.tools.pyromind_debug.mock_platform import (
    DebugPlatformClient,
    MockDebugPlatform,
)
from openhands.tools.workflow.definition import WORKFLOW_RELATIVE_PATH


if TYPE_CHECKING:
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation

logger = get_logger(__name__)


class DebugWorkflowExecutor(
    ToolExecutor[DebugWorkflowAction, DebugWorkflowObservation]
):
    """Submits workflow.py to the debug platform and blocks for the result.

    Attempt counting lives on the executor instance, which is created once
    per conversation by ``DebugWorkflowTool.create`` -- so it resets if the
    conversation/agent is rebuilt (e.g. server restart). See the plan's
    "Known trade-offs" for when this should move to persisted conversation
    state instead.
    """

    def __init__(
        self,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        platform: DebugPlatformClient | None = None,
    ) -> None:
        self._max_attempts = max_attempts
        self._timeout_seconds = timeout_seconds
        self._platform: DebugPlatformClient = platform or MockDebugPlatform()
        self._attempt = 0

    def __call__(
        self,
        action: DebugWorkflowAction,
        conversation: LocalConversation | None = None,
    ) -> DebugWorkflowObservation:
        if conversation is None:
            return DebugWorkflowObservation.from_text(
                text="debug_workflow requires a local conversation context.",
                status="error",
                attempt=self._attempt,
                max_attempts=self._max_attempts,
                is_error=True,
            )

        working_dir = Path(conversation.workspace.working_dir)
        workflow_path = working_dir / WORKFLOW_RELATIVE_PATH
        if not workflow_path.is_file():
            return DebugWorkflowObservation.from_text(
                text=(
                    f"No workflow.py found at {workflow_path}. Create the "
                    "workflow before calling debug_workflow."
                ),
                status="error",
                attempt=self._attempt,
                max_attempts=self._max_attempts,
                is_error=True,
            )

        if self._attempt >= self._max_attempts:
            return DebugWorkflowObservation.from_text(
                text=(
                    f"Reached the maximum of {self._max_attempts} debug "
                    "attempts without a passing run. Stop calling "
                    "debug_workflow and report the remaining failure to "
                    "the user."
                ),
                status="error",
                attempt=self._attempt,
                max_attempts=self._max_attempts,
                is_error=True,
            )

        self._attempt += 1
        attempt = self._attempt
        workflow_source = workflow_path.read_text(encoding="utf-8")

        task_id = uuid4().hex
        broker = get_debug_result_broker()
        broker.register(task_id)
        logger.info(
            "Submitting debug_workflow attempt %d/%d (task_id=%s, note=%r)",
            attempt,
            self._max_attempts,
            task_id,
            action.note,
        )
        self._platform.submit(
            task_id=task_id, workflow_source=workflow_source, attempt=attempt
        )

        result = broker.wait(task_id, timeout=self._timeout_seconds)

        if result is None:
            return DebugWorkflowObservation.from_text(
                text=(
                    f"Debug run timed out after {self._timeout_seconds:.0f}s "
                    f"(attempt {attempt}/{self._max_attempts}). The platform "
                    "may still be running in the background; you may retry "
                    "debug_workflow."
                ),
                status="timeout",
                attempt=attempt,
                max_attempts=self._max_attempts,
                is_error=True,
            )

        if result.status == "passed":
            return DebugWorkflowObservation.from_text(
                text=f"Debug run passed on attempt {attempt}/{self._max_attempts}.",
                status="passed",
                attempt=attempt,
                max_attempts=self._max_attempts,
            )

        return DebugWorkflowObservation.from_text(
            text=(
                f"Debug run failed on attempt {attempt}/{self._max_attempts}. "
                f"Runtime error:\n{result.error_log}"
            ),
            status="failed",
            error_log=result.error_log,
            attempt=attempt,
            max_attempts=self._max_attempts,
        )
