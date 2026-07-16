"""Definition of the debug_workflow tool.

This tool only *triggers* a debug run and blocks for its result; the
generate -> debug -> read error -> apply_patch -> re-debug loop itself is
driven by the LLM, guided by the ``debug-workflow`` skill. See the
"工作流 Debug 闭环" plan for the full design.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, Literal

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


DEFAULT_MAX_ATTEMPTS: Final[int] = 10
DEFAULT_TIMEOUT_SECONDS: Final[float] = 180.0


class DebugWorkflowAction(Action):
    """Trigger a real run of the current workflow.py on the debug platform."""

    note: str | None = Field(
        default=None,
        description=(
            "Optional short note on what changed since the previous debug "
            "attempt (e.g. 'fixed missing dataset_path'). Not sent to the "
            "platform; purely for your own conversation history."
        ),
    )


class DebugWorkflowObservation(Observation):
    """Result of one debug attempt."""

    status: Literal["passed", "failed", "timeout", "error"] = Field(
        description=(
            "'passed': the workflow ran successfully. "
            "'failed': the platform ran the workflow and it raised a runtime "
            "error (see error_log). "
            "'timeout': the platform did not respond in time; retrying is safe. "
            "'error': the tool could not even submit the run (e.g. no "
            "workflow.py, or the attempt limit was already reached)."
        )
    )
    error_log: str | None = Field(
        default=None,
        description="Runtime error output from the platform when status='failed'.",
    )
    attempt: int = Field(
        description="This debug attempt number for the current conversation."
    )
    max_attempts: int = Field(
        description="The maximum number of debug attempts allowed."
    )


_DEBUG_WORKFLOW_DESCRIPTION = """Test-run (debug) the current public_data/workflow_canvas/workflow.py: submit it to the
Pyromind training platform for a real execution and block until the result is known.
Whether the user says "test", "测试", "debug", "调试", or "试跑", they all mean this same
action -- there is no separate "test" vs "debug" tool.

`public_data/workflow_canvas/workflow.py` is a declarative DSL (it borrows Python syntax to describe nodes and their
connections), not a runnable Python script. There is no local interpreter for it and no
other way to execute or validate it -- this tool call is the *only* way to actually run it.
Do not try to run it with a shell/Python interpreter or reason about it using Python
runtime semantics; treat `error_log` below as the ground truth for what happened.

This call blocks for roughly 30 seconds to a few minutes while the platform
actually executes the workflow end-to-end. There is no separate "check
status" call and no progress feedback while waiting -- the tool call simply
returns once the run finishes (or times out).

How to use the result:
- status="passed": the workflow is verified. Tell the user it passed and stop.
- status="failed": this is a real runtime error from actually running the
  workflow (not a static lint). Read `error_log` carefully, use `apply_patch`
  to fix only the specific lines responsible for the error -- do not rewrite
  the whole file -- and then call `debug_workflow` again.
- status="timeout": the platform did not respond in time. You may call
  `debug_workflow` again to retry the same workflow.
- status="error": the run could not even be submitted (for example,
  public_data/workflow_canvas/workflow.py does not exist yet, or you already used all allowed attempts).
  Do not call `debug_workflow` again in this case.

Each conversation allows a limited number of debug attempts, enforced by the
tool itself. `attempt`/`max_attempts` in the observation tell you how many
you have used. If you reach the limit without a passing run, stop calling
this tool and report the remaining failure to the user instead.
"""  # noqa: E501


class DebugWorkflowTool(ToolDefinition[DebugWorkflowAction, DebugWorkflowObservation]):
    """Tool that submits workflow.py for a real debug run and waits for it."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,
        **params,
    ) -> Sequence[ToolDefinition]:
        del conv_state
        from openhands.tools.pyromind_debug.impl import DebugWorkflowExecutor

        max_attempts = int(params.get("max_attempts", DEFAULT_MAX_ATTEMPTS))
        timeout_seconds = float(params.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        callback_base_url = params.get("callback_base_url")
        platform = None
        if callback_base_url is not None:
            from openhands.tools.pyromind_debug.mock_platform import MockDebugPlatform

            platform = MockDebugPlatform(callback_base_url=callback_base_url)
        executor = DebugWorkflowExecutor(
            max_attempts=max_attempts,
            timeout_seconds=timeout_seconds,
            platform=platform,
        )
        return [
            cls(
                description=_DEBUG_WORKFLOW_DESCRIPTION,
                action_type=DebugWorkflowAction,
                observation_type=DebugWorkflowObservation,
                annotations=ToolAnnotations(
                    title="debug_workflow",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


# New Pyromind conversations use run_workflow(test_mode=True), but persisted
# conversations may still reference this legacy tool by name.
register_tool(DebugWorkflowTool.name, DebugWorkflowTool)
