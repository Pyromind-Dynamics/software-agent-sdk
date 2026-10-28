"""One tool contract for background checkpoint evaluation across harnesses."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any, Literal, Self

from pydantic import Field
from pyromind_runtime.domain.context import RequestContext

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.data_preparation.submission import submission_files


CheckpointCall = Callable[
    [str, str, dict[str, Any], str, RequestContext], Coroutine[Any, Any, dict[str, Any]]
]

DESCRIPTION = """Manage server-side checkpoint evaluation while training runs.
Read inference-evaluation. register starts background polling of a confirmed training
 task and evaluates every complete checkpoint with one frozen validation template.
arguments: training_task_id, training_node_id, template (input_path,
dataset_config_path, evaluation_config_path, inference resource options, optional
scoring_spec_path and scoring_script_path, optional datasets.train/test with path),
selection (primary metric/direction, secondary objectives, constraints).
get and cancel take arguments.run_id. reconcile takes run_id, step and a verified
platform task_id to bind an uncertain submission; never guess a task ID.
cancel stops evaluation children, not training.
request_id is a UUID; reuse it for registration retries. The returned run_id identifies
this selection. Polling continues without an active Agent; do not loop or resubmit.
W&B is diagnostic; only complete validation evaluations rank checkpoints.
"""


class CheckpointEvaluationsAction(Action):
    action: Literal["register", "get", "cancel", "reconcile"]
    request_id: str = Field(
        description="Stable UUID for registration; reuse on retries."
    )
    arguments: dict[str, Any] = Field(default_factory=dict)


class CheckpointEvaluationsObservation(Observation):
    """A durable checkpoint selection or its progress."""


class CheckpointEvaluationsExecutor(
    ToolExecutor[CheckpointEvaluationsAction, CheckpointEvaluationsObservation]
):
    def __init__(
        self,
        callback: CheckpointCall,
        scope: str,
        context: RequestContext,
        loop: asyncio.AbstractEventLoop,
    ):
        self.callback, self.scope, self.context, self.loop = (
            callback,
            scope,
            context,
            loop,
        )

    def __call__(
        self, action: CheckpointEvaluationsAction, conversation: Any = None
    ) -> CheckpointEvaluationsObservation:
        try:
            arguments = dict(action.arguments)
            if action.action == "register":
                template = dict(arguments["template"])
                with submission_files(conversation) as files:
                    for key in (
                        "dataset_config",
                        "evaluation_config",
                        "scoring_spec",
                        "scoring_script",
                    ):
                        path = template.pop(key + "_path", None)
                        if path is not None:
                            content = files.read_text(path)
                            template[key] = (
                                content
                                if key == "scoring_script"
                                else json.loads(content)
                            )
                arguments["template"] = template
            result = asyncio.run_coroutine_threadsafe(
                self.callback(
                    self.scope,
                    action.action,
                    arguments,
                    action.request_id,
                    self.context,
                ),
                self.loop,
            ).result()
            return CheckpointEvaluationsObservation.from_text(
                json.dumps(result, ensure_ascii=False)
            )
        except Exception as error:
            return CheckpointEvaluationsObservation.from_text(
                str(error) if isinstance(error, ValueError) else type(error).__name__,
                is_error=True,
            )


class CheckpointEvaluationsTool(
    ToolDefinition[CheckpointEvaluationsAction, CheckpointEvaluationsObservation]
):
    @classmethod
    def create(cls, conv_state: Any = None, **params: Any) -> list[Self]:  # noqa: ARG003
        return [
            cls(
                description=DESCRIPTION,
                action_type=CheckpointEvaluationsAction,
                observation_type=CheckpointEvaluationsObservation,
                executor=CheckpointEvaluationsExecutor(**params) if params else None,
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(CheckpointEvaluationsTool.name, CheckpointEvaluationsTool)
