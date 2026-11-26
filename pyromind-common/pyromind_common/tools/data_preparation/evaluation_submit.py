"""Dedicated, flat tool contract for model evaluation and derived artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

from pyromind_common.platform.evaluation_core import (
    _COMMON_FIELDS as _COMMON_FIELDS,
    _INFERENCE_FIELDS as _INFERENCE_FIELDS,
    _PRODUCT_FIELDS as _PRODUCT_FIELDS,
    _REQUIRED_FIELDS as _REQUIRED_FIELDS,
    _SCORE_FIELDS as _SCORE_FIELDS,
    EvaluationOperation as EvaluationOperation,
    EvaluationOptions,
    EvaluationRequest as EvaluationRequest,
)
from pyromind_common.sdk.tool import (
    Action,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from pyromind_common.tools.data_preparation.inference import (
    EvaluationConfigurationError,
    InferencePipelineHandler,
)
from pyromind_common.tools.data_preparation.submission import (
    SubmissionConfig,
    SubmissionObservation,
    submission_config,
    submission_files,
)


if TYPE_CHECKING:
    from pyromind_common.sdk.conversation.base import ToolConversation
    from pyromind_common.sdk.conversation.state import ToolState


TOOL_DESCRIPTION = """\
Submit model evaluation, prediction scoring or report artifacts to Pyromind.

Choose operation and pass its fields directly at the top level:
- inference: model_path, dataset_config_path, evaluation_config_path. Generates
  optional model preparation -> VLLMInference -> CPU scoring. Set gpu_product and
  gpu_count for the chosen model. Model source is in evaluation_config JSON.
  source_checkpoint_version_id binds the registered checkpoint used as model_path.
  selection_run_id binds an independent test to its validation-selected checkpoint.
- score: source_manifest_path, scoring_spec_path, scoring_script_path. Scores
  existing predictions on CPU; does not run a model or initialize annotations.
- profile: manifest_path, extraction_script_path defining parse_record.
- review: manifest_path identifying an existing profile and scoring/badcase source.
- report: manifest_path identifying results to render. CPU-only.
- resume: resume_run_id and the unchanged input_path; reuses frozen inputs and
  configuration. Omit cpu/memory to inherit resources. Changed inputs or rules
  require a new run; stop an active task before resuming.

input_path is a user Storage path such as `/workspace/datasets/test.jsonl`.
Configs, manifests and scripts are conversation workspace files.
No inference/evaluation/products wrapper object, mode, script_path, model_profile
or output_schema. For user-authored data-processing pipelines use df_submit_pipeline.
Read inference-evaluation before preparing the task. Submit only when requested.
Returns task_id, run_id and a `/workspace/...` output_dir asynchronously;
df_check_progress and
df_stop_task query or stop it. After the callback, inspect report.json and its
artifacts; task completion alone does not establish business success. Never
blindly resubmit when platform acceptance is uncertain.
"""


class DfSubmitEvaluationAction(EvaluationOptions, Action):
    """Legacy Tool schema backed by the plain evaluation contract."""


class DfSubmitEvaluationObservation(SubmissionObservation):
    """Asynchronous evaluation receipt or submission error."""


class EvaluationSubmissionRunner:
    def __init__(
        self,
        *,
        config: SubmissionConfig | None = None,
        inference_handler: InferencePipelineHandler | None = None,
    ) -> None:
        self.config = config or SubmissionConfig()
        self.inference_handler = inference_handler

    def run(
        self,
        action: DfSubmitEvaluationAction,
        conversation: ToolConversation | None = None,
    ) -> DfSubmitEvaluationObservation:
        if conversation is None:
            return DfSubmitEvaluationObservation.from_text(
                text="df_submit_evaluation requires an active conversation.",
                status="Failed",
                is_error=True,
            )
        try:
            # Validate copied actions before reading or uploading any files.
            action = DfSubmitEvaluationAction.model_validate(action.model_dump())
            if self.inference_handler is None:
                raise ValueError("inference evaluation is not configured")
            with submission_files(conversation) as files:
                return DfSubmitEvaluationObservation.from_text(
                    **self.inference_handler.submit(
                        self.config, action.request(), conversation, files
                    )
                )
        except Exception as exc:
            return DfSubmitEvaluationObservation.from_text(
                text=f"Evaluation submission failed: {exc}",
                status="Failed",
                is_error=True,
                failure_stage="config_validation"
                if isinstance(exc, (ValueError, EvaluationConfigurationError))
                else None,
                retryable=False if isinstance(exc, ValueError) else None,
            )


class DfSubmitEvaluationExecutor(
    EvaluationSubmissionRunner,
    ToolExecutor[DfSubmitEvaluationAction, DfSubmitEvaluationObservation],
):
    def __call__(
        self,
        action: DfSubmitEvaluationAction,
        conversation: ToolConversation | None = None,
    ) -> DfSubmitEvaluationObservation:
        return self.run(action, conversation)


def _evaluation_runner_options(params: dict[str, Any]) -> dict[str, Any]:
    config = submission_config(params)
    handler = params.pop("inference_handler", None)
    if params:
        raise ValueError(
            "DfSubmitEvaluationTool got unknown params: " + ", ".join(sorted(params))
        )
    return {"config": config, "inference_handler": handler}


def create_evaluation_runner(**params: Any) -> EvaluationSubmissionRunner:
    return EvaluationSubmissionRunner(**_evaluation_runner_options(params))


class DfSubmitEvaluationTool(
    ToolDefinition[DfSubmitEvaluationAction, DfSubmitEvaluationObservation]
):
    @classmethod
    def create(
        cls,
        conv_state: ToolState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=DfSubmitEvaluationAction,
                observation_type=DfSubmitEvaluationObservation,
                executor=DfSubmitEvaluationExecutor(
                    **_evaluation_runner_options(params)
                ),
                annotations=ToolAnnotations(
                    title="df_submit_evaluation",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool("df_submit_evaluation", DfSubmitEvaluationTool)
