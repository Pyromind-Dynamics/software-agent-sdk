"""Dedicated, flat tool contract for model evaluation and derived artifacts."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import Field, model_validator

from openhands.sdk.tool import (
    Action,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.data_preparation.evaluation_products import (
    EvaluationProductsOptions,
)
from openhands.tools.data_preparation.inference import (
    EvaluationConfigurationError,
    InferenceOptions,
    InferencePipelineHandler,
)
from openhands.tools.data_preparation.prediction_evaluation import (
    PredictionEvaluationOptions,
)
from openhands.tools.data_preparation.submission import (
    SubmissionConfig,
    SubmissionObservation,
    submission_config,
    submission_files,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState

EvaluationOperation = Literal[
    "inference", "score", "profile", "review", "report", "resume"
]

_INFERENCE_FIELDS = frozenset(InferenceOptions.model_fields)
_SCORE_FIELDS = frozenset(PredictionEvaluationOptions.model_fields)
_PRODUCT_FIELDS = frozenset({"manifest_path", "extraction_script_path"})
_COMMON_FIELDS = frozenset({"operation", "input_path", "cpu", "memory"})
_REQUIRED_FIELDS = {
    "inference": ("model_path", "dataset_config_path", "evaluation_config_path"),
    "score": ("source_manifest_path", "scoring_spec_path", "scoring_script_path"),
    "profile": ("manifest_path", "extraction_script_path"),
    "review": ("manifest_path",),
    "report": ("manifest_path",),
    "resume": ("resume_run_id",),
}

TOOL_DESCRIPTION = """\
Submit model evaluation, prediction scoring or report artifacts to Pyromind.

Choose operation and pass its fields directly at the top level:
- inference: model_path, dataset_config_path, evaluation_config_path. Generates
  optional model preparation -> VLLMInference -> CPU scoring. Set gpu_product and
  gpu_count for the chosen model. Model source is in evaluation_config JSON.
  selection_run_id binds an independent test to its validation-selected checkpoint.
- score: source_manifest_path, scoring_spec_path, scoring_script_path. Scores
  existing predictions on CPU; does not run a model or initialize annotations.
- profile: manifest_path, extraction_script_path defining parse_record.
- review: manifest_path identifying an existing profile and scoring/badcase source.
- report: manifest_path identifying results to render. CPU-only.
- resume: resume_run_id and the unchanged input_path; reuses frozen inputs and
  configuration. Omit cpu/memory to inherit resources. Changed inputs or rules
  require a new run; stop an active task before resuming.

input_path is a Storage path. Configs, manifests and scripts are workspace files.
No inference/evaluation/products wrapper object, mode, script_path, model_profile
or output_schema. For user-authored data-processing pipelines use df_submit_pipeline.
Read inference-evaluation before preparing the task. Submit only when requested.
Returns task_id, run_id and output_dir asynchronously; df_check_progress and
df_stop_task query or stop it. After the callback, inspect report.json and its
artifacts; task completion alone does not establish business success. Never
blindly resubmit when platform acceptance is uncertain.
"""


class DfSubmitEvaluationAction(Action):
    """Submit an evaluation operation with flat, operation-specific fields."""

    operation: EvaluationOperation = Field(
        description="Task: inference, score, profile, review, report, or resume."
    )
    input_path: str = Field(
        min_length=1,
        description="Source data path in Pyromind Storage; unchanged on resume.",
    )
    model_path: str | None = Field(
        default=None,
        min_length=1,
        description="inference: model reference; source is in evaluation config JSON.",
    )
    dataset_config_path: str | None = Field(
        default=None,
        min_length=1,
        description="inference: workspace JSON mapping input and reference fields.",
    )
    evaluation_config_path: str | None = Field(
        default=None,
        min_length=1,
        description="inference: workspace rubric, generation and model_source JSON.",
    )
    selection_run_id: str | None = Field(
        default=None,
        min_length=1,
        description="inference: validation selection run authorizing independent test.",
    )
    report_spec_path: str | None = Field(
        default=None,
        min_length=1,
        description="inference: optional workspace report specification JSON.",
    )
    served_model_name: str | None = Field(default=None, min_length=1)
    gpu_product: str | None = Field(default=None, min_length=1)
    gpu_count: int | None = Field(default=None, ge=1)
    port: int | None = Field(default=None, ge=1, le=65535)
    max_model_len: int | None = Field(default=None, ge=1)
    source_manifest_path: str | None = Field(
        default=None,
        min_length=1,
        description="score: workspace prediction sources and checkpoints manifest.",
    )
    scoring_spec_path: str | None = Field(
        default=None,
        min_length=1,
        description="score: workspace business rules and validation examples JSON.",
    )
    scoring_script_path: str | None = Field(
        default=None,
        min_length=1,
        description="score: workspace Python with parse_record and score_case.",
    )
    manifest_path: str | None = Field(
        default=None,
        min_length=1,
        description="profile/review/report: workspace artifact manifest JSON.",
    )
    extraction_script_path: str | None = Field(
        default=None,
        min_length=1,
        description="profile only: workspace Python script defining parse_record.",
    )
    resume_run_id: uuid.UUID | None = Field(
        default=None, description="resume only: existing evaluation run UUID."
    )
    cpu: int | None = Field(
        default=None,
        ge=1,
        le=64,
        description="CPU cores; new runs default to 4, resume inherits when omitted.",
    )
    memory: int | None = Field(
        default=None,
        ge=1,
        le=256,
        description="Memory GiB; new runs default to 32, resume inherits when omitted.",
    )

    @model_validator(mode="after")
    def operation_fields(self) -> Self:
        missing = [
            key
            for key in _REQUIRED_FIELDS[self.operation]
            if getattr(self, key) is None
        ]
        if missing:
            raise ValueError(
                f"operation={self.operation} requires top-level fields: "
                + ", ".join(missing)
            )
        allowed = (
            _COMMON_FIELDS
            | {
                "inference": _INFERENCE_FIELDS,
                "score": _SCORE_FIELDS,
                "profile": _PRODUCT_FIELDS,
                "review": frozenset({"manifest_path"}),
                "report": frozenset({"manifest_path"}),
                "resume": frozenset({"resume_run_id"}),
            }[self.operation]
        )
        supplied = {
            key for key in self.model_fields_set if getattr(self, key, None) is not None
        }
        invalid = supplied - allowed - {"kind"}
        if invalid:
            raise ValueError(
                f"operation={self.operation} does not accept: "
                + ", ".join(sorted(invalid))
            )
        return self

    def request(self) -> EvaluationRequest:
        """Build one typed request independent of the script tool."""
        values = self.model_dump(exclude_none=True)
        options: (
            InferenceOptions
            | PredictionEvaluationOptions
            | EvaluationProductsOptions
            | None
        )
        if self.operation == "inference":
            options = InferenceOptions.model_validate(
                {key: values[key] for key in _INFERENCE_FIELDS if key in values}
            )
        elif self.operation == "score":
            options = PredictionEvaluationOptions.model_validate(
                {key: values[key] for key in _SCORE_FIELDS if key in values}
            )
        elif self.operation in {"profile", "review", "report"}:
            options = EvaluationProductsOptions.model_validate(
                {
                    "operation": self.operation,
                    **{key: values[key] for key in _PRODUCT_FIELDS if key in values},
                }
            )
        else:
            options = None
        return EvaluationRequest(
            input_path=self.input_path,
            options=options,
            resume_run_id=self.resume_run_id,
            cpu=self.cpu if self.cpu is not None else 4,
            memory=self.memory if self.memory is not None else 32,
            resource_overrides=frozenset(
                key for key in ("cpu", "memory") if values.get(key) is not None
            ),
        )


@dataclass(frozen=True, kw_only=True)
class EvaluationRequest:
    input_path: str
    options: (
        InferenceOptions
        | PredictionEvaluationOptions
        | EvaluationProductsOptions
        | None
    )
    resume_run_id: uuid.UUID | None = None
    cpu: int = 4
    memory: int = 32
    resource_overrides: frozenset[str] = frozenset()


class DfSubmitEvaluationObservation(SubmissionObservation):
    """Asynchronous evaluation receipt or submission error."""


class DfSubmitEvaluationExecutor(
    ToolExecutor[DfSubmitEvaluationAction, DfSubmitEvaluationObservation]
):
    def __init__(
        self,
        *,
        config: SubmissionConfig | None = None,
        inference_handler: InferencePipelineHandler | None = None,
    ) -> None:
        self.config = config or SubmissionConfig()
        self.inference_handler = inference_handler

    def __call__(
        self,
        action: DfSubmitEvaluationAction,
        conversation: BaseConversation | None = None,
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


class DfSubmitEvaluationTool(
    ToolDefinition[DfSubmitEvaluationAction, DfSubmitEvaluationObservation]
):
    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        config = submission_config(params)
        handler = params.pop("inference_handler", None)
        if params:
            raise ValueError(
                "DfSubmitEvaluationTool got unknown params: "
                + ", ".join(sorted(params))
            )
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=DfSubmitEvaluationAction,
                observation_type=DfSubmitEvaluationObservation,
                executor=DfSubmitEvaluationExecutor(
                    config=config, inference_handler=handler
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
