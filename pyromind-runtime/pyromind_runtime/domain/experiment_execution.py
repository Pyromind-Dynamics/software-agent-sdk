from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.execution_policy import (
    CheckpointView,
    ExecutionCheckpoint,
)


class TaskArtifactRef(ContractModel):
    input_name: str = Field(min_length=1, max_length=128)
    task_run_id: str = Field(min_length=1)
    artifact_name: str = Field(min_length=1, max_length=128)


class DelegatedTask(ContractModel):
    artifact_refs: tuple[TaskArtifactRef, ...] = Field(
        default=(), max_length=16, exclude_if=lambda value: not value
    )
    name: str = Field(min_length=1, max_length=255)
    instruction: str = Field(min_length=1, max_length=32000)
    inputs: JsonObject = Field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()


class RequestExperimentTask(ContractModel):
    task_id: str | None = Field(default=None, min_length=1)
    expected_revision: int | None = Field(default=None, ge=1)
    role_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1, max_length=128)
    task: DelegatedTask | None = None
    intent: Literal["execute", "prepare"] = "execute"
    stage_id: str | None = None

    @model_validator(mode="after")
    def validate_contract(self):
        if self.task is None and (
            self.task_id is None or self.expected_revision is None
        ):
            raise ValueError("provide a task, or legacy task_id and expected_revision")
        if self.task is not None and (
            self.task_id is not None or self.expected_revision is not None
        ):
            raise ValueError("standalone tasks must not depend on a plan revision")
        return self


class DelegateExperimentTask(ContractModel):
    """Current model-facing contract; legacy requests remain API-compatible."""

    role_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1, max_length=128)
    task: DelegatedTask
    intent: Literal["execute", "prepare"] = "execute"
    stage_id: str | None = None


class GetExperimentExecution(ContractModel):
    run_id: str | None = None


class ReadExperimentArtifact(ContractModel):
    run_id: str = Field(min_length=1)
    name: str = Field(min_length=1, description="Name from the execution's artifacts.")


class ExecutionCondition(ContractModel):
    kind: Literal["external_task_succeeded", "artifacts_verified", "agent_result_valid"]


class EvaluationProposal(ContractModel):
    run_id: str = Field(min_length=1)
    configuration: JsonObject
    input_fingerprints: dict[str, str] = Field(min_length=1)
    expected_artifacts: tuple[str, ...] = Field(min_length=1)
    conditions: tuple[ExecutionCondition, ...] = Field(min_length=1)


EXECUTION_TOOLS = {
    "request_experiment_task": RequestExperimentTask,
    "get_experiment_execution": GetExperimentExecution,
    "submit_experiment_evaluation": EvaluationProposal,
}


class ExecutionBinding(ContractModel):
    project_id: str
    task_run_id: str
    stage_id: str
    plan_sequence: int


class ExperimentExecutionView(ContractModel):
    run_id: str
    project_id: str
    plan_revision: int | None = None
    plan_sequence: int | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    stage_snapshot: JsonObject | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    task_id: str
    stage_id: str | None = None
    task_name: str
    role_id: str
    state: str
    report_state: str
    fingerprint: str | None = None
    proposal: JsonObject | None = None
    metrics: JsonObject | None = None
    platform_task_id: str | None = None
    artifacts: tuple[JsonObject, ...] = ()
    intent: str = "prepare"
    summary: str | None = None
    report_uri: str | None = None
    blockers: tuple[str, ...] = ()
    waiting_reason: str | None = None
    checkpoint: CheckpointView | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ControlExperimentTask(ContractModel):
    run_id: str
    action: Literal["continue", "cancel", "retry_report"]
    command_id: str = Field(min_length=1, max_length=128)
    message: str = Field(
        default="",
        description="Guidance for the existing child context, regardless "
        "of task outcome or platform state. Use request_experiment_task with a new "
        "request_id for an independent task with fresh context.",
    )


class ReadExperimentResource(ContractModel):
    run_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    max_bytes: int = Field(default=16384, ge=1, le=65536)


class SubmitExecutionWorkflow(ContractModel):
    workflow_path: str = Field(min_length=1, max_length=4096)


ROLE_TOOLS = {
    "submit_workflow": (
        SubmitExecutionWorkflow,
        "Stage a validated Storage workflow for the existing user confirmation. "
        "This does not start training.",
    ),
    "record_execution_checkpoint": (
        ExecutionCheckpoint,
        "Register verified sample evidence and pause for Manager to ask the user. "
        "This does not authorize a platform submission.",
    ),
    "read_experiment_resource": (
        ReadExperimentResource,
        "Read a declared input; return a bounded preview and content digest.",
    ),
    "submit_experiment_evaluation": (
        EvaluationProposal,
        "Prepare an evaluation proposal for user review. "
        "This does not start a platform task.",
    ),
    "get_experiment_execution": (
        GetExperimentExecution,
        "Read this execution and its verified artifacts.",
    ),
}
