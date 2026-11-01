from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject


class DelegatedTask(ContractModel):
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


class ExperimentExecutionView(ContractModel):
    run_id: str
    project_id: str
    plan_revision: int | None = None
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


class RecordExperimentResult(ContractModel):
    run_id: str
    summary: str = Field(min_length=1, max_length=16000)
    report_uri: str | None = None
    blockers: tuple[str, ...] = ()


class ControlExperimentTask(ContractModel):
    run_id: str
    action: Literal["continue", "cancel", "retry_report"]
    command_id: str = Field(min_length=1, max_length=128)
    message: str = ""


class ReadExperimentResource(ContractModel):
    run_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    max_bytes: int = Field(default=16384, ge=1, le=65536)


ROLE_TOOLS = {
    "record_experiment_result": (
        RecordExperimentResult,
        "Register a report or essential missing inputs; Runtime verifies result evidence.",
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
