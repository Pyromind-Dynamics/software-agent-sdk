from __future__ import annotations

from typing import Literal

from pydantic import Field

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject


class RequestExperimentTask(ContractModel):
    task_id: str = Field(min_length=1)
    expected_revision: int = Field(ge=1)
    role_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1, max_length=128)


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
    plan_revision: int
    task_id: str
    stage_id: str
    task_name: str
    role_id: str
    state: str
    report_state: str
    fingerprint: str | None = None
    proposal: JsonObject | None = None
    metrics: JsonObject | None = None
    platform_task_id: str | None = None
    artifacts: tuple[JsonObject, ...] = ()


class ReadExperimentResource(ContractModel):
    run_id: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    max_bytes: int = Field(default=16384, ge=1, le=65536)


ROLE_TOOLS = {
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
