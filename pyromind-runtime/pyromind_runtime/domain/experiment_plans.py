from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from pyromind_runtime.domain.base import AwareDatetime, ContractModel, utc_now


class PlanResource(ContractModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1)
    uri: str | None = Field(
        default=None,
        description="Resource URI; if unknown, use null and note it in unresolved.",
    )
    description: str = ""


class ExperimentStage(ContractModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1)
    description: str = ""
    enabled: bool = True
    placeholder: bool = False

    @model_validator(mode="after")
    def validate_placeholder(self) -> ExperimentStage:
        if self.placeholder and self.enabled:
            raise ValueError("placeholder stages must be disabled")
        return self


class PlanInput(ContractModel):
    name: str = Field(min_length=1)
    resource_id: str | None = Field(
        default=None,
        description=(
            "Reference an entry in plan.resources. Required for external inputs; "
            "use either resource_id or task_id, never both or neither. "
            "For unknown resources, register a placeholder with uri=null and "
            "record missing details in unresolved; a name alone is not valid."
        ),
    )
    task_id: str | None = Field(
        default=None,
        description=(
            "Upstream task ID. Required with output_name for task outputs; "
            "leave resource_id null."
        ),
    )
    output_name: str | None = Field(
        default=None,
        description="Name in the upstream task's outputs; set only with task_id.",
    )

    @model_validator(mode="after")
    def validate_reference(self) -> PlanInput:
        if bool(self.resource_id) == bool(self.task_id):
            raise ValueError("input must reference one resource or upstream task")
        if bool(self.task_id) != bool(self.output_name):
            raise ValueError("task inputs require an output_name")
        return self


class PlanOutput(ContractModel):
    name: str = Field(min_length=1)
    description: str = ""


class CompletionCondition(ContractModel):
    kind: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ExperimentTask(ContractModel):
    id: str = Field(min_length=1, max_length=128)
    stage_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    inputs: tuple[PlanInput, ...] = ()
    outputs: tuple[PlanOutput, ...] = ()
    completion_conditions: tuple[CompletionCondition, ...] = ()


class PlanEdge(ContractModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


def _unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def _acyclic(ids: set[str], edges: list[tuple[str, str]]) -> None:
    incoming = dict.fromkeys(ids, 0)
    outgoing: dict[str, set[str]] = {key: set() for key in ids}
    for source, target in edges:
        if source not in ids or target not in ids:
            raise ValueError("edge references an unknown node")
        if target not in outgoing[source]:
            outgoing[source].add(target)
            incoming[target] += 1
    ready = [key for key, count in incoming.items() if count == 0]
    visited = 0
    while ready:
        visited += 1
        for target in outgoing[ready.pop()]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if visited != len(ids):
        raise ValueError("plan contains a cycle")


class WorkflowPlan(ContractModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=255)
    goal: str = Field(min_length=1)
    resources: tuple[PlanResource, ...] = ()
    unresolved: tuple[str, ...] = ()
    stages: tuple[ExperimentStage, ...] = Field(min_length=1)
    tasks: tuple[ExperimentTask, ...] = ()
    edges: tuple[PlanEdge, ...] = ()

    @model_validator(mode="after")
    def validate_graph(self) -> WorkflowPlan:
        _unique([stage.id for stage in self.stages], "stage id")
        _unique([task.id for task in self.tasks], "task id")
        _unique([resource.id for resource in self.resources], "resource id")
        stage_ids = {stage.id for stage in self.stages}
        resources = {resource.id for resource in self.resources}
        tasks = {task.id: task for task in self.tasks}
        dependencies: list[tuple[str, str]] = []
        stage_edges = [(edge.source, edge.target) for edge in self.edges]
        _unique([f"{a}\0{b}" for a, b in stage_edges], "edge")
        for task in self.tasks:
            if task.stage_id not in stage_ids:
                raise ValueError("task references an unknown stage")
            _unique([item.name for item in task.inputs], "input name")
            _unique([item.name for item in task.outputs], "output name")
            for item in task.inputs:
                if item.resource_id and item.resource_id not in resources:
                    raise ValueError("input references an unknown resource")
                if item.task_id:
                    upstream = tasks.get(item.task_id)
                    if upstream is None or item.output_name not in {
                        output.name for output in upstream.outputs
                    }:
                        raise ValueError("input references an unknown task output")
                    dependencies.append((item.task_id, task.id))
                    if upstream.stage_id != task.stage_id:
                        stage_edges.append((upstream.stage_id, task.stage_id))
        _acyclic(stage_ids, stage_edges)
        _acyclic(set(tasks), dependencies)
        return self


class PlanOwner(ContractModel):
    user_id: str
    conversation_id: str
    profile_id: str | None = None


class PlanProposal(ContractModel):
    proposal_id: str
    base_revision: int = Field(ge=0)
    plan: WorkflowPlan
    summary: str
    changes: tuple[str, ...] = ()
    status: Literal["pending", "superseded", "rejected", "confirmed"] = "pending"
    created_at: AwareDatetime = Field(default_factory=utc_now)


class PlanRevision(ContractModel):
    project_id: str
    revision: int = Field(ge=1)
    plan: WorkflowPlan
    confirmed_by: str
    confirmed_at: AwareDatetime = Field(default_factory=utc_now)


class PlanRevisionSummary(ContractModel):
    revision: int
    name: str
    confirmed_at: AwareDatetime


class ExperimentPlanView(ContractModel):
    project_id: str | None = None
    current_revision: int = 0
    current: PlanRevision | None = None
    pending_proposal: PlanProposal | None = None
    history: tuple[PlanRevisionSummary, ...] = ()


class CreateExperimentPlan(ContractModel):
    plan: WorkflowPlan
    summary: str = "初始实验计划"


class ReviseExperimentPlan(CreateExperimentPlan):
    expected_revision: int = Field(ge=0)
    source_proposal_id: str | None = None


class GetExperimentPlan(ContractModel):
    revision: int | None = Field(default=None, ge=1)


PLAN_TOOLS = {
    "create_experiment_plan": CreateExperimentPlan,
    "revise_experiment_plan": ReviseExperimentPlan,
    "get_experiment_plan": GetExperimentPlan,
}


class PlanConflictError(ValueError):
    pass
