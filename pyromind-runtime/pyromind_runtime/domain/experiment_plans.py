from __future__ import annotations

from typing import Annotated, Literal

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
    enabled: bool = Field(
        default=True, description="Must be false when placeholder is true."
    )
    placeholder: bool = Field(
        default=False, description="An unimplemented stage; also set enabled=false."
    )

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


def _validate_task_references(
    stages: tuple[ExperimentStage, ...],
    tasks: tuple[ExperimentTask, ...],
    resources: tuple[PlanResource, ...],
) -> list[tuple[str, str]]:
    _unique([stage.id for stage in stages], "stage id")
    _unique([task.id for task in tasks], "task id")
    _unique([resource.id for resource in resources], "resource id")
    stage_ids = {stage.id for stage in stages}
    resource_ids = {resource.id for resource in resources}
    by_id = {task.id: task for task in tasks}
    dependencies: list[tuple[str, str]] = []
    for task in tasks:
        if task.stage_id not in stage_ids:
            raise ValueError("task references an unknown stage")
        _unique([item.name for item in task.inputs], "input name")
        _unique([item.name for item in task.outputs], "output name")
        for item in task.inputs:
            if item.resource_id and item.resource_id not in resource_ids:
                raise ValueError("input references an unknown resource")
            if item.task_id:
                upstream = by_id.get(item.task_id)
                if upstream is None or item.output_name not in {
                    output.name for output in upstream.outputs
                }:
                    raise ValueError("input references an unknown task output")
                dependencies.append((item.task_id, task.id))
    return dependencies


class WorkflowPlan(ContractModel):
    schema_version: Literal[1] = 1
    skill: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Planning Skill that produced this plan. Scenario topology comes from "
            "stages, tasks, and edges rather than the Skill name."
        ),
    )
    name: str = Field(min_length=1, max_length=255)
    goal: str = Field(min_length=1)
    resources: tuple[PlanResource, ...] = ()
    unresolved: tuple[str, ...] = ()
    stages: tuple[ExperimentStage, ...] = Field(min_length=1)
    tasks: tuple[ExperimentTask, ...] = ()
    edges: tuple[PlanEdge, ...] = ()

    @model_validator(mode="after")
    def validate_graph(self) -> WorkflowPlan:
        dependencies = _validate_task_references(
            self.stages, self.tasks, self.resources
        )
        stage_ids = {stage.id for stage in self.stages}
        stage_edges = [(edge.source, edge.target) for edge in self.edges]
        _unique([f"{a}\0{b}" for a, b in stage_edges], "edge")
        tasks = {task.id: task for task in self.tasks}
        stage_edges.extend(
            (tasks[source].stage_id, tasks[target].stage_id)
            for source, target in dependencies
            if tasks[source].stage_id != tasks[target].stage_id
        )
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


class ExperimentContext(ContractModel):
    """Optional advice, not an executable plan or an authorization boundary."""

    goal: str = ""
    constraints: tuple[str, ...] = ()
    resources: tuple[PlanResource, ...] = ()
    stages: tuple[ExperimentStage, ...] = ()
    stage_executions: dict[
        str, tuple[Annotated[str, Field(min_length=1, max_length=128)], ...]
    ] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
        description="Optional display associations: stage ID to existing TaskRun IDs. "
        "Does not schedule execution or change a task's frozen specification.",
    )
    tasks: tuple[ExperimentTask, ...] = ()
    edges: tuple[PlanEdge, ...] = ()
    conclusions: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    status: Literal["active", "completed"] = "active"

    @model_validator(mode="after")
    def validate_references(self):
        ids = [stage.id for stage in self.stages]
        if any(stage_id not in ids for stage_id in self.stage_executions):
            raise ValueError("execution association references an unknown stage")
        for run_ids in self.stage_executions.values():
            _unique(list(run_ids), "execution association")
        _validate_task_references(self.stages, self.tasks, self.resources)
        if any(e.source not in ids or e.target not in ids for e in self.edges):
            raise ValueError("guide edge references an unknown stage")
        # Advice may return to a previous stage. It is not a scheduling DAG.
        return self


class ExperimentContextView(ContractModel):
    project_id: str | None = None
    sequence: int = 0
    context: ExperimentContext = Field(default_factory=ExperimentContext)


class UpdateExperimentContext(ContractModel):
    context: ExperimentContext = Field(
        description=(
            "Complete replacement context, not a patch. Include all resources, "
            "stages, tasks and edges to keep; omitted arrays become empty. "
            "Task inputs reference resources or upstream task outputs."
        )
    )
    expected_sequence: int = Field(default=0, ge=0)


CONTEXT_TOOLS = {
    "get_experiment_context": ContractModel,
    "update_experiment_context": UpdateExperimentContext,
}


PLAN_TOOLS = {
    "create_experiment_plan": CreateExperimentPlan,
    "revise_experiment_plan": ReviseExperimentPlan,
    "get_experiment_plan": GetExperimentPlan,
}


class PlanConflictError(ValueError):
    pass
