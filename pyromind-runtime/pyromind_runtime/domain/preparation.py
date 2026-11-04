from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.experiment_execution import DelegateExperimentTask
from pyromind_runtime.domain.experiment_plans import ExperimentContext


class ChecklistField(ContractModel):
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = ""
    value_type: Literal["string", "object", "array", "integer", "boolean"] = "string"
    required: bool = True
    depends_on: tuple[str, ...] = ()
    required_keys: tuple[str, ...] = ()


class WorkflowTaskDefinition(ContractModel):
    id: str = Field(min_length=1)
    description: str = ""
    display_name: str | None = None


class WorkflowNodeDefinition(ContractModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    roles: tuple[str, ...] = ()
    max_instances: int | None = Field(default=None, ge=1)
    tasks: tuple[WorkflowTaskDefinition, ...] = ()


class PreparationPolicy(ContractModel):
    checklist_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    fields: tuple[ChecklistField, ...] = Field(min_length=1)
    allowed_roles: tuple[str, ...] = Field(min_length=1)

    workflow_nodes: tuple[WorkflowNodeDefinition, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_keys(self):
        node_ids = [node.id for node in self.workflow_nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("duplicate workflow node type")
        for node in self.workflow_nodes:
            task_ids = [task.id for task in node.tasks]
            if len(task_ids) != len(set(task_ids)):
                raise ValueError("duplicate workflow task type")
            if not set(node.roles) <= set(self.allowed_roles):
                raise ValueError("workflow node role is not enabled by this policy")
        keys = [item.key for item in self.fields]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate checklist field")
        if any(key not in keys for item in self.fields for key in item.depends_on):
            raise ValueError("unknown checklist dependency")
        pending = {item.key: set(item.depends_on) for item in self.fields}
        while pending:
            roots = {key for key, dependencies in pending.items() if not dependencies}
            if not roots:
                raise ValueError("cyclic checklist dependencies")
            pending = {
                key: dependencies - roots
                for key, dependencies in pending.items()
                if key not in roots
            }
        return self


class PreparationFact(ContractModel):
    value: JsonValue = None
    status: Literal["missing", "resolved", "conflict"] = "missing"
    evidence_refs: tuple[str, ...] = ()


class PreparationView(ContractModel):
    policy: PreparationPolicy
    revision: int = 0
    phase: Literal["collecting", "planning", "ready", "active"] = "collecting"
    facts: dict[str, PreparationFact] = Field(default_factory=dict)
    missing: tuple[str, ...] = ()
    plan: ExperimentContext | None = None
    entry_task: DelegateExperimentTask | None = None
    review_pending: bool = False
    project_id: str | None = None
    evidence: dict[str, str] = Field(default_factory=dict)


class UpdatePreparation(ContractModel):
    expected_revision: int = Field(ge=0)
    facts: dict[str, PreparationFact]
    plan: ExperimentContext | None = None
    entry_task: DelegateExperimentTask | None = None


class CollectPreparationFacts(ContractModel):
    expected_revision: int = Field(ge=0)
    facts: dict[str, PreparationFact]


class SetPreparationPlan(ContractModel):
    expected_revision: int = Field(ge=0)
    plan: ExperimentContext
    entry_task_id: str = Field(
        min_length=1, description="ID of the first task in plan.tasks"
    )
    role_id: str = Field(
        min_length=1,
        description="Enabled delegated role from preparation.policy.allowed_roles",
    )


class InspectResource(ContractModel):
    path: str = Field(min_length=1, max_length=4096)
    kind: Literal["file", "directory"] = "file"
    max_bytes: int = Field(default=16384, ge=1, le=65536)


class ProjectView(ContractModel):
    project_id: str
    conversation_id: str
    plan_sequence: int
    plan: ExperimentContext
    creation_command_id: str
    initial_run_id: str


PREPARATION_TOOLS = {
    "get_preparation": ContractModel,
    "update_preparation": CollectPreparationFacts,
    "set_preparation_plan": SetPreparationPlan,
    "inspect_resource": InspectResource,
}
