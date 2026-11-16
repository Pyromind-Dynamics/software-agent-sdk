from __future__ import annotations

from typing import Literal

from pydantic import Field, GetJsonSchemaHandler, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.experiment_execution import DelegateExperimentTask
from pyromind_runtime.domain.experiment_plans import ExperimentContext


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
    allowed_roles: tuple[str, ...] = Field(min_length=1)
    workflow_nodes: tuple[WorkflowNodeDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self):
        ids = [node.id for node in self.workflow_nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate workflow node type")
        for node in self.workflow_nodes:
            if not set(node.roles) <= set(self.allowed_roles):
                raise ValueError("workflow node role is not enabled by this policy")
            tasks = [task.id for task in node.tasks]
            if len(tasks) != len(set(tasks)):
                raise ValueError("duplicate workflow task type")
        return self


class PreparationState(ContractModel):
    """Durable plan and review state."""

    revision: int = 0
    phase: Literal["collecting", "planning", "ready", "active"] = "collecting"
    plan: ExperimentContext | None = None
    entry_task: DelegateExperimentTask | None = None
    review_pending: bool = False
    project_id: str | None = None


class PreparationView(PreparationState):
    """Review state and workflow catalog exposed to the frontend."""

    policy: PreparationPolicy

    @classmethod
    def from_state(
        cls, state: PreparationState, policy: PreparationPolicy
    ) -> PreparationView:
        return cls(**state.model_dump(), policy=policy)


class SetPreparationPlan(ContractModel):
    plan: ExperimentContext
    entry_task_id: str = Field(
        min_length=1, description="ID from plan.tasks of the task to run first"
    )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        plan = handler.resolve_ref_schema(schema["properties"]["plan"])
        for name in ("goal", "stages", "tasks"):
            plan.setdefault("required", []).append(name)
            plan["properties"][name].pop("default", None)
        for name in ("stages", "tasks"):
            collection = plan["properties"][name]
            collection["minItems"] = 1
            item = handler.resolve_ref_schema(collection["items"])
            item.setdefault("required", []).append("kind")
            item["properties"]["kind"].pop("default", None)
        return schema


class ProjectView(ContractModel):
    project_id: str
    conversation_id: str
    plan_sequence: int
    plan: ExperimentContext
    creation_command_id: str
    cluster: str | None = None


PREPARATION_TOOLS = {"set_preparation_plan": SetPreparationPlan}
