from __future__ import annotations

from typing import Literal

from pydantic import (
    Field,
    GetJsonSchemaHandler,
    JsonValue,
    field_validator,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

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
    required_fields: tuple[str, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
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
            if not set(node.required_fields) <= {field.key for field in self.fields}:
                raise ValueError("workflow node references unknown checklist fields")
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

    @model_validator(mode="after")
    def validate_ready_fields(self):
        if self.phase == "ready" and self.plan is not None:
            nodes = {node.id: node for node in self.policy.workflow_nodes}
            required = {
                field
                for stage in self.plan.stages
                if stage.enabled and not stage.placeholder and stage.kind in nodes
                for field in nodes[stage.kind].required_fields
            }
            missing = sorted(
                field
                for field in required
                if self.facts.get(field, PreparationFact()).status != "resolved"
            )
            if missing:
                raise ValueError(
                    "workflow requires checklist fields: " + ", ".join(missing)
                )
        return self


class UpdatePreparation(ContractModel):
    expected_revision: int = Field(ge=0)
    facts: dict[str, PreparationFact]

    @field_validator("facts", mode="before")
    @classmethod
    def resolve_supplied_values(cls, facts: object) -> object:
        if not isinstance(facts, dict):
            return facts
        return {
            key: {**fact, "status": "resolved"}
            if isinstance(fact, dict)
            and "status" not in fact
            and fact.get("value") is not None
            else fact
            for key, fact in facts.items()
        }

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        facts = schema["properties"]["facts"]
        fact = handler.resolve_ref_schema(facts["additionalProperties"])
        status = fact["properties"]["status"]
        status.pop("default", None)
        status["description"] = "If omitted, a non-null value is resolved."
        fact["properties"]["value"] = {
            "description": "Checklist value in its declared type, or null if unknown.",
            "anyOf": [
                {"type": kind}
                for kind in ("string", "object", "array", "integer", "boolean", "null")
            ],
        }
        _require_tool_fields(fact, "value")
        return schema


def _require_tool_fields(schema: JsonSchemaValue, *names: str) -> None:
    required = schema.setdefault("required", [])
    for name in names:
        if name not in required:
            required.append(name)
        schema["properties"][name].pop("default", None)


class SetPreparationPlan(ContractModel):
    expected_revision: int = Field(ge=0)
    plan: ExperimentContext
    entry_task_id: str = Field(
        min_length=1, description="ID from plan.tasks of the task to run first"
    )
    role_id: str = Field(
        min_length=1,
        description="Enabled delegated role from preparation.policy.allowed_roles",
    )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        schema = handler(core_schema)
        plan = handler.resolve_ref_schema(schema["properties"]["plan"])
        _require_tool_fields(plan, "stages", "tasks")
        for name in ("stages", "tasks"):
            collection = plan["properties"][name]
            collection["minItems"] = 1
            item = handler.resolve_ref_schema(collection["items"])
            _require_tool_fields(item, "kind")
            if name == "tasks":
                item["properties"]["stage_id"]["description"] = (
                    "ID of this task's stage from plan.stages[].id, not the stage kind."
                )
        return schema


class InspectResource(ContractModel):
    compute_digest: bool = Field(default=False, exclude_if=lambda value: not value)
    path: str = Field(min_length=1, max_length=4096)
    kind: Literal["file", "directory"] = "file"
    max_bytes: int = Field(default=16384, ge=1, le=65536)


class ProjectView(ContractModel):
    project_id: str
    conversation_id: str
    plan_sequence: int
    plan: ExperimentContext
    creation_command_id: str
    initial_run_id: str | None = None
    cluster: str | None = None


PREPARATION_TOOLS = {
    "get_preparation": ContractModel,
    "update_preparation": UpdatePreparation,
    "set_preparation_plan": SetPreparationPlan,
    "inspect_resource": InspectResource,
}
