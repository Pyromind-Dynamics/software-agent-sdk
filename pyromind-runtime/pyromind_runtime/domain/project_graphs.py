from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from pyromind_runtime.domain.base import ContractModel


type ProjectLifecycle = Literal["development", "frozen"]
type ProjectNodeLaunchMode = Literal["agent", "executor"]
type RequirementStatus = Literal["confirmed", "missing"]
type ProjectNodeStatus = Literal[
    "idle",
    "running",
    "succeeded",
    "failed",
    "skipped",
    "not_implemented",
]


class ProjectRequirement(ContractModel):
    key: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=256)
    value: str = Field(max_length=4000)
    status: RequirementStatus


class ProjectPortBinding(ContractModel):
    kind: Literal["run_input", "node_output", "literal"]
    name: str | None = Field(default=None, min_length=1, max_length=128)
    node_id: str | None = Field(default=None, min_length=1, max_length=128)
    port: str | None = Field(default=None, min_length=1, max_length=128)
    value: Any = None

    @model_validator(mode="after")
    def validate_binding(self) -> ProjectPortBinding:
        if self.kind == "run_input":
            if not self.name or self.node_id or self.port or self.value is not None:
                raise ValueError("run_input binding requires only name")
        elif self.kind == "node_output":
            if not self.node_id or not self.port or self.name or self.value is not None:
                raise ValueError("node_output binding requires node_id and port")
        elif self.name or self.node_id or self.port:
            raise ValueError("literal binding does not accept name, node_id or port")
        return self


class ProjectInputPort(ContractModel):
    artifact_type: str = Field(min_length=1, max_length=256)
    schema_ref: str | None = Field(default=None, max_length=256)
    required: bool = True
    binding: ProjectPortBinding | None = None


class ProjectOutputPort(ContractModel):
    artifact_type: str = Field(min_length=1, max_length=256)
    schema_ref: str | None = Field(default=None, max_length=256)
    required: bool = True


class ProjectExecutorRuntime(ContractModel):
    image: str | None = None
    entrypoint: str | None = None


class ProjectExecutorBinding(ContractModel):
    kind: str = Field(min_length=1, max_length=128)
    implementation_ref: str | None = None
    implementation_digest: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    runtime: ProjectExecutorRuntime = Field(default_factory=ProjectExecutorRuntime)


class ProjectNode(ContractModel):
    id: str = Field(min_length=1, max_length=128)
    type: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=256)
    task_type: str = Field(min_length=1, max_length=256)
    status: ProjectNodeStatus = "idle"
    launch_mode: ProjectNodeLaunchMode = "executor"
    execution_role: str = Field(default="", max_length=128)
    inputs: dict[str, ProjectInputPort] = Field(default_factory=dict)
    outputs: dict[str, ProjectOutputPort] = Field(default_factory=dict)
    executor: ProjectExecutorBinding
    config: dict[str, Any] = Field(default_factory=dict)
    policies: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_node(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        node_type = payload.get("type")
        task_type = payload.get("task_type")
        if not task_type and isinstance(node_type, str) and node_type:
            payload["task_type"] = _legacy_task_type(node_type)
        if not payload.get("execution_role") and isinstance(node_type, str):
            config = payload.get("config")
            configured_role = (
                config.get("execution_role") if isinstance(config, dict) else None
            )
            payload["execution_role"] = (
                configured_role
                if isinstance(configured_role, str) and configured_role
                else _default_execution_role(node_type)
            )
        executor = payload.get("executor")
        if executor is None:
            payload["executor"] = {"kind": "contract_probe"}
        elif isinstance(executor, str):
            payload["executor"] = {"kind": executor}
        return payload

    @property
    def executor_kind(self) -> str:
        return self.executor.kind


class ProjectEdge(ContractModel):
    source: str = Field(min_length=1, max_length=128)
    source_port: str | None = Field(default=None, min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=128)
    target_port: str | None = Field(default=None, min_length=1, max_length=128)


class ProjectGraph(ContractModel):
    schema_version: Literal[1] = 1
    graph_id: str
    project_id: str
    template: str | None = None
    lifecycle: ProjectLifecycle = "development"
    nodes: list[ProjectNode] = Field(default_factory=list)
    edges: list[ProjectEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_topology(self) -> ProjectGraph:
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("graph node IDs must be unique")

        adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        edge_keys: set[tuple[str, str]] = set()
        for edge in self.edges:
            key = (edge.source, edge.target)
            if key in edge_keys:
                raise ValueError("graph edges must be unique")
            edge_keys.add(key)
            if edge.source == edge.target:
                raise ValueError("graph must not contain self edges")
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError("graph edge references an unknown node")
            adjacency[edge.source].append(edge.target)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("graph must not contain cycles")
            if node_id in visited:
                return
            visiting.add(node_id)
            for target in adjacency[node_id]:
                visit(target)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in node_ids:
            visit(node_id)
        return self


class ProjectGraphView(ContractModel):
    schema_version: Literal[1] = 1
    project_id: str
    graph_id: str
    name: str
    description: str
    lifecycle: ProjectLifecycle
    template: str | None = None
    requirements: list[ProjectRequirement] = Field(default_factory=list)
    nodes: list[ProjectNode] = Field(default_factory=list)
    edges: list[ProjectEdge] = Field(default_factory=list)


class AddProjectNode(ContractModel):
    op: Literal["add_node"] = "add_node"
    node: ProjectNode


class UpdateProjectNode(ContractModel):
    op: Literal["update_node"] = "update_node"
    node_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, min_length=1, max_length=256)
    type: str | None = Field(default=None, min_length=1, max_length=128)
    task_type: str | None = Field(default=None, min_length=1, max_length=256)
    launch_mode: ProjectNodeLaunchMode | None = None
    execution_role: str | None = Field(default=None, min_length=1, max_length=128)
    inputs: dict[str, ProjectInputPort] | None = None
    outputs: dict[str, ProjectOutputPort] | None = None
    executor: ProjectExecutorBinding | None = None
    config: dict[str, Any] | None = None
    policies: dict[str, Any] | None = None

    @field_validator("executor", mode="before")
    @classmethod
    def migrate_legacy_executor(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"kind": value}
        return value

    @model_validator(mode="after")
    def require_change(self) -> UpdateProjectNode:
        editable = {
            "title",
            "type",
            "task_type",
            "launch_mode",
            "execution_role",
            "inputs",
            "outputs",
            "executor",
            "config",
            "policies",
        }
        supplied = editable.intersection(self.model_fields_set)
        if not supplied:
            raise ValueError("update_node requires at least one changed field")
        if any(getattr(self, field) is None for field in supplied):
            raise ValueError("update_node fields cannot be null")
        return self


class RemoveProjectNode(ContractModel):
    op: Literal["remove_node"] = "remove_node"
    node_id: str = Field(min_length=1, max_length=128)


class AddProjectEdge(ContractModel):
    op: Literal["add_edge"] = "add_edge"
    edge: ProjectEdge


class RemoveProjectEdge(ContractModel):
    op: Literal["remove_edge"] = "remove_edge"
    edge: ProjectEdge


class UpdateProjectRequirements(ContractModel):
    op: Literal["update_requirements"] = "update_requirements"
    requirements: list[ProjectRequirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_keys(self) -> UpdateProjectRequirements:
        keys = [item.key for item in self.requirements]
        if len(keys) != len(set(keys)):
            raise ValueError("requirement keys must be unique")
        return self


type ProjectGraphOperation = Annotated[
    AddProjectNode
    | UpdateProjectNode
    | RemoveProjectNode
    | AddProjectEdge
    | RemoveProjectEdge
    | UpdateProjectRequirements,
    Field(discriminator="op"),
]


class GraphPatch(ContractModel):
    request_id: str = Field(min_length=1, max_length=128)
    operations: tuple[ProjectGraphOperation, ...] = Field(min_length=1)


class ApplyProjectGraphPatch(GraphPatch):
    project_id: str = Field(min_length=1, max_length=128)


class GetProjectGraph(ContractModel):
    project_id: str = Field(min_length=1, max_length=128)


PROJECT_GRAPH_TOOLS = {
    "get_project_graph": GetProjectGraph,
    "apply_graph_patch": ApplyProjectGraphPatch,
}


def _legacy_task_type(node_type: str) -> str:
    if node_type == "dataset":
        return "dataset.prepare"
    return f"{node_type}.run"


def _default_execution_role(node_type: str) -> str:
    return {
        "dataset": "dataset",
        "train": "train",
        "inference": "inference",
        "evaluator": "eval",
    }.get(node_type, "")


def canonical_execution_role(role_id: str) -> str:
    """Keep the historical evaluator alias pointed at the single eval role."""
    return "eval" if role_id == "evaluator" else role_id
