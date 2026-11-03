from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from pyromind_runtime.domain.base import ContractModel


type ProjectNodeType = Literal["dataset", "inference", "evaluator", "train"]
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


class ProjectNode(ContractModel):
    id: str = Field(min_length=1, max_length=128)
    type: ProjectNodeType
    title: str = Field(min_length=1, max_length=256)
    status: ProjectNodeStatus = "idle"
    launch_mode: ProjectNodeLaunchMode = "executor"
    executor: str = Field(default="contract_probe", min_length=1, max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)


class ProjectEdge(ContractModel):
    source: str = Field(min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=128)


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
    type: ProjectNodeType | None = None
    launch_mode: ProjectNodeLaunchMode | None = None
    executor: str | None = Field(default=None, min_length=1, max_length=128)
    config: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_change(self) -> UpdateProjectNode:
        editable = {"title", "type", "launch_mode", "executor", "config"}
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
