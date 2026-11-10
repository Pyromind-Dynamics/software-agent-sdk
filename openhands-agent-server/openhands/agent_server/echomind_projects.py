from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pyromind_runtime.domain.experiment_plans import (
    ExperimentContext,
    PlanRevision,
    WorkflowPlan,
)
from pyromind_runtime.domain.preparation import ProjectView

from openhands.agent_server.echomind_graph_templates import (
    EXPERIMENT_GRAPH_TEMPLATE,
    EXPERIMENT_PLANNING_SKILL,
    build_graph_payload,
)


ProjectNodeType = Literal["dataset", "inference", "evaluator", "train"]
ProjectLifecycle = Literal["development", "frozen"]
ProjectNodeLaunchMode = Literal["agent", "executor"]
RequirementStatus = Literal["confirmed", "missing"]
ProjectNodeStatus = Literal[
    "idle",
    "running",
    "succeeded",
    "failed",
    "skipped",
    "not_implemented",
]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ProjectRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=256)
    value: str = Field(max_length=4000)
    status: RequirementStatus


class ProjectNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=128)
    type: ProjectNodeType
    title: str = Field(min_length=1, max_length=256)
    status: ProjectNodeStatus = "idle"
    launch_mode: ProjectNodeLaunchMode = "executor"
    executor: str = Field(default="contract_probe", min_length=1, max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)


class ProjectEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=128)


class ProjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    project_id: str
    name: str
    description: str
    skill: str
    requirements: list[ProjectRequirement] = Field(default_factory=list)
    owner_user_id: str | None = None
    conversation_id: str
    cluster: str
    lifecycle: ProjectLifecycle = "development"
    frozen_at: datetime | None = None
    created_at: datetime = Field(default_factory=_utc_now)


class ProjectGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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
        for edge in self.edges:
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


class ProjectInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project: ProjectRecord
    graph: ProjectGraph

    @model_validator(mode="after")
    def require_matching_lifecycle(self) -> ProjectInfo:
        if self.project.lifecycle != self.graph.lifecycle:
            raise ValueError("project and graph lifecycle must match")
        if self.project.project_id != self.graph.project_id:
            raise ValueError("project graph belongs to another project")
        return self


class ProjectSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    name: str
    description: str
    skill: str
    owner_user_id: str | None = None
    conversation_id: str
    cluster: str
    lifecycle: ProjectLifecycle
    node_count: int
    created_at: datetime
    updated_at: datetime


class ProjectNotFoundError(FileNotFoundError):
    pass


class ProjectClusterMismatchError(PermissionError):
    pass


class ProjectStore:
    """File-backed Project store whose identity comes from an ExperimentPlan."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def for_conversations_dir(cls, conversations_dir: Path | str) -> ProjectStore:
        return cls(Path(conversations_dir).parent / "projects")

    def bind_plan(
        self,
        revision: PlanRevision,
        *,
        conversation_id: str,
        cluster: str,
        owner_user_id: str | None,
    ) -> ProjectInfo:
        return self._bind(
            project_id=revision.project_id,
            plan=revision.plan,
            name=revision.plan.name,
            description=revision.plan.goal,
            skill=revision.plan.skill or EXPERIMENT_PLANNING_SKILL,
            conversation_id=conversation_id,
            cluster=cluster,
            owner_user_id=owner_user_id,
        )

    def bind_preparation(
        self,
        project: ProjectView,
        *,
        cluster: str,
        owner_user_id: str | None,
    ) -> ProjectInfo:
        return self._bind(
            project_id=project.project_id,
            plan=project.plan,
            name=project.plan.goal,
            description=project.plan.goal,
            skill=EXPERIMENT_PLANNING_SKILL,
            conversation_id=project.conversation_id,
            cluster=cluster,
            owner_user_id=owner_user_id,
        )

    def _bind(
        self,
        *,
        project_id: str,
        plan: ExperimentContext | WorkflowPlan,
        name: str,
        description: str,
        skill: str,
        conversation_id: str,
        cluster: str,
        owner_user_id: str | None,
    ) -> ProjectInfo:
        try:
            canonical_project_id = str(UUID(project_id))
        except ValueError as exc:
            raise ValueError("plan revision has an invalid project_id") from exc

        final_dir = self.root / canonical_project_id
        existing = self._load_optional(canonical_project_id)
        if existing is not None:
            if (
                existing.project.owner_user_id != owner_user_id
                or existing.project.conversation_id != conversation_id
                or existing.project.cluster != cluster
            ):
                raise PermissionError("Project belongs to another owner")
            graph_id = existing.graph.graph_id
            created_at = existing.project.created_at
        else:
            graph_id = str(uuid4())
            created_at = _utc_now()

        node_payloads, edge_payloads = build_graph_payload(plan=plan)
        project = ProjectRecord(
            project_id=project_id,
            name=name,
            description=description,
            skill=skill,
            requirements=_requirements_from_plan(plan),
            owner_user_id=owner_user_id,
            conversation_id=conversation_id,
            cluster=cluster,
            lifecycle="development",
            created_at=created_at,
        )
        graph = ProjectGraph(
            graph_id=graph_id,
            project_id=project_id,
            template=EXPERIMENT_GRAPH_TEMPLATE,
            lifecycle="development",
            nodes=[
                ProjectNode.model_validate(node_payload)
                for node_payload in node_payloads
            ],
            edges=[
                ProjectEdge.model_validate(edge_payload)
                for edge_payload in edge_payloads
            ],
        )

        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        final_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_json(final_dir / "project.json", project.model_dump(mode="json"))
        _write_json(final_dir / "graph.json", graph.model_dump(mode="json"))
        return ProjectInfo(project=project, graph=graph)

    def load(
        self,
        project_id: str,
        *,
        owner_user_id: str | None,
    ) -> ProjectInfo:
        try:
            canonical_project_id = str(UUID(project_id))
        except ValueError as exc:
            raise ProjectNotFoundError(project_id) from exc
        project = self._load_optional(canonical_project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        if owner_user_id is not None and project.project.owner_user_id != owner_user_id:
            raise ProjectNotFoundError(project_id)
        return project

    def list(self, *, owner_user_id: str | None) -> list[ProjectSummary]:
        if not self.root.is_dir():
            return []
        summaries: list[ProjectSummary] = []
        for project_dir in self.root.iterdir():
            if not project_dir.is_dir():
                continue
            try:
                project = self._load_optional(project_dir.name)
            except (OSError, ValidationError, ValueError):
                continue
            if project is None:
                continue
            if (
                owner_user_id is not None
                and project.project.owner_user_id != owner_user_id
            ):
                continue
            project_path = project_dir / "project.json"
            graph_path = project_dir / "graph.json"
            created_at = project.project.created_at
            updated_at = max(
                created_at,
                datetime.fromtimestamp(project_path.stat().st_mtime, UTC),
                datetime.fromtimestamp(graph_path.stat().st_mtime, UTC),
            )
            summaries.append(
                ProjectSummary(
                    project_id=project.project.project_id,
                    name=project.project.name,
                    description=project.project.description,
                    skill=project.project.skill,
                    owner_user_id=project.project.owner_user_id,
                    conversation_id=project.project.conversation_id,
                    cluster=project.project.cluster,
                    lifecycle=project.project.lifecycle,
                    node_count=len(project.graph.nodes),
                    created_at=created_at,
                    updated_at=updated_at,
                )
            )
        return sorted(summaries, key=lambda item: item.updated_at, reverse=True)

    def require_cluster(self, project: ProjectInfo, cluster: str | None) -> None:
        requested = (cluster or "").strip()
        if not requested:
            raise ProjectClusterMismatchError(
                "x-cluster is required for Project writes"
            )
        if project.project.cluster != requested:
            raise ProjectClusterMismatchError(
                f"Project belongs to cluster {project.project.cluster}"
            )

    def _load_optional(self, canonical_project_id: str) -> ProjectInfo | None:
        project_dir = self.root / canonical_project_id
        project_path = project_dir / "project.json"
        graph_path = project_dir / "graph.json"
        if not project_path.is_file() or not graph_path.is_file():
            return None
        project = ProjectRecord.model_validate_json(
            project_path.read_text(encoding="utf-8")
        )
        graph = ProjectGraph.model_validate_json(graph_path.read_text(encoding="utf-8"))
        if graph.project_id != project.project_id:
            raise ValueError("project graph belongs to another project")
        return ProjectInfo(project=project, graph=graph)


def _requirements_from_plan(
    plan: ExperimentContext | WorkflowPlan,
) -> list[ProjectRequirement]:
    return [
        ProjectRequirement(
            key=resource.id,
            label=resource.name,
            value=resource.storage_path or "",
            status="confirmed" if resource.storage_path else "missing",
        )
        for resource in plan.resources
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
