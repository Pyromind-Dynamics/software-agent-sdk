from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyromind_runtime.domain.experiment_plans import PlanRevision, WorkflowPlan
from pyromind_runtime.domain.project_graphs import (
    AddProjectEdge,
    AddProjectNode,
    GraphPatch,
    ProjectEdge,
    ProjectGraph,
    ProjectGraphView,
    ProjectLifecycle,
    ProjectNode,
    ProjectRequirement,
    RemoveProjectEdge,
    RemoveProjectNode,
    UpdateProjectNode,
    UpdateProjectRequirements,
)

from openhands.agent_server.echomind_graph_templates import (
    EXPERIMENT_GRAPH_TEMPLATE,
    EXPERIMENT_PLANNING_SKILL,
    build_graph_payload,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ProjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    project_id: str
    name: str
    description: str
    skill: str
    requirements: list[ProjectRequirement] = Field(default_factory=list)
    owner_user_id: str | None = None
    echomind_conversation_id: str
    lifecycle: ProjectLifecycle = "development"
    frozen_at: datetime | None = None
    created_at: datetime = Field(default_factory=_utc_now)


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


class ProjectNotFoundError(FileNotFoundError):
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
        echomind_conversation_id: str,
        owner_user_id: str | None,
    ) -> ProjectInfo:
        plan = revision.plan
        project_id = revision.project_id
        try:
            canonical_project_id = str(UUID(project_id))
        except ValueError as exc:
            raise ValueError("plan revision has an invalid project_id") from exc

        final_dir = self.root / canonical_project_id
        existing = self._load_optional(canonical_project_id)
        if existing is not None:
            if (
                existing.project.owner_user_id not in {None, owner_user_id}
                or existing.project.echomind_conversation_id != echomind_conversation_id
            ):
                raise PermissionError("Project belongs to another conversation")
            graph_id = existing.graph.graph_id
            created_at = existing.project.created_at
        else:
            graph_id = str(uuid4())
            created_at = _utc_now()

        skill = plan.skill or EXPERIMENT_PLANNING_SKILL
        node_payloads, edge_payloads = build_graph_payload(plan=plan)
        project = ProjectRecord(
            project_id=project_id,
            name=plan.name,
            description=plan.goal,
            skill=skill,
            requirements=_requirements_from_plan(plan),
            owner_user_id=owner_user_id,
            echomind_conversation_id=echomind_conversation_id,
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
        if (
            owner_user_id is not None
            and project.project.owner_user_id is not None
            and project.project.owner_user_id != owner_user_id
        ):
            raise ProjectNotFoundError(project_id)
        return project

    def get_graph(
        self,
        project_id: str,
        *,
        owner_user_id: str | None,
    ) -> ProjectGraphView:
        return _graph_view(self.load(project_id, owner_user_id=owner_user_id))

    def apply_graph_patch(
        self,
        project_id: str,
        *,
        owner_user_id: str | None,
        patch: GraphPatch,
    ) -> ProjectGraphView:
        try:
            canonical_project_id = str(UUID(project_id))
        except ValueError as exc:
            raise ProjectNotFoundError(project_id) from exc
        project_dir = self.root / canonical_project_id
        if not project_dir.is_dir():
            raise ProjectNotFoundError(project_id)
        lock = FileLock(str(project_dir / ".graph-mutation.lock"))
        with lock:
            current = self.load(
                canonical_project_id,
                owner_user_id=owner_user_id,
            )
            if current.project.lifecycle != "development":
                raise ValueError("frozen Projects cannot be modified")

            ledger_path = project_dir / "graph-mutations.json"
            ledger = _read_json_object(ledger_path)
            request_digest = _graph_patch_digest(patch)
            previous_digest = ledger.get(patch.request_id)
            if previous_digest is not None:
                if previous_digest != request_digest:
                    raise ValueError(
                        "request_id was already used for another graph patch"
                    )
                return _graph_view(current)

            nodes = list(current.graph.nodes)
            edges = list(current.graph.edges)
            requirements = list(current.project.requirements)
            for operation in patch.operations:
                nodes, edges, requirements = _apply_operation(
                    operation,
                    nodes=nodes,
                    edges=edges,
                    requirements=requirements,
                )

            graph = ProjectGraph.model_validate(
                {
                    **current.graph.model_dump(mode="python"),
                    "nodes": nodes,
                    "edges": edges,
                }
            )
            project = current.project.model_copy(update={"requirements": requirements})
            updated = ProjectInfo(project=project, graph=graph)

            if graph != current.graph:
                _write_json(project_dir / "graph.json", graph.model_dump(mode="json"))
            if project != current.project:
                _write_json(
                    project_dir / "project.json",
                    project.model_dump(mode="json"),
                )
            ledger[patch.request_id] = request_digest
            _write_json(ledger_path, ledger)
            return _graph_view(updated)

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


def _requirements_from_plan(plan: WorkflowPlan) -> list[ProjectRequirement]:
    return [
        ProjectRequirement(
            key=resource.id,
            label=resource.name,
            value=resource.uri or "",
            status="confirmed" if resource.uri else "missing",
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


def _graph_view(info: ProjectInfo) -> ProjectGraphView:
    return ProjectGraphView(
        project_id=info.project.project_id,
        graph_id=info.graph.graph_id,
        name=info.project.name,
        description=info.project.description,
        lifecycle=info.graph.lifecycle,
        template=info.graph.template,
        requirements=list(info.project.requirements),
        nodes=list(info.graph.nodes),
        edges=list(info.graph.edges),
    )


def _apply_operation(
    operation: Any,
    *,
    nodes: list[ProjectNode],
    edges: list[ProjectEdge],
    requirements: list[ProjectRequirement],
) -> tuple[list[ProjectNode], list[ProjectEdge], list[ProjectRequirement]]:
    nodes = list(nodes)
    edges = list(edges)
    requirements = list(requirements)

    if isinstance(operation, AddProjectNode):
        nodes.append(operation.node)
    elif isinstance(operation, UpdateProjectNode):
        index = _node_index(nodes, operation.node_id)
        changes = operation.model_dump(exclude_unset=True, exclude={"op", "node_id"})
        payload = {
            **nodes[index].model_dump(mode="python"),
            **changes,
            "status": "idle",
        }
        nodes[index] = ProjectNode.model_validate(payload)
    elif isinstance(operation, RemoveProjectNode):
        _node_index(nodes, operation.node_id)
        nodes = [node for node in nodes if node.id != operation.node_id]
        edges = [
            edge
            for edge in edges
            if edge.source != operation.node_id and edge.target != operation.node_id
        ]
    elif isinstance(operation, AddProjectEdge):
        edges.append(operation.edge)
    elif isinstance(operation, RemoveProjectEdge):
        if operation.edge not in edges:
            raise ValueError("graph edge does not exist")
        edges.remove(operation.edge)
    elif isinstance(operation, UpdateProjectRequirements):
        requirements = list(operation.requirements)
    else:
        raise ValueError("unsupported graph patch operation")
    return nodes, edges, requirements


def _node_index(nodes: list[ProjectNode], node_id: str) -> int:
    for index, node in enumerate(nodes):
        if node.id == node_id:
            return index
    raise ValueError(f"graph node does not exist: {node_id}")


def _graph_patch_digest(patch: GraphPatch) -> str:
    payload = json.dumps(
        patch.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json_object(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in payload.items()
    ):
        raise ValueError(f"invalid graph mutation ledger: {path}")
    return payload
