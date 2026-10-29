from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openhands.agent_server.echomind_graph_templates import (
    AVI_REQUIRED_REQUIREMENT_KEYS,
    AVI_SINGLE_TURN_SKILL,
    build_graph_payload,
)


PROJECT_REQUEST_RELATIVE_PATH = Path("public_data/project_request.json")
_MAX_PROJECT_REQUEST_BYTES = 1024 * 1024

ProjectNodeType = Literal["dataset", "inference", "evaluator", "train"]
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
    executor: str = Field(default="shell", min_length=1, max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)


class ProjectEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=128)


class ProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=8000)
    skill: str = Field(min_length=1, max_length=128)
    requirements: list[ProjectRequirement] = Field(min_length=1)

    @model_validator(mode="after")
    def require_complete_requirements(self) -> ProjectRequest:
        keys = [requirement.key for requirement in self.requirements]
        duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
        if duplicate_keys:
            raise ValueError(
                "project request has duplicate requirements: "
                + ", ".join(duplicate_keys)
            )

        missing = [
            requirement.key
            for requirement in self.requirements
            if requirement.status == "missing"
        ]
        if missing:
            raise ValueError(
                "project request has unresolved requirements: " + ", ".join(missing)
            )

        required_keys = (
            AVI_REQUIRED_REQUIREMENT_KEYS if self.skill == AVI_SINGLE_TURN_SKILL else ()
        )
        confirmed_keys = {
            requirement.key
            for requirement in self.requirements
            if requirement.status == "confirmed" and requirement.value.strip()
        }
        missing_required = sorted(set(required_keys) - confirmed_keys)
        if missing_required:
            raise ValueError(
                "project request is missing required requirements: "
                + ", ".join(missing_required)
            )
        return self


class ProjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    project_id: str
    name: str
    description: str
    skill: str
    requirements: list[ProjectRequirement]
    owner_user_id: str | None = None
    echomind_conversation_id: str
    created_at: datetime = Field(default_factory=_utc_now)


class ProjectGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    graph_id: str
    project_id: str
    template: str | None = None
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


class ProjectRequestNotFoundError(FileNotFoundError):
    pass


class ProjectRequestInvalidError(ValueError):
    pass


class ProjectNotFoundError(FileNotFoundError):
    pass


def read_project_request(conversation_working_dir: Path | str) -> ProjectRequest:
    path = Path(conversation_working_dir) / PROJECT_REQUEST_RELATIVE_PATH
    if not path.is_file():
        raise ProjectRequestNotFoundError(str(PROJECT_REQUEST_RELATIVE_PATH))
    if path.stat().st_size > _MAX_PROJECT_REQUEST_BYTES:
        raise ProjectRequestInvalidError("project request is too large")
    try:
        return ProjectRequest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ProjectRequestInvalidError(str(exc)) from exc


class ProjectStore:
    """File-backed Project store with a Project-owned Graph shell."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def for_conversations_dir(cls, conversations_dir: Path | str) -> ProjectStore:
        return cls(Path(conversations_dir).parent / "projects")

    def create(
        self,
        request: ProjectRequest,
        *,
        echomind_conversation_id: str,
        owner_user_id: str | None,
    ) -> ProjectInfo:
        project_id = str(uuid4())
        graph_id = str(uuid4())
        project = ProjectRecord(
            project_id=project_id,
            name=request.name,
            description=request.description,
            skill=request.skill,
            requirements=request.requirements,
            owner_user_id=owner_user_id,
            echomind_conversation_id=echomind_conversation_id,
        )
        requirement_values = {
            requirement.key: requirement.value for requirement in request.requirements
        }
        node_payloads, edge_payloads = build_graph_payload(
            skill=request.skill,
            requirements=requirement_values,
        )
        graph = ProjectGraph(
            graph_id=graph_id,
            project_id=project_id,
            template=(
                "avi-single-turn/v1" if request.skill == AVI_SINGLE_TURN_SKILL else None
            ),
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
        final_dir = self.root / project_id
        temporary_dir = self.root / f".{project_id}.{uuid4().hex}.tmp"
        temporary_dir.mkdir(mode=0o700)
        try:
            _write_json(temporary_dir / "project.json", project.model_dump(mode="json"))
            _write_json(temporary_dir / "graph.json", graph.model_dump(mode="json"))
            os.replace(temporary_dir, final_dir)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)
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
        project_dir = self.root / canonical_project_id
        project_path = project_dir / "project.json"
        graph_path = project_dir / "graph.json"
        if not project_path.is_file() or not graph_path.is_file():
            raise ProjectNotFoundError(project_id)

        project = ProjectRecord.model_validate_json(
            project_path.read_text(encoding="utf-8")
        )
        graph = ProjectGraph.model_validate_json(graph_path.read_text(encoding="utf-8"))
        if (
            owner_user_id is not None
            and project.owner_user_id is not None
            and project.owner_user_id != owner_user_id
        ):
            raise ProjectNotFoundError(project_id)
        if graph.project_id != project.project_id:
            raise ValueError("project graph belongs to another project")
        return ProjectInfo(project=project, graph=graph)


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
