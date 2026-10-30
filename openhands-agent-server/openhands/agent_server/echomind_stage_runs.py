from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from openhands.agent_server.echomind_projects import (
    ProjectInfo,
    ProjectNode,
    ProjectNodeType,
)
from openhands.agent_server.echomind_stage_executors import (
    StageArtifactPayload,
    StageExecutionResult,
    get_project_node_executor,
)


StageRunStatus = Literal["running", "succeeded", "failed"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _canonical_id(value: str, *, kind: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise FileNotFoundError(f"{kind} not found: {value}") from exc


class ProjectArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_id: str
    project_id: str
    node_id: str
    run_id: str
    kind: str
    format: Literal["json"] = "json"
    uri: str
    content_sha256: str
    size_bytes: int = Field(ge=0)
    created_at: datetime = Field(default_factory=_utc_now)


class StageRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: str
    project_id: str
    node_id: str
    node_type: ProjectNodeType
    executor: str
    status: StageRunStatus
    message: str
    input_refs: list[str] = Field(default_factory=list)
    artifacts: list[ProjectArtifact] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ProjectArtifactDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact: ProjectArtifact
    content: dict[str, Any]


class ProjectRunStore:
    def __init__(self, projects_root: Path | str) -> None:
        self.projects_root = Path(projects_root)

    def start(
        self,
        *,
        project: ProjectInfo,
        node: ProjectNode,
    ) -> StageRun:
        now = _utc_now()
        run = StageRun(
            run_id=str(uuid4()),
            project_id=project.project.project_id,
            node_id=node.id,
            node_type=node.type,
            executor=node.executor,
            status="running",
            message=f"Starting {node.title}.",
            created_at=now,
            started_at=now,
        )
        self._write_run(run)
        return run

    def succeed(
        self,
        run: StageRun,
        result: StageExecutionResult,
    ) -> StageRun:
        artifacts: list[ProjectArtifact] = []
        written_paths: list[Path] = []
        try:
            for payload in result.artifacts:
                artifact, path = self._write_artifact(run, payload)
                artifacts.append(artifact)
                written_paths.append(path)
            completed = run.model_copy(
                update={
                    "status": "succeeded",
                    "message": result.message,
                    "artifacts": artifacts,
                    "finished_at": _utc_now(),
                }
            )
            self._write_run(completed)
        except Exception:
            for path in written_paths:
                path.unlink(missing_ok=True)
            raise
        return completed

    def fail(self, run: StageRun, error: Exception) -> StageRun:
        failed = run.model_copy(
            update={
                "status": "failed",
                "message": f"{run.node_id} failed.",
                "error": f"{type(error).__name__}: {error}"[:4000],
                "finished_at": _utc_now(),
            }
        )
        self._write_run(failed)
        return failed

    def list(self, project_id: str) -> list[StageRun]:
        runs_dir = self._project_dir(project_id) / "runs"
        if not runs_dir.is_dir():
            return []
        runs = [
            StageRun.model_validate_json(path.read_text(encoding="utf-8"))
            for path in runs_dir.glob("*.json")
        ]
        return sorted(
            runs,
            key=lambda run: (run.created_at, run.run_id),
            reverse=True,
        )

    def load_artifact(
        self,
        *,
        project_id: str,
        artifact_id: str,
    ) -> ProjectArtifactDocument:
        canonical_artifact_id = _canonical_id(artifact_id, kind="artifact")
        path = (
            self._project_dir(project_id)
            / "artifacts"
            / f"{canonical_artifact_id}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"artifact not found: {artifact_id}")
        document = ProjectArtifactDocument.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            _canonical_id(document.artifact.project_id, kind="project")
            != _canonical_id(project_id, kind="project")
            or document.artifact.artifact_id != canonical_artifact_id
        ):
            raise ValueError("artifact metadata does not match its storage path")
        return document

    def _write_artifact(
        self,
        run: StageRun,
        payload: StageArtifactPayload,
    ) -> tuple[ProjectArtifact, Path]:
        artifact_id = str(uuid4())
        content_bytes = json.dumps(
            payload.content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        artifact = ProjectArtifact(
            artifact_id=artifact_id,
            project_id=run.project_id,
            node_id=run.node_id,
            run_id=run.run_id,
            kind=payload.kind,
            format=payload.format,
            uri=f"artifacts/{artifact_id}.json",
            content_sha256=sha256(content_bytes).hexdigest(),
            size_bytes=len(content_bytes),
        )
        path = self._project_dir(run.project_id) / "artifacts" / f"{artifact_id}.json"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_json(
            path,
            ProjectArtifactDocument(
                artifact=artifact,
                content=payload.content,
            ).model_dump(mode="json"),
        )
        return artifact, path

    def _write_run(self, run: StageRun) -> None:
        run_dir = self._project_dir(run.project_id) / "runs"
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_json(run_dir / f"{run.run_id}.json", run.model_dump(mode="json"))

    def _project_dir(self, project_id: str) -> Path:
        canonical_project_id = _canonical_id(project_id, kind="project")
        return self.projects_root / canonical_project_id


def run_project_node(
    *,
    project: ProjectInfo,
    node: ProjectNode,
    run_store: ProjectRunStore,
) -> StageRun:
    executor = get_project_node_executor(node.executor)
    run = run_store.start(project=project, node=node)
    try:
        result = executor.run(project=project, node=node)
        return run_store.succeed(run, result)
    except Exception as exc:
        return run_store.fail(run, exc)


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
