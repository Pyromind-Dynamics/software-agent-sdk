from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openhands.agent_server.echomind_graph_runs import ArtifactRef, TaskRun
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


StageRunStatus = Literal[
    "pending",
    "clarifying",
    "running",
    "waiting_user",
    "waiting_external",
    "paused",
    "succeeded",
    "failed",
    "canceled",
]
StageRunMode = Literal["node_validation", "agent_development", "published_run"]


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
    graph_run_id: str | None = None
    project_id: str
    node_id: str
    node_type: ProjectNodeType
    mode: StageRunMode = "node_validation"
    attempt: int = Field(default=1, ge=1)
    executor: str
    executor_kind: str | None = None
    task_type: str | None = None
    implementation_ref: str | None = None
    implementation_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    status: StageRunStatus
    message: str
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[ArtifactRef] = Field(default_factory=list)
    task_runs: list[TaskRun] = Field(default_factory=list)
    artifacts: list[ProjectArtifact] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    reused_from_stage_run_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_run_state(self) -> StageRun:
        if self.reused_from_stage_run_id is not None and (
            self.status != "succeeded" or not self.graph_run_id or self.task_runs
        ):
            raise ValueError("A reused StageRun must be a successful graph stage")
        if self.mode == "published_run" and not self.graph_run_id:
            raise ValueError("published StageRun requires a graph_run_id")
        if any(task_run.stage_run_id != self.run_id for task_run in self.task_runs):
            raise ValueError("StageRun task_runs must belong to the StageRun")
        if self.status in {"running", "waiting_external", "paused"}:
            if self.started_at is None:
                raise ValueError(f"{self.status} StageRun requires started_at")
        if self.status in {"succeeded", "failed", "canceled"}:
            if self.finished_at is None:
                raise ValueError(f"{self.status} StageRun requires finished_at")
        elif self.finished_at is not None:
            raise ValueError("non-terminal StageRun must not have finished_at")
        if self.status == "failed" and not self.error:
            raise ValueError("failed StageRun requires an error")
        return self


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
        graph_run_id: str | None = None,
        mode: StageRunMode = "node_validation",
        attempt: int = 1,
        input_refs: list[str] | None = None,
    ) -> StageRun:
        now = _utc_now()
        run = StageRun(
            run_id=str(uuid4()),
            graph_run_id=graph_run_id,
            project_id=project.project.project_id,
            node_id=node.id,
            node_type=node.type,
            mode=mode,
            attempt=attempt,
            executor=node.executor,
            executor_kind=node.executor,
            task_type=node.task_type,
            implementation_ref=node.implementation_ref,
            implementation_digest=node.implementation_digest,
            status="running",
            message=f"Starting {node.title}.",
            input_refs=input_refs or [],
            created_at=now,
            started_at=now,
        )
        self._write_run(run)
        return run

    def save(self, run: StageRun) -> StageRun:
        self._write_run(run)
        return run

    def discard_reused_stage(self, run: StageRun) -> None:
        if run.reused_from_stage_run_id is None:
            raise ValueError("StageRun has no reused outputs to discard")
        run_id = _canonical_id(run.run_id, kind="StageRun")
        path = self._project_dir(run.project_id) / "runs" / f"{run_id}.json"
        path.unlink(missing_ok=True)

    def load(self, *, project_id: str, run_id: str) -> StageRun:
        canonical_run_id = _canonical_id(run_id, kind="StageRun")
        path = self._project_dir(project_id) / "runs" / f"{canonical_run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"StageRun not found: {run_id}")
        run = StageRun.model_validate_json(path.read_text(encoding="utf-8"))
        if _canonical_id(run.project_id, kind="project") != _canonical_id(
            project_id, kind="project"
        ):
            raise ValueError("StageRun belongs to another project")
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
                artifact, path = self.write_artifact(run, payload)
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

    def list_for_graph_run(
        self,
        *,
        project_id: str,
        graph_run_id: str,
    ) -> list[StageRun]:
        return sorted(
            (run for run in self.list(project_id) if run.graph_run_id == graph_run_id),
            key=lambda run: (run.created_at, run.run_id),
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

    def write_artifact(
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
