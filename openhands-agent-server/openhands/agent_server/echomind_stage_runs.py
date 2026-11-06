from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, model_validator

from openhands.agent_server.echomind_dataset_contract import (
    require_dataset_submission_spec,
)
from openhands.agent_server.echomind_projects import ProjectInfo, ProjectNode
from openhands.agent_server.echomind_stage_executors import (
    NodeExecutionContext,
    StageArtifactPayload,
    StageExecutionResult,
    get_project_node_executor,
)


StageRunStatus = Literal[
    "clarifying",
    "running",
    "waiting_user",
    "waiting_external",
    "succeeded",
    "failed",
]
StageRunMode = Literal["node_validation", "agent_development"]
StageRunConfigDigestVersion = Literal[1, 2]
StageReviewOutcome = Literal[
    "waiting_user",
    "waiting_external",
    "succeeded",
    "failed",
]
StageTaskIntent = Literal["execute", "prepare"]


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
    schema_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    uri: str
    content_sha256: str
    size_bytes: int = Field(ge=0)
    created_at: datetime = Field(default_factory=_utc_now)


class StageRunUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    status: StageRunStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    artifact_id: str | None = None
    command_id: str | None = None
    task_run_id: str | None = None
    external_task_id: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)


class StageTaskRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    task_run_id: str
    role_id: str = ""
    intent: StageTaskIntent = "execute"
    status: StageRunStatus = "running"
    message: str = ""
    external_task_id: str | None = None
    output_uri: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime | None = None


class StageRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    mode: StageRunMode = "node_validation"
    request_id: str | None = None
    run_id: str
    graph_run_id: str | None = None
    project_id: str
    node_id: str
    node_type: str
    task_type: str = ""
    attempt: int = Field(default=1, ge=1)
    executor_kind: str = ""
    executor: str
    implementation_ref: str | None = None
    implementation_digest: str | None = None
    node_config_digest: str = ""
    config_digest_version: StageRunConfigDigestVersion = 1
    status: StageRunStatus
    message: str
    command_id: str | None = None
    external_task_id: str | None = None
    task_run_id: str | None = None
    task_runs: list[StageTaskRun] = Field(default_factory=list)
    output_uri: str | None = None
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    artifacts: list[ProjectArtifact] = Field(default_factory=list)
    updates: list[StageRunUpdate] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_run(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        executor = payload.get("executor")
        if not payload.get("executor_kind") and isinstance(executor, str):
            payload["executor_kind"] = executor
        node_type = payload.get("node_type")
        if not payload.get("task_type") and isinstance(node_type, str) and node_type:
            payload["task_type"] = (
                "dataset.prepare" if node_type == "dataset" else f"{node_type}.run"
            )
        task_runs = payload.get("task_runs")
        if not isinstance(task_runs, list):
            task_runs = []
        if not task_runs and isinstance(payload.get("task_run_id"), str):
            status = payload.get("status")
            task_runs = [
                {
                    "request_id": f"legacy:{payload['task_run_id']}",
                    "task_run_id": payload["task_run_id"],
                    "status": status if isinstance(status, str) else "running",
                    "external_task_id": payload.get("external_task_id"),
                    "output_uri": payload.get("output_uri"),
                }
            ]
        payload["task_runs"] = task_runs
        if task_runs:
            latest = task_runs[-1]
            if isinstance(latest, dict):
                payload.setdefault("task_run_id", latest.get("task_run_id"))
                payload.setdefault("external_task_id", latest.get("external_task_id"))
                payload.setdefault("output_uri", latest.get("output_uri"))
        return payload


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
        request_id: str | None,
    ) -> tuple[StageRun, bool]:
        return self._start(
            project=project,
            node=node,
            request_id=request_id,
            mode="node_validation",
        )

    def start_agent_development(
        self,
        *,
        project: ProjectInfo,
        node: ProjectNode,
        request_id: str | None,
    ) -> tuple[StageRun, bool]:
        return self._start(
            project=project,
            node=node,
            request_id=request_id,
            mode="agent_development",
        )

    def _start(
        self,
        *,
        project: ProjectInfo,
        node: ProjectNode,
        request_id: str | None,
        mode: StageRunMode,
    ) -> tuple[StageRun, bool]:
        project_dir = self._project_dir(project.project.project_id)
        project_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = FileLock(str(project_dir / ".run-mutation.lock"))
        with lock:
            existing = self._find_by_request_id(project.project.project_id, request_id)
            if existing is not None:
                existing_config_digest = project_node_config_digest(
                    node,
                    version=existing.config_digest_version,
                )
                if (
                    existing.node_id != node.id
                    or existing.node_config_digest != existing_config_digest
                ):
                    raise ValueError(
                        "request_id was already used for another node validation"
                    )
                return existing, False

            now = _utc_now()
            config_digest_version: StageRunConfigDigestVersion = 2
            config_digest = project_node_config_digest(
                node,
                version=config_digest_version,
            )
            status: StageRunStatus = (
                "clarifying" if mode == "agent_development" else "running"
            )
            run = StageRun(
                mode=mode,
                request_id=request_id,
                run_id=str(uuid4()),
                project_id=project.project.project_id,
                node_id=node.id,
                node_type=node.type,
                task_type=node.task_type,
                executor_kind=node.executor.kind,
                executor=node.executor.kind,
                implementation_ref=node.executor.implementation_ref,
                implementation_digest=node.executor.implementation_digest,
                node_config_digest=config_digest,
                config_digest_version=config_digest_version,
                status=status,
                message=(
                    f"Clarifying the development plan for {node.title}."
                    if mode == "agent_development"
                    else f"Starting node validation for {node.title}."
                ),
                created_at=now,
                started_at=now,
            )
            self._write_run(run)
            return run, True

    def attach_command(
        self,
        run: StageRun,
        *,
        command_id: str,
    ) -> StageRun:
        if run.command_id is not None:
            if run.command_id != command_id:
                raise ValueError("Stage run is already attached to another command")
            return run
        updated = run.model_copy(update={"command_id": command_id})
        self._write_run(updated)
        return updated

    def bind_task_run(
        self,
        run: StageRun,
        *,
        request_id: str,
        task_run_id: str,
        role_id: str = "",
        intent: StageTaskIntent = "execute",
        allow_waiting_user: bool = False,
    ) -> tuple[StageRun, StageTaskRun, bool]:
        project_dir = self._project_dir(run.project_id)
        lock = FileLock(str(project_dir / ".run-mutation.lock"))
        with lock:
            current = self.get(project_id=run.project_id, run_id=run.run_id)
            existing = next(
                (item for item in current.task_runs if item.request_id == request_id),
                None,
            )
            if existing is not None:
                if existing.task_run_id != task_run_id:
                    raise ValueError(
                        "request_id was already attached to another TaskRun"
                    )
                return current, existing, False
            same_task = next(
                (item for item in current.task_runs if item.task_run_id == task_run_id),
                None,
            )
            if same_task is not None:
                raise ValueError("TaskRun is already attached to another request_id")
            reopen_sample = current.status == "waiting_user" and allow_waiting_user
            if current.status not in {"running", "failed"} and not reopen_sample:
                raise ValueError(
                    f"Stage run must be running or failed before binding a TaskRun: "
                    f"{current.status}"
                )

            now = _utc_now()
            task_run = StageTaskRun(
                request_id=request_id,
                task_run_id=task_run_id,
                role_id=role_id,
                intent=intent,
                status="running",
                created_at=now,
                updated_at=now,
            )
            changes: dict[str, Any] = {
                "task_run_id": task_run_id,
                "task_runs": [*current.task_runs, task_run],
            }
            if current.status == "failed":
                changes.update(
                    {
                        "status": "running",
                        "message": f"Retrying {current.node_id} with a new TaskRun.",
                        "error": None,
                        "finished_at": None,
                    }
                )
            elif reopen_sample:
                changes.update(
                    {
                        "status": "running",
                        "message": (
                            f"Reopening {current.node_id} with a new sample TaskRun."
                        ),
                        "error": None,
                        "finished_at": None,
                    }
                )
            updated = current.model_copy(update=changes)
            self._write_run(updated)
            return updated, task_run, True

    def get(self, *, project_id: str, run_id: str) -> StageRun:
        canonical_run_id = _canonical_id(run_id, kind="run")
        path = self._project_dir(project_id) / "runs" / f"{canonical_run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"run not found: {run_id}")
        run = StageRun.model_validate_json(path.read_text(encoding="utf-8"))
        if (
            _canonical_id(run.project_id, kind="project")
            != _canonical_id(project_id, kind="project")
            or run.run_id != canonical_run_id
        ):
            raise ValueError("run metadata does not match its storage path")
        return run

    def record_result(
        self,
        *,
        project: ProjectInfo,
        run_id: str,
        request_id: str,
        outcome: StageReviewOutcome,
        message: str,
        input_refs: list[str],
        details: dict[str, Any],
        external_task_id: str | None = None,
        task_run_id: str | None = None,
        output_uri: str | None = None,
    ) -> StageRun:
        project_dir = self._project_dir(project.project.project_id)
        lock = FileLock(str(project_dir / ".run-mutation.lock"))
        with lock:
            run = self.get(
                project_id=project.project.project_id,
                run_id=run_id,
            )
            if run.mode != "agent_development":
                raise ValueError("Only Agent development runs accept stage results")
            if any(update.request_id == request_id for update in run.updates):
                return run
            if run.status in {"succeeded", "failed"}:
                raise ValueError(f"Stage run is already terminal: {run.status}")
            selected_task = self._select_task_run(run, task_run_id=task_run_id)
            if selected_task is not None and selected_task.status in {
                "succeeded",
                "failed",
            }:
                raise ValueError(
                    f"TaskRun is already terminal: {selected_task.task_run_id}"
                )
            if (
                outcome in {"waiting_user", "waiting_external"}
                and run.status != "running"
            ):
                raise ValueError(
                    f"Stage run must be running before recording {outcome}: "
                    f"{run.status}"
                )
            if outcome == "succeeded" and run.status not in {
                "running",
                "waiting_external",
            }:
                raise ValueError(
                    f"Stage run must be running or waiting_external before "
                    f"succeeding: {run.status}"
                )
            if outcome == "waiting_external":
                if not external_task_id or not output_uri:
                    raise ValueError(
                        "waiting_external requires external_task_id and output_uri"
                    )
                if selected_task is not None and selected_task.status != "running":
                    raise ValueError(
                        "TaskRun must be running before recording waiting_external: "
                        f"{selected_task.status}"
                    )
                if selected_task is not None and selected_task.external_task_id:
                    raise ValueError("TaskRun already has an external platform task")
            if outcome == "succeeded":
                if selected_task is None:
                    external_task_id = external_task_id or run.external_task_id
                    output_uri = output_uri or run.output_uri
                    if run.status == "waiting_external" and (
                        not external_task_id or not output_uri
                    ):
                        raise ValueError(
                            "succeeded requires the submitted platform task and "
                            "output_uri"
                        )
                    if (
                        run.status == "waiting_external"
                        and external_task_id != run.external_task_id
                    ):
                        raise ValueError(
                            "succeeded cannot change the submitted external_task_id"
                        )
                    if (
                        run.status == "waiting_external"
                        and output_uri != run.output_uri
                    ):
                        raise ValueError(
                            "succeeded cannot change the submitted output_uri"
                        )
                elif selected_task.status == "waiting_external":
                    external_task_id = (
                        external_task_id or selected_task.external_task_id
                    )
                    output_uri = output_uri or selected_task.output_uri
                    if not external_task_id or not output_uri:
                        raise ValueError(
                            "succeeded requires the submitted platform task and "
                            "output_uri"
                        )
                    if external_task_id != selected_task.external_task_id:
                        raise ValueError(
                            "succeeded cannot change the submitted external_task_id"
                        )
                    if output_uri != selected_task.output_uri:
                        raise ValueError(
                            "succeeded cannot change the submitted output_uri"
                        )
                else:
                    external_task_id = (
                        external_task_id or selected_task.external_task_id
                    )
                    output_uri = output_uri or selected_task.output_uri
                if bool(external_task_id) != bool(output_uri):
                    raise ValueError(
                        "external_task_id and output_uri must be recorded together"
                    )

            node = next(
                (item for item in project.graph.nodes if item.id == run.node_id),
                None,
            )
            if node is None:
                raise ValueError("Stage run node no longer exists")
            if _project_node_digest_for_run(node, run) != run.node_config_digest:
                raise ValueError("Stage run configuration changed during development")
            if outcome == "waiting_user":
                require_sample_validation(node, details)
                if node.type == "dataset":
                    task_run_id = (
                        selected_task.task_run_id
                        if selected_task is not None
                        else run.task_run_id
                    )
                    if not task_run_id:
                        raise ValueError(
                            "Dataset sample validation requires an attached TaskRun"
                        )
                    require_dataset_submission_spec(
                        details,
                        task_run_id=task_run_id,
                    )

            payload = StageArtifactPayload(
                kind=_stage_artifact_kind(node, outcome),
                schema_ref=(
                    next(iter(node.outputs.values())).schema_ref
                    if outcome == "succeeded" and node.outputs
                    else None
                ),
                content={
                    "schema_version": 1,
                    "project_id": project.project.project_id,
                    "graph_id": project.graph.graph_id,
                    "run_id": run.run_id,
                    "node_id": run.node_id,
                    "node_type": run.node_type,
                    "task_type": run.task_type,
                    "outcome": outcome,
                    "message": message,
                    "input_refs": input_refs,
                    "external_task_id": external_task_id,
                    "task_run_id": (
                        selected_task.task_run_id
                        if selected_task is not None
                        else run.task_run_id
                    ),
                    "output_uri": output_uri,
                    "details": details,
                },
                metadata=_sample_artifact_metadata(node, outcome, details),
            )
            artifact, artifact_path = self._write_artifact(run, payload)
            now = _utc_now()
            updated_task_runs = list(run.task_runs)
            updated_selected_task = selected_task
            if selected_task is not None:
                updated_selected_task = selected_task.model_copy(
                    update={
                        "status": outcome,
                        "message": message,
                        "external_task_id": external_task_id,
                        "output_uri": output_uri,
                        "artifact_ids": [
                            *selected_task.artifact_ids,
                            artifact.artifact_id,
                        ],
                        "updated_at": now,
                        "finished_at": (
                            now
                            if outcome in {"succeeded", "failed"}
                            else selected_task.finished_at
                        ),
                    }
                )
                updated_task_runs = [
                    (
                        updated_selected_task
                        if item.task_run_id == selected_task.task_run_id
                        else item
                    )
                    for item in run.task_runs
                ]
            update = StageRunUpdate(
                request_id=request_id,
                status=outcome,
                message=message,
                artifact_id=artifact.artifact_id,
                task_run_id=(
                    updated_selected_task.task_run_id
                    if updated_selected_task is not None
                    else run.task_run_id
                ),
                external_task_id=external_task_id,
                created_at=now,
            )
            updated = run.model_copy(
                update={
                    "status": outcome,
                    "message": message,
                    "input_refs": input_refs,
                    "task_run_id": (
                        updated_selected_task.task_run_id
                        if updated_selected_task is not None
                        else run.task_run_id
                    ),
                    "external_task_id": external_task_id,
                    "output_uri": output_uri,
                    "task_runs": updated_task_runs,
                    "artifacts": [*run.artifacts, artifact],
                    "updates": [*run.updates, update],
                    "output_refs": (
                        [artifact.artifact_id]
                        if outcome == "succeeded"
                        else run.output_refs
                    ),
                    "error": message if outcome == "failed" else run.error,
                    "finished_at": (
                        now if outcome in {"succeeded", "failed"} else run.finished_at
                    ),
                }
            )
            try:
                self._write_run(updated)
            except Exception:
                artifact_path.unlink(missing_ok=True)
                raise
            return updated

    def approve_plan(
        self,
        *,
        project: ProjectInfo,
        run_id: str,
        request_id: str,
        summary: str,
        decisions: dict[str, str],
    ) -> StageRun:
        project_dir = self._project_dir(project.project.project_id)
        lock = FileLock(str(project_dir / ".run-mutation.lock"))
        with lock:
            run = self.get(
                project_id=project.project.project_id,
                run_id=run_id,
            )
            if any(update.request_id == request_id for update in run.updates):
                return run
            if run.mode != "agent_development":
                raise ValueError("Only Agent development runs accept plan approval")
            if run.status != "clarifying":
                raise ValueError(
                    f"Stage run must be clarifying before plan approval: {run.status}"
                )

            node = next(
                (item for item in project.graph.nodes if item.id == run.node_id),
                None,
            )
            if node is None:
                raise ValueError("Stage run node no longer exists")
            if _project_node_digest_for_run(node, run) != run.node_config_digest:
                raise ValueError("Stage run configuration changed during clarification")

            message = f"Stage plan approved: {summary}"
            update = StageRunUpdate(
                request_id=request_id,
                status="running",
                message=message,
                details={"summary": summary, "decisions": decisions},
            )
            updated = run.model_copy(
                update={
                    "status": "running",
                    "message": message,
                    "updates": [*run.updates, update],
                }
            )
            self._write_run(updated)
            return updated

    def begin_continuation(
        self,
        *,
        project: ProjectInfo,
        run_id: str,
        request_id: str,
        command_id: str,
    ) -> tuple[StageRun, bool]:
        project_dir = self._project_dir(project.project.project_id)
        lock = FileLock(str(project_dir / ".run-mutation.lock"))
        with lock:
            run = self.get(
                project_id=project.project.project_id,
                run_id=run_id,
            )
            existing = next(
                (update for update in run.updates if update.request_id == request_id),
                None,
            )
            if existing is not None:
                return run, False
            if run.mode != "agent_development":
                raise ValueError("Only Agent development runs can continue")
            if run.status != "waiting_user":
                raise ValueError(
                    f"Stage run must be waiting_user before continuing: {run.status}"
                )

            message = f"Continuing {run.node_id} after user confirmation."
            update = StageRunUpdate(
                request_id=request_id,
                status="running",
                message=message,
                command_id=command_id,
            )
            updated = run.model_copy(
                update={
                    "status": "running",
                    "message": message,
                    "updates": [*run.updates, update],
                }
            )
            self._write_run(updated)
            return updated, True

    def reconcile_interrupted(self) -> list[StageRun]:
        recovered: list[StageRun] = []
        for path in sorted(self.projects_root.glob("*/runs/*.json")):
            try:
                candidate = StageRun.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            if candidate.mode != "agent_development" or candidate.status != "running":
                continue
            project_dir = path.parent.parent
            lock = FileLock(str(project_dir / ".run-mutation.lock"))
            with lock:
                current = StageRun.model_validate_json(path.read_text(encoding="utf-8"))
                if current.mode != "agent_development" or current.status != "running":
                    continue
                failed = current.model_copy(
                    update={
                        "status": "failed",
                        "message": f"{current.node_id} was interrupted.",
                        "error": (
                            "RuntimeRestarted: the Agent development run was "
                            "interrupted before it recorded a result"
                        ),
                        "finished_at": _utc_now(),
                    }
                )
                self._write_run(failed)
                recovered.append(failed)
        return recovered

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
                    "input_refs": result.input_refs,
                    "output_refs": result.output_refs
                    or [artifact.artifact_id for artifact in artifacts],
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

    def _find_by_request_id(
        self,
        project_id: str,
        request_id: str | None,
    ) -> StageRun | None:
        if request_id is None:
            return None
        return next(
            (run for run in self.list(project_id) if run.request_id == request_id),
            None,
        )

    def _select_task_run(
        self,
        run: StageRun,
        *,
        task_run_id: str | None,
    ) -> StageTaskRun | None:
        selected_id = task_run_id or run.task_run_id
        if selected_id is None:
            return None
        selected = next(
            (item for item in run.task_runs if item.task_run_id == selected_id),
            None,
        )
        if selected is None:
            raise ValueError(f"TaskRun is not attached to this StageRun: {selected_id}")
        return selected

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
            schema_ref=payload.schema_ref,
            metadata=payload.metadata,
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
    request_id: str | None = None,
) -> StageRun:
    run, created = run_store.start(
        project=project,
        node=node,
        request_id=request_id,
    )
    if not created:
        return run
    try:
        executor = get_project_node_executor(node.executor.kind)
        result = executor.run(
            context=NodeExecutionContext(
                run_id=run.run_id,
                project=project,
                node=node,
            )
        )
        return run_store.succeed(run, result)
    except Exception as exc:
        return run_store.fail(run, exc)


def project_node_config_digest(
    node: ProjectNode,
    *,
    version: StageRunConfigDigestVersion = 2,
) -> str:
    if version == 1:
        payload_fields: dict[str, Any] = {
            "type": node.type,
            "executor": node.executor.kind,
            "config": node.config,
        }
    else:
        payload_fields = node.model_dump(mode="json", exclude={"status"})
    payload = json.dumps(
        payload_fields,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{sha256(payload.encode('utf-8')).hexdigest()}"


def _project_node_digest_for_run(node: ProjectNode, run: StageRun) -> str:
    return project_node_config_digest(
        node,
        version=run.config_digest_version,
    )


def require_sample_validation(
    node: ProjectNode,
    details: dict[str, Any],
) -> None:
    contract = node.config.get("sample_validation")
    if not isinstance(contract, dict) or contract.get("required") is not True:
        return

    required_details = contract.get("required_details", [])
    if not isinstance(required_details, list):
        raise ValueError("sample_validation.required_details must be a list")
    missing_details = [
        name
        for name in required_details
        if isinstance(name, str) and name != "formal_validation" and name not in details
    ]
    if missing_details:
        raise ValueError(
            "waiting_user is missing required sample details: "
            + ", ".join(missing_details)
        )

    checks = details.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("waiting_user requires details.checks")
    evidence = checks.get("formal_validation")
    if not isinstance(evidence, dict):
        raise ValueError("waiting_user requires details.checks.formal_validation")

    formal = contract.get("formal_validation")
    if not isinstance(formal, dict):
        raise ValueError("sample_validation.formal_validation must be configured")
    required_fields = formal.get("required_fields", [])
    if not isinstance(required_fields, list):
        raise ValueError("formal_validation.required_fields must be a list")
    missing_fields = [
        name
        for name in required_fields
        if isinstance(name, str) and name not in evidence
    ]
    if missing_fields:
        raise ValueError(
            "formal_validation is missing required fields: " + ", ".join(missing_fields)
        )

    for field in ("tool", "schema", "status"):
        expected = formal.get(field)
        if isinstance(expected, str) and evidence.get(field) != expected:
            raise ValueError(f"formal_validation.{field} must be {expected!r}")

    rows = evidence.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise ValueError("formal_validation.rows must be a positive integer")
    rows_detail = formal.get("rows_detail")
    if isinstance(rows_detail, str):
        detail_rows = details.get(rows_detail)
        if detail_rows != rows:
            raise ValueError(f"details.{rows_detail} must match formal_validation.rows")

    command = evidence.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("formal_validation.command must be a non-empty string")
    command_contains = formal.get("command_contains", [])
    if not isinstance(command_contains, list):
        raise ValueError("formal_validation.command_contains must be a list")
    missing_fragments = [
        fragment
        for fragment in command_contains
        if isinstance(fragment, str) and fragment not in command
    ]
    if missing_fragments:
        raise ValueError(
            "formal_validation.command is missing required fragments: "
            + ", ".join(missing_fragments)
        )

    for field in ("report_uri",):
        value = evidence.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"formal_validation.{field} must be a non-empty string")
    output_sha256 = evidence.get("output_sha256")
    if not isinstance(output_sha256, str) or not re.fullmatch(
        r"(?:sha256:)?[0-9a-f]{64}", output_sha256
    ):
        raise ValueError("formal_validation.output_sha256 must be a SHA-256 hex digest")


def _sample_artifact_metadata(
    node: ProjectNode,
    outcome: StageReviewOutcome,
    details: dict[str, Any],
) -> dict[str, Any]:
    if outcome != "waiting_user" or not isinstance(
        node.config.get("sample_validation"), dict
    ):
        return {}
    checks = details.get("checks")
    if not isinstance(checks, dict):
        return {}
    evidence = checks.get("formal_validation")
    return {"formal_validation": evidence} if isinstance(evidence, dict) else {}


def _stage_artifact_kind(
    node: ProjectNode,
    outcome: StageReviewOutcome,
) -> str:
    if outcome == "waiting_user":
        configured = node.config.get("sample_report_kind")
        return (
            configured
            if isinstance(configured, str) and configured
            else ("stage_sample_report")
        )
    if outcome == "waiting_external":
        configured = node.config.get("submission_receipt_kind")
        return (
            configured
            if isinstance(configured, str) and configured
            else ("stage_submission_receipt")
        )
    if outcome == "succeeded":
        if node.outputs:
            return next(iter(node.outputs.values())).artifact_type
        return "stage_result"
    return "stage_error"


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
