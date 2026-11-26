from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyromind_agent_server.echomind_storage import (
    ProjectLease,
    canonical_sha256,
    canonical_uuid,
    project_directory,
    read_model,
    write_model,
)


GraphRunStatus = Literal[
    "pending",
    "running",
    "waiting_external",
    "paused",
    "succeeded",
    "failed",
    "canceled",
]
GraphRunOutcome = Literal[
    "candidate_recorded",
    "promoted",
    "baseline_retained",
    "canceled",
    "failed",
]
GraphRunTrigger = Literal["manual", "tracer_batch"]
GraphRunExecutionMode = Literal["debug", "published"]
TaskRunStatus = Literal[
    "submitting",
    "submitted",
    "running",
    "succeeded",
    "failed",
    "canceled",
    "unknown",
]
ProviderReportedTaskStatus = Literal[
    "pending", "running", "succeeded", "failed", "canceled"
]
FailureCategory = Literal[
    "validation",
    "external",
    "infrastructure",
    "policy",
    "canceled",
]

ACTIVE_GRAPH_RUN_STATUSES: frozenset[GraphRunStatus] = frozenset(
    {"pending", "running", "waiting_external", "paused"}
)
TERMINAL_GRAPH_RUN_STATUSES: frozenset[GraphRunStatus] = frozenset(
    {"succeeded", "failed", "canceled"}
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_type: str = Field(min_length=1, max_length=256)
    uri: str = Field(min_length=1, max_length=4000)
    content_sha256: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: FailureCategory
    message: str = Field(min_length=1, max_length=4000)
    details: dict[str, Any] = Field(default_factory=dict)
    failed_at: datetime = Field(default_factory=_utc_now)


class GraphRunRepairHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    handoff_id: str
    project_id: str
    run_id: str
    release_id: str
    release_digest: str
    stage_run_id: str
    node_id: str
    attempt: int = Field(ge=1)
    provider_task_id: str | None = None
    failure: RuntimeFailure
    delivered_at: datetime | None = None


class GraphRunRepairRestart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    run_id: str
    release_id: str
    release_digest: str
    inputs_digest: str


class TaskRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    task_run_id: str
    stage_run_id: str
    provider: str = Field(min_length=1, max_length=128)
    provider_run_id: str | None = Field(default=None, max_length=256)
    idempotency_key: str = Field(min_length=1, max_length=512)
    status: TaskRunStatus = "submitting"
    provider_reported_status: ProviderReportedTaskStatus | None = None
    request_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=_utc_now)
    submitted_at: datetime | None = None
    last_polled_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def validate_status_fields(self) -> TaskRun:
        if self.status in {"submitted", "running", "succeeded"}:
            if not self.provider_run_id:
                raise ValueError(f"{self.status} TaskRun requires a provider_run_id")
        if self.status in {"submitted", "running", "succeeded"}:
            if self.submitted_at is None:
                raise ValueError(f"{self.status} TaskRun requires submitted_at")
        if self.status in {"succeeded", "failed", "canceled"}:
            if self.finished_at is None:
                raise ValueError(f"{self.status} TaskRun requires finished_at")
        elif self.finished_at is not None:
            raise ValueError("non-terminal TaskRun must not have finished_at")
        if self.status == "failed" and not self.error:
            raise ValueError("failed TaskRun requires an error")
        return self


class GraphRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: str
    project_id: str
    release_id: str
    release_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    round_no: int = Field(ge=1)
    request_id: str = Field(min_length=1, max_length=256)
    trigger: GraphRunTrigger
    execution_mode: GraphRunExecutionMode = "published"
    source_tracer_resource_id: str | None = Field(default=None, max_length=128)
    source_batch_id: str | None = Field(default=None, max_length=256)
    status: GraphRunStatus = "pending"
    outcome: GraphRunOutcome | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    current_stage_ids: list[str] = Field(default_factory=list)
    promotion_result: dict[str, Any] = Field(default_factory=dict)
    failure: RuntimeFailure | None = None
    repair_handoffs: list[GraphRunRepairHandoff] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    repair_restart: GraphRunRepairRestart | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    repair_of_run_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    created_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def validate_run_state(self) -> GraphRun:
        if len(set(self.current_stage_ids)) != len(self.current_stage_ids):
            raise ValueError("current_stage_ids must be unique")
        if self.trigger == "tracer_batch":
            if self.execution_mode == "debug":
                raise ValueError("Debug GraphRuns require a manual trigger")
            if not self.source_tracer_resource_id or not self.source_batch_id:
                raise ValueError("tracer_batch GraphRun requires tracer and batch IDs")
        elif self.source_tracer_resource_id or self.source_batch_id:
            raise ValueError("manual GraphRun must not include tracer batch identity")

        if self.status == "pending" and self.started_at is not None:
            raise ValueError("pending GraphRun must not have started_at")
        if self.status in {"running", "waiting_external", "paused"}:
            if self.started_at is None:
                raise ValueError(f"{self.status} GraphRun requires started_at")

        if self.status in TERMINAL_GRAPH_RUN_STATUSES:
            if self.finished_at is None:
                raise ValueError(f"{self.status} GraphRun requires finished_at")
        elif self.finished_at is not None:
            raise ValueError("non-terminal GraphRun must not have finished_at")

        if self.status == "succeeded":
            if self.outcome not in {
                "promoted",
                "baseline_retained",
                "candidate_recorded",
            }:
                raise ValueError("succeeded GraphRun requires a completed outcome")
        elif self.status == "failed":
            if self.outcome != "failed" or self.failure is None:
                raise ValueError("failed GraphRun requires failure details")
        elif self.status == "canceled":
            if self.outcome != "canceled":
                raise ValueError("canceled GraphRun requires canceled outcome")
        elif self.outcome is not None:
            raise ValueError("non-terminal GraphRun must not have an outcome")
        return self

    @property
    def batch_idempotency_key(self) -> str | None:
        if self.trigger != "tracer_batch":
            return None
        return (
            f"{self.project_id}:{self.source_tracer_resource_id}:{self.source_batch_id}"
        )

    def current_repair_handoff(self) -> GraphRunRepairHandoff | None:
        if self.status != "paused" or self.failure is None:
            return None
        return next(
            (
                handoff
                for handoff in reversed(self.repair_handoffs)
                if handoff.stage_run_id in self.current_stage_ids
            ),
            None,
        )


class ActiveGraphRunExistsError(RuntimeError):
    pass


class GraphRunStore:
    def __init__(self, projects_root: Path | str) -> None:
        self.projects_root = Path(projects_root)

    def create(
        self,
        *,
        project_id: str,
        release_id: str,
        release_digest: str,
        request_id: str,
        trigger: GraphRunTrigger,
        inputs: dict[str, Any],
        execution_mode: GraphRunExecutionMode = "published",
        source_tracer_resource_id: str | None = None,
        source_batch_id: str | None = None,
        repair_of_run_id: str | None = None,
        run_id: str | None = None,
    ) -> GraphRun:
        existing_request = self.find_by_request_id(project_id, request_id)
        if existing_request is not None:
            return existing_request

        if trigger == "tracer_batch" and source_tracer_resource_id and source_batch_id:
            existing_batch = self.find_by_batch_key(
                project_id=project_id,
                tracer_resource_id=source_tracer_resource_id,
                batch_id=source_batch_id,
            )
            if existing_batch is not None:
                return existing_batch

        active = self.active(project_id)
        if active is not None:
            raise ActiveGraphRunExistsError(
                f"Project already has an active GraphRun: {active.run_id}"
            )

        runs = self.list(project_id)
        next_round = max((run.round_no for run in runs), default=0) + 1
        run = GraphRun(
            run_id=run_id or str(uuid4()),
            project_id=project_id,
            release_id=release_id,
            release_digest=release_digest,
            round_no=next_round,
            request_id=request_id,
            trigger=trigger,
            execution_mode=execution_mode,
            source_tracer_resource_id=source_tracer_resource_id,
            source_batch_id=source_batch_id,
            inputs=inputs,
            repair_of_run_id=repair_of_run_id,
        )
        return self.save(run)

    def save(self, run: GraphRun) -> GraphRun:
        write_model(self._run_path(run.project_id, run.run_id), run)
        return run

    def load(self, *, project_id: str, run_id: str) -> GraphRun:
        path = self._run_path(project_id, run_id)
        if not path.is_file():
            raise FileNotFoundError(f"GraphRun not found: {run_id}")
        run = read_model(path, GraphRun)
        if canonical_uuid(run.project_id, kind="project") != canonical_uuid(
            project_id, kind="project"
        ):
            raise ValueError("GraphRun belongs to another project")
        return run

    def list(self, project_id: str) -> list[GraphRun]:
        runs_root = project_directory(self.projects_root, project_id) / "graph_runs"
        if not runs_root.is_dir():
            return []
        runs: list[GraphRun] = []
        for run_dir in runs_root.iterdir():
            path = run_dir / "graph_run.json"
            if path.is_file():
                runs.append(read_model(path, GraphRun))
        return sorted(runs, key=lambda run: (run.created_at, run.run_id), reverse=True)

    def active(self, project_id: str) -> GraphRun | None:
        return next(
            (
                run
                for run in self.list(project_id)
                if run.status in ACTIVE_GRAPH_RUN_STATUSES
            ),
            None,
        )

    def get_repair_handoff(
        self, project_id: str, run_id: str
    ) -> GraphRunRepairHandoff | None:
        return self.load(project_id=project_id, run_id=run_id).current_repair_handoff()

    def list_pending_repair_handoffs(
        self, project_id: str
    ) -> list[GraphRunRepairHandoff]:
        return [
            handoff
            for run in self.list(project_id)
            if (handoff := run.current_repair_handoff()) is not None
            and handoff.delivered_at is None
        ]

    def mark_repair_handoff_delivered(
        self, project_id: str, run_id: str, handoff_id: str
    ) -> None:
        with ProjectLease(self.projects_root, project_id).acquire():
            run = self.load(project_id=project_id, run_id=run_id)
            handoffs = [
                handoff.model_copy(update={"delivered_at": _utc_now()})
                if handoff.handoff_id == handoff_id and handoff.delivered_at is None
                else handoff
                for handoff in run.repair_handoffs
            ]
            if handoffs != run.repair_handoffs:
                self.save(run.model_copy(update={"repair_handoffs": handoffs}))

    def find_by_request_id(
        self,
        project_id: str,
        request_id: str,
    ) -> GraphRun | None:
        return next(
            (run for run in self.list(project_id) if run.request_id == request_id),
            None,
        )

    def find_by_batch_key(
        self,
        *,
        project_id: str,
        tracer_resource_id: str,
        batch_id: str,
    ) -> GraphRun | None:
        return next(
            (
                run
                for run in self.list(project_id)
                if run.trigger == "tracer_batch"
                and run.source_tracer_resource_id == tracer_resource_id
                and run.source_batch_id == batch_id
            ),
            None,
        )

    def save_task_run(
        self,
        *,
        project_id: str,
        graph_run_id: str,
        task_run: TaskRun,
    ) -> TaskRun:
        self.load(project_id=project_id, run_id=graph_run_id)
        write_model(
            self._task_run_path(project_id, graph_run_id, task_run.task_run_id),
            task_run,
        )
        return task_run

    def load_task_run(
        self,
        *,
        project_id: str,
        graph_run_id: str,
        task_run_id: str,
    ) -> TaskRun:
        path = self._task_run_path(project_id, graph_run_id, task_run_id)
        if not path.is_file():
            raise FileNotFoundError(f"TaskRun not found: {task_run_id}")
        return read_model(path, TaskRun)

    def list_task_runs(
        self,
        *,
        project_id: str,
        graph_run_id: str,
    ) -> list[TaskRun]:
        task_runs_root = self._run_dir(project_id, graph_run_id) / "task_runs"
        if not task_runs_root.is_dir():
            return []
        return sorted(
            (read_model(path, TaskRun) for path in task_runs_root.glob("*.json")),
            key=lambda task_run: (task_run.created_at, task_run.task_run_id),
        )

    def find_task_run_by_provider_id(
        self,
        *,
        project_id: str,
        graph_run_id: str,
        provider_run_id: str,
    ) -> TaskRun | None:
        return next(
            (
                task_run
                for task_run in reversed(
                    self.list_task_runs(
                        project_id=project_id,
                        graph_run_id=graph_run_id,
                    )
                )
                if task_run.provider_run_id == provider_run_id
            ),
            None,
        )

    def request_digest(self, payload: dict[str, Any]) -> str:
        return canonical_sha256(payload)

    def _run_dir(self, project_id: str, run_id: str) -> Path:
        canonical_run_id = canonical_uuid(run_id, kind="GraphRun")
        return (
            project_directory(self.projects_root, project_id)
            / "graph_runs"
            / canonical_run_id
        )

    def _run_path(self, project_id: str, run_id: str) -> Path:
        return self._run_dir(project_id, run_id) / "graph_run.json"

    def _task_run_path(
        self,
        project_id: str,
        graph_run_id: str,
        task_run_id: str,
    ) -> Path:
        canonical_task_run_id = canonical_uuid(task_run_id, kind="TaskRun")
        return (
            self._run_dir(project_id, graph_run_id)
            / "task_runs"
            / f"{canonical_task_run_id}.json"
        )
