from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field

from pyromind_agent_server.echomind_dataset_runtime import DatasetRuntimeExecutor
from pyromind_agent_server.echomind_graph_runs import (
    ACTIVE_GRAPH_RUN_STATUSES,
    ActiveGraphRunExistsError,
    ArtifactRef,
    FailureCategory,
    GraphRun,
    GraphRunExecutionMode,
    GraphRunRepairHandoff,
    GraphRunRepairRestart,
    GraphRunStore,
    GraphRunTrigger,
    ProviderReportedTaskStatus,
    RuntimeFailure,
    TaskRun,
)
from pyromind_agent_server.echomind_projects import (
    ProjectInfo,
    ProjectNode,
    ProjectNodeInput,
    ProjectStore,
)
from pyromind_agent_server.echomind_publication import (
    PublicationStore,
    PublishedGraph,
)
from pyromind_agent_server.echomind_runtime_contracts import (
    ArtifactReplayRuntimeExecutor,
    CandidateRecordingFinalizer,
    ContractProbeRuntimeExecutor,
    ExternalTaskIntent,
    RuntimeArtifact,
    RuntimeCredentialUnavailableError,
    RuntimeFinalization,
    RuntimeFinalizer,
    RuntimeNodeExecutor,
    RuntimeStageContext,
    RuntimeSubmissionNotAttemptedError,
    utc_now,
)
from pyromind_agent_server.echomind_runtime_slots import RuntimeSlotStore
from pyromind_agent_server.echomind_stage_runs import ProjectRunStore, StageRun
from pyromind_agent_server.echomind_storage import (
    ProjectLease,
    ProjectLeaseTimeout,
    canonical_sha256,
)
from pyromind_runtime.domain.echomind_deployments import TERMINAL_DEPLOYMENT_STATUSES
from pyromind_runtime.infrastructure.file_deployment_store import FileDeploymentStore


logger = logging.getLogger(__name__)

_PROVIDER_REPORTED_STATUSES: dict[str, ProviderReportedTaskStatus] = {
    "new": "pending",
    "pending": "pending",
    "running": "running",
    "succeeded": "succeeded",
    "success": "succeeded",
    "completed": "succeeded",
    "failed": "failed",
    "error": "failed",
    "canceled": "canceled",
    "cancelled": "canceled",
    "terminated": "canceled",
    "stopped": "canceled",
}
_PROVIDER_STATUS_ORDER: dict[ProviderReportedTaskStatus | None, int] = {
    None: 0,
    "pending": 1,
    "running": 2,
    "succeeded": 3,
    "failed": 3,
    "canceled": 3,
}
_TERMINAL_TASK_STATUSES = frozenset({"succeeded", "failed", "canceled"})


class RuntimeSchedulerError(RuntimeError):
    pass


class ReadyBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tracer_resource_id: str = Field(min_length=1, max_length=128)
    batch_id: str = Field(min_length=1, max_length=256)
    inputs: dict[str, Any] = Field(default_factory=dict)


class ReadyBatchSource(Protocol):
    def list_ready_batches(
        self,
        *,
        project_id: str,
        tracer_resource_id: str,
    ) -> list[ReadyBatch]: ...


class RuntimeExecutorRegistry:
    def __init__(
        self,
        executors: Mapping[str, RuntimeNodeExecutor] | None = None,
    ) -> None:
        self._executors = dict(executors or {})
        self._executors.setdefault(
            "artifact_replay",
            ArtifactReplayRuntimeExecutor(),
        )
        self._executors.setdefault("contract_probe", ContractProbeRuntimeExecutor())
        self._executors.setdefault("dataset_runtime", DatasetRuntimeExecutor())

    def get(self, name: str) -> RuntimeNodeExecutor:
        try:
            return self._executors[name]
        except KeyError as exc:
            raise RuntimeSchedulerError(
                f"Unknown published-stage executor: {name}"
            ) from exc


class RuntimeGraphScheduler:
    def __init__(
        self,
        *,
        projects_root: Path | str,
        executors: Mapping[str, RuntimeNodeExecutor] | None = None,
        finalizer: RuntimeFinalizer | None = None,
        lease_timeout_seconds: float = 30,
    ) -> None:
        self.projects_root = Path(projects_root)
        self.graph_run_store = GraphRunStore(self.projects_root)
        self.stage_run_store = ProjectRunStore(self.projects_root)
        self.publication_store = PublicationStore(self.projects_root)
        self.runtime_slot_store = RuntimeSlotStore(self.projects_root)
        self.executor_registry = RuntimeExecutorRegistry(executors)
        self.finalizer = finalizer or CandidateRecordingFinalizer()
        self.lease_timeout_seconds = lease_timeout_seconds

    def start(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        request_id: str,
        inputs: dict[str, Any],
        trigger: GraphRunTrigger = "manual",
        execution_mode: GraphRunExecutionMode = "published",
        source_tracer_resource_id: str | None = None,
        source_batch_id: str | None = None,
    ) -> GraphRun:
        project_id = project.project.project_id
        with self._lease(project_id):
            project = ProjectStore(self.projects_root).load(
                project_id,
                owner_user_id=project.project.owner_user_id,
            )
            self._validate_start(
                project=project,
                published=published,
                execution_mode=execution_mode,
                require_runtime_slot=trigger == "tracer_batch",
            )
            run = self.graph_run_store.create(
                project_id=project_id,
                release_id=published.release_id,
                release_digest=published.content_sha256,
                request_id=request_id,
                trigger=trigger,
                execution_mode=execution_mode,
                inputs=inputs,
                source_tracer_resource_id=source_tracer_resource_id,
                source_batch_id=source_batch_id,
            )
            if run.execution_mode != execution_mode:
                raise RuntimeSchedulerError(
                    "This request ID belongs to a different GraphRun execution mode."
                )
            return self._advance_locked(
                project=project,
                published=published,
                run_id=run.run_id,
            )

    def advance(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        run_id: str,
    ) -> GraphRun:
        with self._lease(project.project.project_id):
            return self._advance_locked(
                project=project,
                published=published,
                run_id=run_id,
            )

    def get_repair_handoff(
        self, *, project_id: str, run_id: str
    ) -> GraphRunRepairHandoff | None:
        with self._lease(project_id):
            run = self.graph_run_store.load(project_id=project_id, run_id=run_id)
            return self._ensure_repair_handoff_locked(run)

    def _ensure_repair_handoff_locked(
        self, run: GraphRun
    ) -> GraphRunRepairHandoff | None:
        handoff = run.current_repair_handoff()
        if handoff is not None:
            return handoff
        if (
            run.execution_mode != "debug"
            or run.status != "paused"
            or run.failure is None
            or len(run.current_stage_ids) != 1
        ):
            return None
        stage = self.stage_run_store.load(
            project_id=run.project_id, run_id=run.current_stage_ids[0]
        )
        if stage.graph_run_id != run.run_id:
            raise RuntimeSchedulerError("Failed stage belongs to another GraphRun")
        handoff = _repair_handoff_for_failure(run, stage, run.failure)
        if handoff is not None:
            self.graph_run_store.save(
                run.model_copy(
                    update={"repair_handoffs": [*run.repair_handoffs, handoff]}
                )
            )
        return handoff

    def start_repair(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        repair_of_run_id: str,
        request_id: str,
        inputs: dict[str, Any],
    ) -> GraphRun:
        project_id = project.project.project_id
        with self._lease(project_id):
            project = ProjectStore(self.projects_root).load(
                project_id, owner_user_id=project.project.owner_user_id
            )
            self._validate_start(
                project=project,
                published=published,
                execution_mode="debug",
                require_runtime_slot=False,
            )
            original = self.graph_run_store.load(
                project_id=project_id, run_id=repair_of_run_id
            )
            if original.execution_mode != "debug":
                raise RuntimeSchedulerError("Only debug GraphRuns can start a repair")
            if published.content_sha256 == original.release_digest:
                raise RuntimeSchedulerError("Repair requires a changed debug snapshot")

            restart = GraphRunRepairRestart(
                request_id=request_id,
                run_id=str(uuid5(UUID(original.run_id), f"repair:{request_id}")),
                release_id=published.release_id,
                release_digest=published.content_sha256,
                inputs_digest=canonical_sha256(inputs),
            )
            existing = self.graph_run_store.find_by_request_id(project_id, request_id)
            if existing is not None:
                if (
                    original.repair_restart != restart
                    or existing.run_id != restart.run_id
                    or existing.repair_of_run_id != original.run_id
                ):
                    raise RuntimeSchedulerError("Request ID belongs to another run")
                return self._advance_locked(
                    project=project, published=published, run_id=existing.run_id
                )
            active = self.graph_run_store.active(project_id)
            if active is not None and active.run_id != original.run_id:
                raise ActiveGraphRunExistsError(
                    f"Project already has an active GraphRun: {active.run_id}"
                )
            previous_snapshot = self.publication_store.load(
                project_id=project_id, release_id=original.release_id
            )
            self._validate_loaded_run(
                project=project, published=previous_snapshot, run=original
            )
            if (
                published.nodes == previous_snapshot.nodes
                and published.edges == previous_snapshot.edges
                and published.runtime_requirements
                == previous_snapshot.runtime_requirements
                and published.promotion_policy == previous_snapshot.promotion_policy
            ):
                raise RuntimeSchedulerError("Repair requires a changed debug snapshot")
            if original.repair_restart is None:
                handoff = self._ensure_repair_handoff_locked(original)
                if handoff is None or (
                    handoff.provider_task_id is None
                    and handoff.failure.category != "validation"
                ):
                    raise RuntimeSchedulerError(
                        "Repair requires a paused debug run with a definite failure"
                    )
                tasks = self.graph_run_store.list_task_runs(
                    project_id=project_id, graph_run_id=original.run_id
                )
                if any(task.status not in _TERMINAL_TASK_STATUSES for task in tasks):
                    raise RuntimeSchedulerError(
                        "Cannot start repair while an original platform task is active"
                    )
                if handoff.provider_task_id is not None and not any(
                    task.provider_run_id == handoff.provider_task_id for task in tasks
                ):
                    raise RuntimeSchedulerError("Missing terminal platform task record")
                original = self.graph_run_store.load(
                    project_id=project_id, run_id=original.run_id
                )
                original = self.graph_run_store.save(
                    original.model_copy(
                        update={
                            "status": "failed",
                            "outcome": "failed",
                            "finished_at": utc_now(),
                            "repair_restart": restart,
                        }
                    )
                )
            elif original.repair_restart != restart:
                raise RuntimeSchedulerError(
                    "This GraphRun already has a repair restart"
                )

            if inputs == original.inputs:
                self._reuse_repair_stages(
                    original=original,
                    previous_snapshot=previous_snapshot,
                    published=published,
                    new_run_id=restart.run_id,
                )
            run = self.graph_run_store.create(
                project_id=project_id,
                release_id=published.release_id,
                release_digest=published.content_sha256,
                request_id=request_id,
                trigger="manual",
                execution_mode="debug",
                inputs=inputs,
                repair_of_run_id=original.run_id,
                run_id=restart.run_id,
            )
            return self._advance_locked(
                project=project, published=published, run_id=run.run_id
            )

    def _reuse_repair_stages(
        self,
        *,
        original: GraphRun,
        previous_snapshot: PublishedGraph,
        published: PublishedGraph,
        new_run_id: str,
    ) -> None:
        old_nodes = {node.id: node for node in previous_snapshot.nodes}
        old_stages = self.stage_run_store.list_for_graph_run(
            project_id=original.project_id, graph_run_id=original.run_id
        )
        staged = self.stage_run_store.list_for_graph_run(
            project_id=original.project_id, graph_run_id=new_run_id
        )
        reused: set[str] = set()
        reused_stage_ids: set[str] = set()
        for node in _topological_nodes(published):
            stage = _latest_stage_for_node(old_stages, node.id)
            if (
                node != old_nodes.get(node.id)
                or stage is None
                or stage.status != "succeeded"
            ):
                continue
            predecessors = _predecessors(published, node.id)
            if predecessors != _predecessors(
                previous_snapshot, node.id
            ) or not predecessors.issubset(reused):
                continue
            if not self._has_registered_outputs(stage, node):
                continue
            reused_id = str(uuid5(UUID(new_run_id), f"reuse:{stage.run_id}"))
            self.stage_run_store.save(
                stage.model_copy(
                    update={
                        "run_id": reused_id,
                        "graph_run_id": new_run_id,
                        "attempt": 1,
                        "message": f"Reused outputs from {stage.run_id}.",
                        "task_runs": [],
                        "reused_from_stage_run_id": stage.run_id,
                    }
                )
            )
            reused.add(node.id)
            reused_stage_ids.add(reused_id)
        for stage in staged:
            if stage.run_id not in reused_stage_ids:
                self.stage_run_store.discard_reused_stage(stage)

    def _has_registered_outputs(self, stage: StageRun, node: ProjectNode) -> bool:
        if not stage.output_refs:
            return False
        artifacts = {item.artifact_id: item for item in stage.artifacts}
        ports: set[str] = set()
        for ref in stage.output_refs:
            artifact_id = ref.metadata.get("artifact_id")
            port = ref.metadata.get("port")
            if not isinstance(artifact_id, str) or not isinstance(port, str):
                return False
            artifact = artifacts.get(artifact_id)
            output = node.outputs.get(port)
            if (
                artifact is None
                or output is None
                or port in ports
                or ref.artifact_type != output.artifact_type
                or ref.uri != artifact.uri
                or ref.content_sha256 != f"sha256:{artifact.content_sha256}"
            ):
                return False
            try:
                document = self.stage_run_store.load_artifact(
                    project_id=stage.project_id, artifact_id=artifact_id
                )
            except (OSError, ValueError):
                return False
            if (
                document.artifact != artifact
                or canonical_sha256(document.content) != ref.content_sha256
                or (
                    "source_uri" in ref.metadata
                    and document.content.get("uri") != ref.metadata["source_uri"]
                )
            ):
                return False
            ports.add(port)
        return all(
            not output.required or port in ports
            for port, output in node.outputs.items()
        )

    def callback(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        run_id: str,
        provider_run_id: str,
        status: str,
        observe_provider_status: bool = False,
    ) -> GraphRun:
        with self._lease(project.project.project_id):
            run = self.graph_run_store.load(
                project_id=project.project.project_id, run_id=run_id
            )
            self._validate_loaded_run(project=project, published=published, run=run)
            task = self.graph_run_store.find_task_run_by_provider_id(
                project_id=project.project.project_id,
                graph_run_id=run_id,
                provider_run_id=provider_run_id,
            )
            if task is None:
                raise RuntimeSchedulerError(
                    f"Unknown provider task for callback: {provider_run_id}"
                )
            if observe_provider_status:
                reported_status = _PROVIDER_REPORTED_STATUSES.get(
                    status.strip().lower()
                )
                if reported_status is not None:
                    self._record_provider_status(run, task, reported_status)
            return self._advance_locked(
                project=project, published=published, run_id=run_id
            )

    def _record_provider_status(
        self, run: GraphRun, task: TaskRun, reported_status: ProviderReportedTaskStatus
    ) -> None:
        stage = self.stage_run_store.load(
            project_id=run.project_id, run_id=task.stage_run_id
        )
        embedded = next(
            (item for item in stage.task_runs if item.task_run_id == task.task_run_id),
            None,
        )
        observed = max(
            (
                task.provider_reported_status,
                embedded.provider_reported_status if embedded is not None else None,
                reported_status,
            ),
            key=_PROVIDER_STATUS_ORDER.__getitem__,
        )
        if embedded is not None and embedded.provider_reported_status != observed:
            updated = embedded.model_copy(update={"provider_reported_status": observed})
            self.stage_run_store.save(self._replace_task(stage, updated))
        if task.provider_reported_status != observed:
            self._save_task(
                run=run,
                task=task.model_copy(update={"provider_reported_status": observed}),
            )

    def callback_by_provider_task(
        self,
        *,
        provider_run_id: str,
        status: str,
        conversation_id: str | None = None,
        task_kind: str | None = None,
    ) -> GraphRun | None:
        project_store = ProjectStore(self.projects_root)
        match: tuple[ProjectInfo, PublishedGraph, GraphRun] | None = None
        for summary in project_store.list(owner_user_id=None):
            if conversation_id is not None and _conversation_id(
                summary.conversation_id
            ) != _conversation_id(conversation_id):
                continue
            for run in self.graph_run_store.list(summary.project_id):
                if summary.lifecycle != "frozen" and run.execution_mode != "debug":
                    continue
                task = self.graph_run_store.find_task_run_by_provider_id(
                    project_id=summary.project_id,
                    graph_run_id=run.run_id,
                    provider_run_id=provider_run_id,
                )
                if task is None:
                    continue
                stage = self.stage_run_store.load(
                    project_id=summary.project_id, run_id=task.stage_run_id
                )
                project = project_store.load(
                    summary.project_id,
                    owner_user_id=None,
                )
                published = self.publication_store.load(
                    project_id=summary.project_id,
                    release_id=run.release_id,
                )
                node = _published_node(published, stage.node_id)
                if task_kind is not None and task_kind != f"echomind_{node.type}":
                    continue
                if match is not None:
                    logger.warning(
                        "Ambiguous GraphRun callback task: %s", provider_run_id
                    )
                    return None
                match = (project, published, run)
        if match is None:
            return None
        project, published, run = match
        is_correlated = conversation_id is not None and task_kind is not None
        return self.callback(
            project=project,
            published=published,
            run_id=run.run_id,
            provider_run_id=provider_run_id,
            status=status,
            observe_provider_status=is_correlated,
        )

    def pause(self, *, project: ProjectInfo, run_id: str) -> GraphRun:
        with self._lease(project.project.project_id):
            run = self.graph_run_store.load(
                project_id=project.project.project_id,
                run_id=run_id,
            )
            if run.status not in {"pending", "running", "waiting_external"}:
                return run
            for stage in self.stage_run_store.list_for_graph_run(
                project_id=run.project_id,
                graph_run_id=run.run_id,
            ):
                if stage.status in {"pending", "running", "waiting_external"}:
                    self.stage_run_store.save(
                        stage.model_copy(
                            update={
                                "status": "paused",
                                "message": f"{stage.node_id} paused.",
                            }
                        )
                    )
            return self.graph_run_store.save(
                run.model_copy(
                    update={
                        "status": "paused",
                        "started_at": run.started_at or utc_now(),
                        "current_stage_ids": [],
                    }
                )
            )

    def resume(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        run_id: str,
    ) -> GraphRun:
        with self._lease(project.project.project_id):
            run = self.graph_run_store.load(
                project_id=project.project.project_id,
                run_id=run_id,
            )
            self._validate_loaded_run(project=project, published=published, run=run)
            if run.status != "paused":
                return run
            for stage in self.stage_run_store.list_for_graph_run(
                project_id=run.project_id,
                graph_run_id=run.run_id,
            ):
                if stage.status != "paused":
                    continue
                restored = (
                    "waiting_external"
                    if any(
                        task.status in {"submitting", "submitted", "running", "unknown"}
                        for task in stage.task_runs
                    )
                    else "running"
                )
                self.stage_run_store.save(
                    stage.model_copy(
                        update={
                            "status": restored,
                            "message": f"{stage.node_id} resumed.",
                        }
                    )
                )
            self.graph_run_store.save(
                run.model_copy(update={"status": "running", "failure": None})
            )
        return self.advance(
            project=project,
            published=published,
            run_id=run_id,
        )

    def retry(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        run_id: str,
        inputs: dict[str, Any] | None = None,
    ) -> GraphRun:
        with self._lease(project.project.project_id):
            run = self.graph_run_store.load(
                project_id=project.project.project_id,
                run_id=run_id,
            )
            self._validate_loaded_run(project=project, published=published, run=run)
            if run.status != "paused":
                raise RuntimeSchedulerError("Only paused GraphRuns can be retried")

            stages = self.stage_run_store.list_for_graph_run(
                project_id=run.project_id,
                graph_run_id=run.run_id,
            )
            failed_stages = [
                stage
                for stage in stages
                if stage.status == "failed"
                and (not run.current_stage_ids or stage.run_id in run.current_stage_ids)
            ]
            if not failed_stages:
                raise RuntimeSchedulerError(
                    "GraphRun has no failed StageRun; resume it instead"
                )
            failed_stage = max(
                failed_stages,
                key=lambda stage: (stage.attempt, stage.created_at),
            )
            node = _published_node(published, failed_stage.node_id)
            next_inputs = dict(run.inputs if inputs is None else inputs)
            resumable_task = _credential_blocked_task(failed_stage)
            if next_inputs != run.inputs:
                if resumable_task is not None:
                    raise RuntimeSchedulerError(
                        "Retry cannot change inputs while a platform task "
                        "may be active."
                    )
                _validate_retry_inputs(published, stages, run.inputs, next_inputs)
            try:
                _, input_refs = resolve_stage_inputs(
                    node=node,
                    run_inputs=next_inputs,
                    stages=stages,
                )
            except ValueError as exc:
                raise RuntimeSchedulerError(str(exc)) from exc

            attempt = (
                max(
                    stage.attempt
                    for stage in stages
                    if stage.node_id == failed_stage.node_id
                )
                + 1
            )
            retry_stage = self.stage_run_store.start(
                project=project,
                node=node,
                graph_run_id=run.run_id,
                mode="published_run",
                attempt=attempt,
                input_refs=input_refs,
            )
            if resumable_task is not None:
                now = utc_now()
                resumed_task = resumable_task.model_copy(
                    update={
                        "task_run_id": str(uuid4()),
                        "stage_run_id": retry_stage.run_id,
                        "status": "submitted",
                        "created_at": now,
                        "submitted_at": resumable_task.submitted_at or now,
                        "last_polled_at": None,
                        "finished_at": None,
                        "error": None,
                    }
                )
                self._save_task(run=run, task=resumed_task)
                self.stage_run_store.save(
                    retry_stage.model_copy(
                        update={
                            "status": "waiting_external",
                            "message": (
                                "Reconnected the existing platform task while "
                                "waiting for request credentials."
                            ),
                            "task_runs": [resumed_task],
                        }
                    )
                )
                self.graph_run_store.save(
                    run.model_copy(
                        update={
                            "status": "waiting_external",
                            "inputs": next_inputs,
                            "current_stage_ids": [retry_stage.run_id],
                            "failure": None,
                        }
                    )
                )
            else:
                self.graph_run_store.save(
                    run.model_copy(
                        update={
                            "status": "running",
                            "inputs": next_inputs,
                            "current_stage_ids": [],
                            "failure": None,
                        }
                    )
                )
        return self.advance(
            project=project,
            published=published,
            run_id=run_id,
        )

    def cancel(self, *, project: ProjectInfo, run_id: str) -> GraphRun:
        with self._lease(project.project.project_id):
            run = self.graph_run_store.load(
                project_id=project.project.project_id,
                run_id=run_id,
            )
            if run.status not in ACTIVE_GRAPH_RUN_STATUSES:
                return run
            run = self.graph_run_store.save(
                run.model_copy(
                    update={
                        "status": "paused",
                        "started_at": run.started_at or utc_now(),
                    }
                )
            )
            published = self.publication_store.load(
                project_id=run.project_id, release_id=run.release_id
            )
            stages = self.stage_run_store.list_for_graph_run(
                project_id=run.project_id,
                graph_run_id=run.run_id,
            )
            for node in published.nodes:
                stage = _latest_stage_for_node(stages, node.id)
                if stage is None or stage.status in {"succeeded", "canceled"}:
                    continue
                task = _latest_active_task(stage) or _credential_blocked_task(stage)
                if task is not None:
                    try:
                        self._cancel_task(published, run, stage, task)
                    except Exception as exc:
                        return self._pause_run(
                            run=run,
                            stage=stage,
                            category="canceled",
                            message=f"Platform cancellation could not be confirmed: "
                            f"{type(exc).__name__}: {exc}"[:4000],
                        )
                    canceled = task.model_copy(
                        update={
                            "status": "canceled",
                            "finished_at": utc_now(),
                            "error": None,
                        }
                    )
                    stage = self._replace_task(stage, canceled)
                    self._save_task(run=run, task=canceled)
                elif stage.status == "failed":
                    continue
                self.stage_run_store.save(
                    stage.model_copy(
                        update={
                            "status": "canceled",
                            "message": f"{stage.node_id} canceled.",
                            "finished_at": utc_now(),
                            "error": None,
                        }
                    )
                )
            return self.graph_run_store.save(
                run.model_copy(
                    update={
                        "status": "canceled",
                        "outcome": "canceled",
                        "current_stage_ids": [],
                        "failure": None,
                        "finished_at": utc_now(),
                    }
                )
            )

    def _cancel_task(
        self,
        published: PublishedGraph,
        run: GraphRun,
        stage: StageRun,
        task: TaskRun,
    ) -> None:
        if task.status == "submitting" and task.provider_run_id is None:
            return
        context = self._resolved_stage_context(published, run, stage)
        executor = self.executor_registry.get(stage.executor)
        result = executor.cancel(context, task)
        if result.status != "canceled":
            raise RuntimeSchedulerError(result.message)

    def recover(self, *, project: ProjectInfo, run_id: str) -> GraphRun:
        run = self.graph_run_store.load(
            project_id=project.project.project_id, run_id=run_id
        )
        published = self.publication_store.load(
            project_id=run.project_id, release_id=run.release_id
        )
        return self.advance(project=project, published=published, run_id=run_id)

    def recover_all(self) -> list[GraphRun]:
        if not self.projects_root.is_dir():
            return []
        project_store = ProjectStore(self.projects_root)
        recovered: list[GraphRun] = []
        for project_dir in sorted(self.projects_root.iterdir()):
            if not project_dir.is_dir():
                continue
            try:
                project = project_store.load(
                    project_dir.name,
                    owner_user_id=None,
                )
                run = self.graph_run_store.active(project.project.project_id)
                if run is None or run.status == "paused":
                    continue
                recovered.append(
                    self.recover(
                        project=project,
                        run_id=run.run_id,
                    )
                )
            except (OSError, ValueError, FileNotFoundError, RuntimeSchedulerError):
                continue
        return recovered

    def dispatch_ready_batches(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        source: ReadyBatchSource,
        tracer_resource_id: str,
    ) -> list[GraphRun]:
        started: list[GraphRun] = []
        batches = sorted(
            source.list_ready_batches(
                project_id=project.project.project_id,
                tracer_resource_id=tracer_resource_id,
            ),
            key=lambda batch: batch.batch_id,
        )
        for batch in batches:
            if batch.tracer_resource_id != tracer_resource_id:
                continue
            if self.graph_run_store.active(project.project.project_id) is not None:
                break
            existing = self.graph_run_store.find_by_batch_key(
                project_id=project.project.project_id,
                tracer_resource_id=batch.tracer_resource_id,
                batch_id=batch.batch_id,
            )
            if existing is not None:
                continue
            try:
                run = self.start(
                    project=project,
                    published=published,
                    request_id=f"tracer:{batch.tracer_resource_id}:{batch.batch_id}",
                    inputs=batch.inputs,
                    trigger="tracer_batch",
                    source_tracer_resource_id=batch.tracer_resource_id,
                    source_batch_id=batch.batch_id,
                )
            except ActiveGraphRunExistsError:
                break
            started.append(run)
        return started

    def _validate_start(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        execution_mode: GraphRunExecutionMode,
        require_runtime_slot: bool,
    ) -> None:
        deployments = FileDeploymentStore(self.projects_root).list(
            project.project.project_id
        )
        if any(item.status not in TERMINAL_DEPLOYMENT_STATUSES for item in deployments):
            raise RuntimeSchedulerError("A model deployment is still in progress")
        if published.project_id != project.project.project_id:
            raise RuntimeSchedulerError("PublishedGraph belongs to another Project")
        if execution_mode == "debug":
            if project.project.lifecycle != "development":
                raise RuntimeSchedulerError(
                    "Debug GraphRuns require a development Project"
                )
            if published.publication_mode != "debug" or require_runtime_slot:
                raise RuntimeSchedulerError(
                    "Debug GraphRuns require a debug snapshot and a manual trigger"
                )
            saved = self.publication_store.load(
                project_id=published.project_id, release_id=published.release_id
            )
            if saved.content_sha256 != published.content_sha256:
                raise RuntimeSchedulerError("Debug snapshot digest changed")
            return
        if project.project.lifecycle != "frozen":
            raise RuntimeSchedulerError("Project must be published before GraphRun")
        if project.project.active_release_id != published.release_id:
            raise RuntimeSchedulerError("PublishedGraph is not the active release")
        if not require_runtime_slot:
            return
        slot = self.runtime_slot_store.load_optional(project.project.project_id)
        if (
            slot is None
            or slot.status != "ready"
            or slot.active_release_id != published.release_id
        ):
            raise RuntimeSchedulerError(
                "RuntimeSlot is not ready for this published release"
            )

    def _advance_locked(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        run_id: str,
    ) -> GraphRun:
        run = self.graph_run_store.load(
            project_id=project.project.project_id,
            run_id=run_id,
        )
        self._validate_loaded_run(project=project, published=published, run=run)
        if run.status in {"succeeded", "failed", "canceled", "paused"}:
            return run
        if run.status == "pending":
            run = self.graph_run_store.save(
                run.model_copy(update={"status": "running", "started_at": utc_now()})
            )

        while True:
            stages = self.stage_run_store.list_for_graph_run(
                project_id=run.project_id,
                graph_run_id=run.run_id,
            )
            waiting = next(
                (stage for stage in stages if stage.status == "waiting_external"),
                None,
            )
            if waiting is not None:
                polled = self._poll_waiting_stage(
                    published=published,
                    run=run,
                    stage=waiting,
                )
                if polled is not None:
                    return polled
                continue

            node = self._next_node(published=published, stages=stages)
            if node is None:
                return self._finalize(project=project, run=run, stages=stages)

            stage = _latest_stage_for_node(stages, node.id)
            if stage is None:
                try:
                    inputs, input_refs = resolve_stage_inputs(
                        node=node,
                        run_inputs=run.inputs,
                        stages=stages,
                    )
                except ValueError as exc:
                    return self._pause_run(
                        run=run,
                        stage=None,
                        category="validation",
                        message=str(exc),
                    )
                stage = self.stage_run_store.start(
                    project=project,
                    node=node,
                    graph_run_id=run.run_id,
                    mode="published_run",
                    attempt=1,
                    input_refs=input_refs,
                )
            else:
                try:
                    inputs, _ = resolve_stage_inputs(
                        node=node,
                        run_inputs=run.inputs,
                        stages=stages,
                    )
                except ValueError as exc:
                    return self._pause_run(
                        run=run,
                        stage=stage,
                        category="validation",
                        message=str(exc),
                    )

            if stage.status == "succeeded":
                continue
            if stage.status in {"failed", "canceled"}:
                return self._pause_run(
                    run=run,
                    stage=stage,
                    category="external",
                    message=stage.error or stage.message,
                )

            active_task = _latest_active_task(stage)
            if active_task is not None:
                stage = self.stage_run_store.save(
                    stage.model_copy(update={"status": "waiting_external"})
                )
                run = self.graph_run_store.save(
                    run.model_copy(
                        update={
                            "status": "waiting_external",
                            "current_stage_ids": [stage.run_id],
                        }
                    )
                )
                if active_task.provider_run_id is None:
                    return self._submit_task(run=run, stage=stage, task=active_task)
                return (
                    self._poll_waiting_stage(
                        published=published,
                        run=run,
                        stage=stage,
                    )
                    or run
                )

            context = self._stage_context(published, run, stage, inputs)
            executor = self.executor_registry.get(stage.executor)
            try:
                prepared = executor.prepare(context)
            except Exception as exc:
                return self._pause_run(
                    run=run,
                    stage=stage,
                    category="infrastructure",
                    message=f"{type(exc).__name__}: {exc}",
                )
            if prepared.status == "completed":
                try:
                    self._complete_stage(
                        stage=stage,
                        node=node,
                        message=prepared.message,
                        artifacts=prepared.artifacts,
                    )
                except Exception as exc:
                    return self._pause_run(
                        run=run,
                        stage=stage,
                        category="validation",
                        message=f"{type(exc).__name__}: {exc}",
                    )
                run = self.graph_run_store.save(
                    run.model_copy(
                        update={"status": "running", "current_stage_ids": []}
                    )
                )
                continue
            if prepared.status == "failed":
                failed_stage = self._fail_stage(
                    stage=stage,
                    category=prepared.failure_category,
                    message=prepared.error or prepared.message,
                )
                return self._pause_run(
                    run=run,
                    stage=failed_stage,
                    category=prepared.failure_category,
                    message=prepared.error or prepared.message,
                )
            if prepared.external_task is None:
                raise RuntimeSchedulerError(
                    "External stage preparation is missing task intent"
                )
            task = self._new_task(
                run=run,
                stage=stage,
                intent=prepared.external_task,
                inputs=inputs,
            )
            stage = self.stage_run_store.save(
                stage.model_copy(update={"task_runs": [*stage.task_runs, task]})
            )
            run = self.graph_run_store.save(
                run.model_copy(
                    update={
                        "status": "waiting_external",
                        "current_stage_ids": [stage.run_id],
                    }
                )
            )
            return self._submit_task(run=run, stage=stage, task=task)

    def _poll_waiting_stage(
        self,
        *,
        published: PublishedGraph,
        run: GraphRun,
        stage: StageRun,
    ) -> GraphRun | None:
        task = _latest_active_task(stage)
        if task is None:
            self._fail_stage(
                stage=stage,
                category="infrastructure",
                message="Waiting StageRun has no active TaskRun.",
            )
            return self._pause_run(
                run=run,
                stage=stage,
                category="infrastructure",
                message="Waiting StageRun has no active TaskRun.",
            )
        if task.provider_run_id is None:
            return self._submit_task(run=run, stage=stage, task=task)
        context = self._resolved_stage_context(published, run, stage)
        executor = self.executor_registry.get(stage.executor)
        try:
            polled = executor.poll(context, task)
        except RuntimeCredentialUnavailableError as exc:
            return self._wait_for_task(
                run=run,
                stage=stage,
                task=task,
                message=(
                    f"{stage.node_id} is waiting for request credentials "
                    f"to poll the platform task: {exc}"
                ),
            )
        except Exception as exc:
            pending = task.model_copy(
                update={
                    "error": f"{type(exc).__name__}: {exc}"[:4000],
                    "last_polled_at": utc_now(),
                }
            )
            return self._wait_for_task(
                run=run,
                stage=stage,
                task=pending,
                message=f"Platform task status could not be verified: {pending.error}",
            )
        if polled.status == "running":
            running = task.model_copy(
                update={
                    "status": "running",
                    "last_polled_at": utc_now(),
                    "error": None,
                }
            )
            return self._wait_for_task(
                run=run, stage=stage, task=running, message=polled.message
            )
        if polled.status == "failed":
            failed = task.model_copy(
                update={
                    "status": "failed",
                    "error": polled.message,
                    "last_polled_at": utc_now(),
                    "finished_at": utc_now(),
                }
            )
            self._save_task(run=run, task=failed)
            stage = self._replace_task(stage, failed)
            failed_stage = self._fail_stage(
                stage=stage,
                category=polled.failure_category,
                message=polled.message,
            )
            return self._pause_run(
                run=run,
                stage=failed_stage,
                category=polled.failure_category,
                message=failed_stage.error or polled.message,
            )
        succeeded = task.model_copy(
            update={
                "status": "succeeded",
                "last_polled_at": utc_now(),
                "finished_at": utc_now(),
                "error": None,
            }
        )
        self._save_task(run=run, task=succeeded)
        stage = self._replace_task(stage, succeeded)
        try:
            self._complete_stage(
                stage=stage,
                node=_published_node(published, stage.node_id),
                message=polled.message,
                artifacts=polled.artifacts,
            )
        except Exception as exc:
            return self._pause_run(
                run=run,
                stage=stage,
                category="validation",
                message=f"{type(exc).__name__}: {exc}",
            )
        return None

    def _submit_task(
        self,
        *,
        run: GraphRun,
        stage: StageRun,
        task: TaskRun,
    ) -> GraphRun:
        context = self._resolved_stage_context(
            self.publication_store.load(
                project_id=run.project_id,
                release_id=run.release_id,
            ),
            run,
            stage,
        )
        executor = self.executor_registry.get(stage.executor)
        # A restart at the submission boundary must reconcile before creating again.
        uncertain = task.model_copy(update={"status": "unknown"})
        self._wait_for_task(run=run, stage=stage, task=uncertain, message=stage.message)
        try:
            submitted = executor.submit(context, task)
        except RuntimeCredentialUnavailableError as exc:
            return self._wait_for_task(
                run=run,
                stage=stage,
                task=task,
                message=(
                    f"{stage.node_id} is waiting for request credentials "
                    f"to submit the platform task: {exc}"
                ),
            )
        except RuntimeSubmissionNotAttemptedError as exc:
            return self._wait_for_task(
                run=run,
                stage=stage,
                task=task,
                message=f"Platform task submission could not start: {exc}",
            )
        except Exception as exc:
            uncertain = uncertain.model_copy(
                update={"error": f"{type(exc).__name__}: {exc}"[:4000]}
            )
            return self._wait_for_task(
                run=run,
                stage=stage,
                task=uncertain,
                message=f"Platform task submission is unconfirmed: {uncertain.error}",
            )
        now = utc_now()
        if submitted.status == "failed":
            failed = task.model_copy(
                update={
                    "status": "failed",
                    "error": submitted.message,
                    "finished_at": now,
                }
            )
            self._save_task(run=run, task=failed)
            stage = self._replace_task(stage, failed)
            failed_stage = self._fail_stage(
                stage=stage,
                category=submitted.failure_category,
                message=submitted.message,
            )
            return self._pause_run(
                run=run,
                stage=failed_stage,
                category=submitted.failure_category,
                message=failed_stage.error or submitted.message,
            )
        submitted_task = task.model_copy(
            update={
                "status": "submitted",
                "provider_run_id": submitted.provider_run_id,
                "submitted_at": now,
                "last_polled_at": now,
                "error": None,
            }
        )
        return self._wait_for_task(
            run=run, stage=stage, task=submitted_task, message=submitted.message
        )

    def _wait_for_task(
        self,
        *,
        run: GraphRun,
        stage: StageRun,
        task: TaskRun,
        message: str,
    ) -> GraphRun:
        stage = self._replace_task(stage, task).model_copy(
            update={
                "status": "waiting_external",
                "message": message[:4000],
            }
        )
        self.stage_run_store.save(stage)
        self._save_task(run=run, task=task)
        return self.graph_run_store.save(
            run.model_copy(
                update={
                    "status": "waiting_external",
                    "current_stage_ids": [stage.run_id],
                }
            )
        )

    def _new_task(
        self,
        *,
        run: GraphRun,
        stage: StageRun,
        intent: ExternalTaskIntent,
        inputs: dict[str, Any],
    ) -> TaskRun:
        return TaskRun(
            task_run_id=str(uuid4()),
            stage_run_id=stage.run_id,
            provider=intent.provider,
            idempotency_key=(
                f"{run.run_id}:{stage.run_id}:{stage.attempt}:{intent.provider}"
            ),
            request_digest=canonical_sha256(
                {
                    "project_id": run.project_id,
                    "release_id": run.release_id,
                    "node_id": stage.node_id,
                    "attempt": stage.attempt,
                    "inputs": inputs,
                    "implementation_digest": stage.implementation_digest,
                }
            ),
        )

    def _replace_task(self, stage: StageRun, task: TaskRun) -> StageRun:
        return stage.model_copy(
            update={
                "task_runs": [
                    task if item.task_run_id == task.task_run_id else item
                    for item in stage.task_runs
                ]
            }
        )

    def _save_task(self, *, run: GraphRun, task: TaskRun) -> None:
        self.graph_run_store.save_task_run(
            project_id=run.project_id,
            graph_run_id=run.run_id,
            task_run=task,
        )

    def _complete_stage(
        self,
        *,
        stage: StageRun,
        node: ProjectNode,
        message: str,
        artifacts: list[RuntimeArtifact],
    ) -> StageRun:
        project_artifacts = []
        output_refs = []
        by_port = {item.port: item for item in artifacts}
        if len(by_port) != len(artifacts):
            raise RuntimeSchedulerError("Stage artifacts must use unique output ports")
        missing = [
            port
            for port, output in node.outputs.items()
            if output.required and port not in by_port
        ]
        if missing:
            raise RuntimeSchedulerError(
                "Stage did not produce required outputs: " + ", ".join(missing)
            )
        for item in artifacts:
            output = node.outputs.get(item.port)
            if output is None:
                raise RuntimeSchedulerError(
                    f"Unknown stage output {stage.node_id}.{item.port}"
                )
            if item.payload.kind != output.artifact_type:
                raise RuntimeSchedulerError(
                    f"Artifact kind {item.payload.kind} does not match "
                    f"{stage.node_id}.{item.port}"
                )
            artifact, _ = self.stage_run_store.write_artifact(stage, item.payload)
            project_artifacts.append(artifact)
            output_refs.append(
                ArtifactRef(
                    artifact_type=output.artifact_type,
                    uri=artifact.uri,
                    content_sha256=f"sha256:{artifact.content_sha256}",
                    metadata={
                        **item.metadata,
                        "artifact_id": artifact.artifact_id,
                        "port": item.port,
                    },
                )
            )
        return self.stage_run_store.save(
            stage.model_copy(
                update={
                    "status": "succeeded",
                    "message": message,
                    "artifacts": [*stage.artifacts, *project_artifacts],
                    "output_refs": output_refs,
                    "finished_at": utc_now(),
                }
            )
        )

    def _fail_stage(
        self,
        *,
        stage: StageRun,
        category: FailureCategory,
        message: str,
    ) -> StageRun:
        return self.stage_run_store.save(
            stage.model_copy(
                update={
                    "status": "failed",
                    "message": f"{stage.node_id} failed.",
                    "error": f"{category}: {message}"[:4000],
                    "finished_at": utc_now(),
                }
            )
        )

    def _pause_run(
        self,
        *,
        run: GraphRun,
        stage: StageRun | None,
        category: FailureCategory,
        message: str,
    ) -> GraphRun:
        failure = RuntimeFailure(
            category=category,
            message=message[:4000],
            details={"node_id": stage.node_id} if stage else {},
        )
        handoff = _repair_handoff_for_failure(run, stage, failure)
        handoffs = run.repair_handoffs
        if handoff is not None and all(
            item.handoff_id != handoff.handoff_id for item in handoffs
        ):
            handoffs = [*handoffs, handoff]
        return self.graph_run_store.save(
            run.model_copy(
                update={
                    "status": "paused",
                    "current_stage_ids": [stage.run_id] if stage else [],
                    "failure": failure,
                    "repair_handoffs": handoffs,
                }
            )
        )

    def _finalize(
        self,
        *,
        project: ProjectInfo,
        run: GraphRun,
        stages: list[StageRun],
    ) -> GraphRun:
        try:
            if run.execution_mode == "debug":
                result = RuntimeFinalization(
                    outcome="baseline_retained",
                    promotion_result={
                        "decision": "baseline_retained",
                        "reason": "debug_run",
                    },
                )
            else:
                result = self.finalizer.finalize(
                    project_id=project.project.project_id,
                    graph_run=run,
                    stages=stages,
                )
        except Exception as exc:
            return self._pause_run(
                run=run,
                stage=None,
                category="infrastructure",
                message=f"{type(exc).__name__}: {exc}",
            )
        return self.graph_run_store.save(
            run.model_copy(
                update={
                    "status": "succeeded",
                    "outcome": result.outcome,
                    "promotion_result": result.promotion_result,
                    "current_stage_ids": [],
                    "finished_at": utc_now(),
                }
            )
        )

    def _next_node(
        self,
        *,
        published: PublishedGraph,
        stages: list[StageRun],
    ) -> ProjectNode | None:
        succeeded = {stage.node_id for stage in stages if stage.status == "succeeded"}
        for node in _topological_nodes(published):
            if node.id in succeeded:
                continue
            predecessors = _predecessors(published, node.id)
            if predecessors.issubset(succeeded):
                return node
        return None

    def _stage_context(
        self,
        published: PublishedGraph,
        run: GraphRun,
        stage: StageRun,
        inputs: dict[str, Any],
    ) -> RuntimeStageContext:
        return RuntimeStageContext(
            project_id=run.project_id,
            release_id=run.release_id,
            graph_run_id=run.run_id,
            stage_run_id=stage.run_id,
            node=_published_node(published, stage.node_id),
            inputs=inputs,
        )

    def _resolved_stage_context(
        self,
        published: PublishedGraph,
        run: GraphRun,
        stage: StageRun,
    ) -> RuntimeStageContext:
        node = _published_node(published, stage.node_id)
        stages = self.stage_run_store.list_for_graph_run(
            project_id=run.project_id,
            graph_run_id=run.run_id,
        )
        inputs, _ = resolve_stage_inputs(
            node=node,
            run_inputs=run.inputs,
            stages=stages,
        )
        return self._stage_context(published, run, stage, inputs)

    def _validate_loaded_run(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        run: GraphRun,
    ) -> None:
        if run.project_id != project.project.project_id:
            raise RuntimeSchedulerError("GraphRun belongs to another Project")
        if run.project_id != published.project_id:
            raise RuntimeSchedulerError("PublishedGraph belongs to another Project")
        if run.execution_mode == "debug" and published.publication_mode != "debug":
            raise RuntimeSchedulerError("Debug GraphRun must use its debug snapshot")
        if run.release_id != published.release_id:
            raise RuntimeSchedulerError("GraphRun release changed")
        if run.release_digest != published.content_sha256:
            raise RuntimeSchedulerError("GraphRun release digest changed")

    @contextmanager
    def _lease(self, project_id: str) -> Iterator[None]:
        try:
            with ProjectLease(
                self.projects_root,
                project_id,
                timeout_seconds=self.lease_timeout_seconds,
            ).acquire():
                yield
        except ProjectLeaseTimeout as exc:
            raise RuntimeSchedulerError(str(exc)) from exc


class RuntimeGraphPoller:
    def __init__(
        self,
        scheduler: RuntimeGraphScheduler,
        *,
        interval_seconds: float = 15,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.scheduler = scheduler
        self.interval_seconds = interval_seconds
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run(),
            name="echomind-runtime-graph-poller",
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.scheduler.recover_all)
            except Exception:
                logger.exception("EchoMind runtime graph polling failed")
            await asyncio.sleep(self.interval_seconds)


def _repair_handoff_for_failure(
    run: GraphRun, stage: StageRun | None, failure: RuntimeFailure
) -> GraphRunRepairHandoff | None:
    if (
        run.execution_mode != "debug"
        or stage is None
        or stage.status != "failed"
        or failure.category == "canceled"
        or any(task.status not in _TERMINAL_TASK_STATUSES for task in stage.task_runs)
    ):
        return None
    task = next(
        (task for task in reversed(stage.task_runs) if task.provider_run_id),
        None,
    )
    return GraphRunRepairHandoff(
        handoff_id=str(uuid5(UUID(run.run_id), f"failure:{stage.run_id}")),
        project_id=run.project_id,
        run_id=run.run_id,
        release_id=run.release_id,
        release_digest=run.release_digest,
        stage_run_id=stage.run_id,
        node_id=stage.node_id,
        attempt=stage.attempt,
        provider_task_id=task.provider_run_id if task else None,
        failure=failure,
    )


def _predecessors(published: PublishedGraph, node_id: str) -> set[str]:
    return {edge.source for edge in published.edges if edge.target == node_id}


def resolve_stage_inputs(
    *,
    node: ProjectNode,
    run_inputs: dict[str, Any],
    stages: list[StageRun],
) -> tuple[dict[str, Any], list[str]]:
    resolved: dict[str, Any] = {}
    input_refs: list[str] = []
    for name, node_input in node.inputs.items():
        value = _resolve_input(
            node_input=node_input,
            run_inputs=run_inputs,
            stages=stages,
        )
        if value is None:
            if node_input.required:
                raise ValueError(f"Missing required input: {node.id}.{name}")
            continue
        resolved[name] = value
        ref = _artifact_ref_uri(value)
        if ref is not None and ref not in input_refs:
            input_refs.append(ref)
    return resolved, input_refs


def _validate_retry_inputs(
    published: PublishedGraph,
    stages: list[StageRun],
    old_inputs: dict[str, Any],
    new_inputs: dict[str, Any],
) -> None:
    succeeded = {stage.node_id for stage in stages if stage.status == "succeeded"}
    for node in published.nodes:
        if node.id not in succeeded:
            continue
        for node_input in node.inputs.values():
            binding = node_input.binding
            if (
                binding.kind in {"run_input", "runtime_slot"}
                and binding.name is not None
                and old_inputs.get(binding.name) != new_inputs.get(binding.name)
            ):
                raise RuntimeSchedulerError(
                    f"Retry cannot change {binding.name}, already consumed by "
                    f"{node.id}. Start a new GraphRun to use different inputs."
                )


def _resolve_input(
    *,
    node_input: ProjectNodeInput,
    run_inputs: dict[str, Any],
    stages: list[StageRun],
) -> Any:
    binding = node_input.binding
    if binding.kind == "literal":
        return binding.value
    if binding.kind in {"run_input", "runtime_slot"}:
        return run_inputs.get(binding.name or "")
    source_stage = _latest_stage_for_node(stages, binding.node_id or "")
    if source_stage is None or source_stage.status != "succeeded":
        return None
    for artifact in source_stage.output_refs:
        if (
            artifact.artifact_type == node_input.artifact_type
            and artifact.metadata.get("port") == binding.port
        ):
            return artifact.model_dump(mode="json")
    return None


def _artifact_ref_uri(value: Any) -> str | None:
    if isinstance(value, str) and ("://" in value or value.startswith("/")):
        return value
    if isinstance(value, dict):
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            source_uri = metadata.get("source_uri")
            if isinstance(source_uri, str):
                return source_uri
        uri = value.get("uri")
        if isinstance(uri, str) and ("://" in uri or uri.startswith("/")):
            return uri
    return None


def _topological_nodes(published: PublishedGraph) -> list[ProjectNode]:
    nodes_by_id = {node.id: node for node in published.nodes}
    indegree = {node.id: 0 for node in published.nodes}
    adjacency = {node.id: [] for node in published.nodes}
    for edge in published.edges:
        indegree[edge.target] += 1
        adjacency[edge.source].append(edge.target)
    ready = [node.id for node in published.nodes if indegree[node.id] == 0]
    ordered: list[ProjectNode] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(nodes_by_id[node_id])
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(ordered) != len(published.nodes):
        raise RuntimeSchedulerError("PublishedGraph is not acyclic")
    return ordered


def _latest_stage_for_node(
    stages: list[StageRun],
    node_id: str,
) -> StageRun | None:
    candidates = [stage for stage in stages if stage.node_id == node_id]
    if not candidates:
        return None
    return max(candidates, key=lambda stage: (stage.attempt, stage.created_at))


def _latest_active_task(stage: StageRun) -> TaskRun | None:
    active = [
        task
        for task in stage.task_runs
        if task.status in {"submitting", "submitted", "running", "unknown"}
    ]
    return max(active, key=lambda task: task.created_at) if active else None


def _credential_blocked_task(stage: StageRun) -> TaskRun | None:
    candidates = [
        task
        for task in stage.task_runs
        if task.status == "failed"
        and task.provider_run_id
        and task.error
        and (
            "request credentials" in task.error
            or "PYROMIND_RUNTIME_API_KEY" in task.error
        )
    ]
    return max(candidates, key=lambda task: task.created_at) if candidates else None


def _published_node(published: PublishedGraph, node_id: str) -> ProjectNode:
    node = next((item for item in published.nodes if item.id == node_id), None)
    if node is None:
        raise RuntimeSchedulerError(
            f"StageRun node is not in PublishedGraph: {node_id}"
        )
    return node


def _conversation_id(value: str) -> str:
    try:
        return UUID(value).hex
    except ValueError:
        return value
