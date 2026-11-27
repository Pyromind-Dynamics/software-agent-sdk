import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import Barrier
from unittest.mock import Mock
from uuid import uuid4

import pytest

from pyromind_agent_server.echomind_dataset_runtime import DatasetRuntimeExecutor
from pyromind_agent_server.echomind_evaluator_runtime import (
    EvaluatorRuntimeExecutor,
    EvaluatorTaskState,
)
from pyromind_agent_server.echomind_graph_runs import RuntimeFailure
from pyromind_agent_server.echomind_projects import (
    ProjectEdge,
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectNodeOutput,
    ProjectRecord,
    ProjectStore,
)
from pyromind_agent_server.echomind_publication import (
    PromotionPolicy,
    PublicationStore,
    PublishedGraph,
    RuntimeRequirements,
)
from pyromind_agent_server.echomind_runtime_contracts import (
    ExternalTaskIntent,
    RuntimeCredentialUnavailableError,
    StageCancellationResult,
    StagePollResult,
    StagePreparationResult,
    StageSubmissionResult,
)
from pyromind_agent_server.echomind_runtime_scheduler import (
    ReadyBatch,
    RuntimeGraphScheduler,
    RuntimeSchedulerError,
)
from pyromind_agent_server.echomind_runtime_slots import (
    RuntimeHealth,
    RuntimeSlot,
    RuntimeSlotStore,
    ServingInferenceRuntime,
    TracerBufferConfig,
    TracerRuntime,
)
from pyromind_agent_server.echomind_storage import canonical_sha256
from pyromind_agent_server.echomind_train_runtime import (
    TrainRuntimeExecutor,
    TrainTaskState,
)
from pyromind_runtime.domain.echomind_deployments import (
    DeploymentRecord,
    DeploymentRequest,
    DeploymentTarget,
)
from pyromind_runtime.infrastructure.file_deployment_store import FileDeploymentStore


def test_tracer_batch_requires_ready_runtime_slot(tmp_path):
    project, published = _published_project(tmp_path / "projects")
    scheduler = RuntimeGraphScheduler(projects_root=tmp_path / "projects")

    with pytest.raises(RuntimeSchedulerError, match="RuntimeSlot"):
        scheduler.start(
            project=project,
            published=published,
            request_id="tracer:tracer-1:batch-1",
            inputs=_run_inputs("batch-1"),
            trigger="tracer_batch",
            source_tracer_resource_id="tracer-1",
            source_batch_id="batch-1",
        )


def test_dispatch_ready_batches_is_idempotent_after_success(tmp_path):
    project, published = _published_project(
        tmp_path / "projects",
        with_runtime_slot=True,
    )
    scheduler = RuntimeGraphScheduler(projects_root=tmp_path / "projects")
    source = _ReadyBatchSource(
        [
            ReadyBatch(
                tracer_resource_id="tracer-1",
                batch_id="batch-2",
                inputs=_run_inputs("batch-2"),
            ),
            ReadyBatch(
                tracer_resource_id="tracer-1",
                batch_id="batch-1",
                inputs=_run_inputs("batch-1"),
            ),
        ]
    )

    first = scheduler.dispatch_ready_batches(
        project=project,
        published=published,
        source=source,
        tracer_resource_id="tracer-1",
    )
    repeated = scheduler.dispatch_ready_batches(
        project=project,
        published=published,
        source=source,
        tracer_resource_id="tracer-1",
    )

    assert [run.source_batch_id for run in first] == ["batch-1", "batch-2"]
    assert all(run.status == "succeeded" for run in first)
    assert repeated == []
    assert len(scheduler.graph_run_store.list(project.project.project_id)) == 2


def test_train_runtime_waits_recovers_and_resumes_graph(tmp_path):
    projects_root = tmp_path / "projects"
    project, published = _published_project(
        projects_root,
        train_runtime=True,
    )
    backend = _SchedulerTrainBackend()
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={
            "train_runtime": TrainRuntimeExecutor(
                projects_root=str(projects_root),
                backend=backend,
            )
        },
    )

    started = scheduler.start(
        project=project,
        published=published,
        request_id="manual-train-runtime",
        inputs=_run_inputs("batch-1"),
    )

    assert started.status == "waiting_external"
    assert started.execution_mode == "published"
    assert backend.submit_count == 1
    train_stage = _stage_for_node(scheduler, project, started.run_id, "train")
    assert train_stage.status == "waiting_external"
    assert [task.status for task in train_stage.task_runs] == ["submitted"]

    running = scheduler.callback_by_provider_task(
        provider_run_id="train-task-1",
        status="Succeeded",
    )

    assert running is not None
    assert running.status == "waiting_external"
    train_stage = _stage_for_node(scheduler, project, started.run_id, "train")
    assert [task.status for task in train_stage.task_runs] == ["running"]
    assert backend.submit_count == 1

    completed = scheduler.callback_by_provider_task(
        provider_run_id="train-task-1",
        status="Succeeded",
    )

    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.outcome == "candidate_recorded"
    assert completed.promotion_result["decision"] == "awaiting_user_selection"
    assert backend.submit_count == 1
    stages = scheduler.stage_run_store.list_for_graph_run(
        project_id=project.project.project_id,
        graph_run_id=started.run_id,
    )
    assert [stage.node_id for stage in stages] == ["dataset", "train", "evaluator"]
    assert all(stage.status == "succeeded" for stage in stages)
    train_stage = next(stage for stage in stages if stage.node_id == "train")
    assert [task.status for task in train_stage.task_runs] == ["succeeded"]


def test_debug_run_recovers_original_project_without_publishing(tmp_path):
    projects_root = tmp_path / "projects"
    project, snapshot = _published_project(
        projects_root, train_runtime=True, development=True
    )
    backend = _SchedulerTrainBackend()
    executor = TrainRuntimeExecutor(projects_root=str(projects_root), backend=backend)
    finalizer = Mock()
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={"train_runtime": executor},
        finalizer=finalizer,
    )
    request_id = "debug-original-project"
    inputs = _run_inputs("batch-1")
    with pytest.raises(RuntimeSchedulerError, match="must be published"):
        scheduler.start(
            project=project, published=snapshot, request_id=request_id, inputs=inputs
        )

    started = scheduler.start(
        project=project,
        published=snapshot,
        request_id=request_id,
        inputs=inputs,
        execution_mode="debug",
    )
    assert started.status == "waiting_external"
    assert started.execution_mode == "debug"
    assert started.project_id == project.project.project_id
    assert started.release_id == snapshot.release_id
    assert started.release_digest == snapshot.content_sha256
    with pytest.raises(FileNotFoundError):
        PublicationStore(projects_root).load_active(started.project_id)

    restarted = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={"train_runtime": executor},
        finalizer=finalizer,
    )
    [recovered] = restarted.recover_all()
    assert recovered.run_id == started.run_id
    completed = restarted.callback_by_provider_task(
        provider_run_id="train-task-1",
        status="Succeeded",
        conversation_id=project.project.conversation_id,
        task_kind="echomind_train",
    )
    assert completed is not None
    assert completed.run_id == started.run_id
    assert completed.status == "succeeded"
    assert completed.promotion_result["reason"] == "debug_run"
    assert backend.submit_count == 1
    finalizer.finalize.assert_not_called()
    persisted_project = ProjectStore(projects_root).load(
        started.project_id, owner_user_id=project.project.owner_user_id
    )
    assert persisted_project == project


def test_debug_repair_persists_handoff_and_reuses_verified_ancestors(
    tmp_path, monkeypatch
):
    root = tmp_path / "projects"
    project, original_snapshot = _published_project(
        root, evaluator_runtime=True, development=True
    )
    backend = _SchedulerEvaluatorBackend()
    monkeypatch.setattr(
        backend,
        "query_task",
        lambda **_: EvaluatorTaskState(status="failed", message="Missing Eval input"),
    )
    executor = EvaluatorRuntimeExecutor(projects_root=str(root), backend=backend)
    scheduler = RuntimeGraphScheduler(
        projects_root=root, executors={"evaluator_runtime": executor}
    )
    inputs = _run_inputs("batch-1")
    project_id = project.project.project_id
    original = scheduler.start(
        project=project,
        published=original_snapshot,
        request_id="original-debug",
        inputs=inputs,
        execution_mode="debug",
    )
    assert not scheduler.graph_run_store.list_pending_repair_handoffs(project_id)
    paused = scheduler.recover(project=project, run_id=original.run_id)
    assert paused.status == "paused"
    [handoff] = scheduler.graph_run_store.list_pending_repair_handoffs(project_id)
    assert handoff.node_id == "evaluator"
    assert handoff.provider_task_id == "evaluator-task-1"
    assert handoff.failure == paused.failure
    assert scheduler.recover(project=project, run_id=original.run_id) == paused

    # Existing persisted failures can reconstruct the same stable incident.
    scheduler.graph_run_store.save(paused.model_copy(update={"repair_handoffs": []}))
    assert (
        scheduler.get_repair_handoff(project_id=project_id, run_id=original.run_id)
        == handoff
    )
    scheduler.graph_run_store.mark_repair_handoff_delivered(
        project_id, original.run_id, handoff.handoff_id
    )
    assert not scheduler.graph_run_store.list_pending_repair_handoffs(project_id)
    original_stages = scheduler.stage_run_store.list_for_graph_run(
        project_id=project_id, graph_run_id=original.run_id
    )
    revised = _save_revised_evaluator(root, original_snapshot)
    with pytest.raises(RuntimeSchedulerError, match="changed debug snapshot"):
        scheduler.start_repair(
            project=project,
            published=original_snapshot,
            repair_of_run_id=original.run_id,
            request_id="repair-request",
            inputs=inputs,
        )

    def interrupted_create(**_):
        raise OSError("simulated interruption before writing the successor")

    with monkeypatch.context() as patch:
        patch.setattr(scheduler.graph_run_store, "create", interrupted_create)
        with pytest.raises(OSError, match="simulated interruption"):
            scheduler.start_repair(
                project=project,
                published=revised,
                repair_of_run_id=original.run_id,
                request_id="repair-request",
                inputs=inputs,
            )
    old = scheduler.graph_run_store.load(project_id=project_id, run_id=original.run_id)
    assert old.status == "failed"
    assert old.failure == paused.failure

    repaired = scheduler.start_repair(
        project=project,
        published=revised,
        repair_of_run_id=original.run_id,
        request_id="repair-request",
        inputs=inputs,
    )
    assert repaired.release_id == revised.release_id
    assert repaired.repair_of_run_id == original.run_id
    assert repaired.status == "waiting_external"
    assert backend.submit_count == 2
    for old_stage in original_stages:
        stage = _stage_for_node(scheduler, project, repaired.run_id, old_stage.node_id)
        if old_stage.status == "succeeded":
            assert stage.reused_from_stage_run_id == old_stage.run_id
            assert stage.output_refs == old_stage.output_refs
            assert stage.task_runs == []
        else:
            assert stage.reused_from_stage_run_id is None
    assert (
        scheduler.stage_run_store.list_for_graph_run(
            project_id=project_id, graph_run_id=original.run_id
        )
        == original_stages
    )
    repeated = scheduler.start_repair(
        project=project,
        published=revised,
        repair_of_run_id=original.run_id,
        request_id="repair-request",
        inputs=inputs,
    )
    assert repeated.run_id == repaired.run_id
    assert backend.submit_count == 2


def test_debug_repair_can_replace_definite_local_validation_failure(tmp_path):
    root = tmp_path / "projects"
    project, snapshot = _published_project(
        root, evaluator_runtime=True, development=True
    )
    original_executor = Mock()
    original_executor.prepare.return_value = StagePreparationResult(
        status="failed",
        message="Missing Eval input binding.",
        error="Missing Eval input binding.",
        failure_category="validation",
    )
    scheduler = RuntimeGraphScheduler(
        projects_root=root, executors={"evaluator_runtime": original_executor}
    )
    inputs = _run_inputs("batch-1")
    original = scheduler.start(
        project=project,
        published=snapshot,
        request_id="local-validation",
        inputs=inputs,
        execution_mode="debug",
    )
    assert original.status == "paused"
    [handoff] = scheduler.graph_run_store.list_pending_repair_handoffs(
        project.project.project_id
    )
    assert handoff.failure.category == "validation"
    assert handoff.provider_task_id is None
    original_executor.submit.assert_not_called()

    backend = _SchedulerEvaluatorBackend()
    scheduler = RuntimeGraphScheduler(
        projects_root=root,
        executors={
            "evaluator_runtime": EvaluatorRuntimeExecutor(
                projects_root=str(root), backend=backend
            )
        },
    )
    repaired = scheduler.start_repair(
        project=project,
        published=_save_revised_evaluator(root, snapshot),
        repair_of_run_id=original.run_id,
        request_id="local-validation-repair",
        inputs=inputs,
    )
    assert repaired.status == "waiting_external"
    assert backend.submit_count == 1


def _save_revised_evaluator(root, snapshot):
    evaluator = snapshot.nodes[-1]
    revised = PublishedGraph.build(
        release_id=str(uuid4()),
        project_id=snapshot.project_id,
        source_graph_id=snapshot.source_graph_id,
        source_graph_digest=snapshot.source_graph_digest,
        runtime_requirements=snapshot.runtime_requirements,
        promotion_policy=snapshot.promotion_policy,
        nodes=[
            *snapshot.nodes[:-1],
            evaluator.model_copy(
                update={"config": {**evaluator.config, "fixed": True}}
            ),
        ],
        edges=snapshot.edges,
        published_by="42",
    )
    return PublicationStore(root).save(revised)


def test_concurrent_start_reuses_request_id(tmp_path):
    projects_root = tmp_path / "projects"
    project, published = _published_project(projects_root, train_runtime=True)
    backend = _SchedulerTrainBackend()
    barrier = Barrier(2)

    def start():
        scheduler = RuntimeGraphScheduler(
            projects_root=projects_root,
            executors={
                "train_runtime": TrainRuntimeExecutor(
                    projects_root=str(projects_root), backend=backend
                ),
            },
        )
        barrier.wait(timeout=5)
        return scheduler.start(
            project=project,
            published=published,
            request_id="concurrent-start",
            inputs=_run_inputs("batch-1"),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(start) for _ in range(2)]
        first, second = [future.result(timeout=10) for future in futures]
    assert first.run_id == second.run_id
    assert backend.submit_count == 1


def test_callback_validates_conversation_and_task_kind_before_poll(tmp_path):
    projects_root = tmp_path / "projects"
    project, published = _published_project(projects_root, train_runtime=True)
    backend = _SchedulerTrainBackend()
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={
            "train_runtime": TrainRuntimeExecutor(
                projects_root=str(projects_root), backend=backend
            ),
        },
    )
    started = scheduler.start(
        project=project,
        published=published,
        request_id="callback-scope",
        inputs=_run_inputs("batch-1"),
    )

    for conversation, kind in (
        ("another-conversation", "echomind_train"),
        (project.project.conversation_id, "echomind_evaluator"),
    ):
        assert (
            scheduler.callback_by_provider_task(
                provider_run_id="train-task-1",
                status="Succeeded",
                conversation_id=conversation,
                task_kind=kind,
            )
            is None
        )
    assert backend.poll_count == 0
    owned = scheduler.callback_by_provider_task(
        provider_run_id="train-task-1",
        status="Succeeded",
        conversation_id=project.project.conversation_id,
        task_kind="echomind_train",
    )
    assert owned is not None and owned.run_id == started.run_id
    assert backend.poll_count == 1


def test_poll_timeout_preserves_existing_task_for_recovery(tmp_path, monkeypatch):
    projects_root = tmp_path / "projects"
    project, published = _published_project(projects_root, train_runtime=True)
    backend = _SchedulerTrainBackend()
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={
            "train_runtime": TrainRuntimeExecutor(
                projects_root=str(projects_root), backend=backend
            ),
        },
    )
    original_query = backend.query_task

    def timeout_then_succeed(**kwargs):
        monkeypatch.setattr(backend, "query_task", original_query)
        raise TimeoutError("query timed out")

    monkeypatch.setattr(backend, "query_task", timeout_then_succeed)
    started = scheduler.start(
        project=project,
        published=published,
        request_id="query-timeout",
        inputs=_run_inputs("batch-1"),
    )
    waiting = scheduler.recover(project=project, run_id=started.run_id)
    assert waiting.status == "waiting_external"
    stage = _stage_for_node(scheduler, project, started.run_id, "train")
    task = stage.task_runs[0]
    assert task.status == "submitted"
    assert task.provider_run_id == "train-task-1"
    assert "query timed out" in stage.message

    assert scheduler.recover(project=project, run_id=started.run_id).status == (
        "waiting_external"
    )
    assert scheduler.recover(project=project, run_id=started.run_id).status == (
        "succeeded"
    )
    assert backend.submit_count == 1


def test_uncertain_submission_only_reconciles_same_task(tmp_path, monkeypatch):
    projects_root = tmp_path / "projects"
    project, published = _published_project(projects_root, train_runtime=True)
    backend = _SchedulerTrainBackend()
    names = []
    initial_lookup = backend.find_task_by_name

    def fail_first_lookup(**kwargs):
        monkeypatch.setattr(backend, "find_task_by_name", initial_lookup)
        raise TimeoutError("lookup timed out before submission")

    def lose_submit_response(*, spec, request_context):
        names.append(spec.task_name)
        raise TimeoutError("submit response lost")

    monkeypatch.setattr(backend, "find_task_by_name", fail_first_lookup)
    monkeypatch.setattr(backend, "submit_workflow", lose_submit_response)
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={
            "train_runtime": TrainRuntimeExecutor(
                projects_root=str(projects_root), backend=backend
            ),
        },
    )
    started = scheduler.start(
        project=project,
        published=published,
        request_id="lost-submit",
        inputs=_run_inputs("batch-1"),
    )
    assert started.status == "waiting_external"
    stage = _stage_for_node(scheduler, project, started.run_id, "train")
    assert stage.task_runs[0].status == "submitting"
    assert names == []

    scheduler.recover(project=project, run_id=started.run_id)
    stage = _stage_for_node(scheduler, project, started.run_id, "train")
    assert stage.task_runs[0].status == "unknown"
    assert scheduler.recover(project=project, run_id=started.run_id).status == (
        "waiting_external"
    )
    assert len(names) == 1

    def find_visible_task(*, name, spec, request_context):
        assert names == [name]
        return "train-existing"

    monkeypatch.setattr(backend, "find_task_by_name", find_visible_task)
    scheduler.recover(project=project, run_id=started.run_id)
    recovered = _stage_for_node(scheduler, project, started.run_id, "train")
    assert recovered.task_runs[0].provider_run_id == "train-existing"
    assert recovered.task_runs[0].task_run_id == stage.task_runs[0].task_run_id


def test_cancel_waits_for_provider_and_updates_task_records(tmp_path, monkeypatch):
    projects_root = tmp_path / "projects"
    project, published = _published_project(projects_root, train_runtime=True)
    backend = _SchedulerTrainBackend()
    calls = []

    def stop(*, task_id, spec, request_context):
        calls.append(task_id)
        if len(calls) == 1:
            raise TimeoutError("stop response timed out")

    monkeypatch.setattr(backend, "cancel_task", stop)
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={
            "train_runtime": TrainRuntimeExecutor(
                projects_root=str(projects_root), backend=backend
            ),
        },
    )
    started = scheduler.start(
        project=project,
        published=published,
        request_id="cancel-platform-task",
        inputs=_run_inputs("batch-1"),
    )
    paused = scheduler.cancel(project=project, run_id=started.run_id)
    assert paused.status == "paused"
    assert paused.failure is not None
    assert "stop response timed out" in paused.failure.message
    assert scheduler.cancel(project=project, run_id=started.run_id).status == "canceled"
    assert calls == ["train-task-1", "train-task-1"]
    stage = _stage_for_node(scheduler, project, started.run_id, "train")
    assert stage.status == "canceled"
    assert stage.task_runs[0].status == "canceled"
    assert (
        scheduler.graph_run_store.list_task_runs(
            project_id=started.project_id, graph_run_id=started.run_id
        )
        == stage.task_runs
    )


def test_retry_rejects_replacement_of_succeeded_stage_input(tmp_path, monkeypatch):
    projects_root = tmp_path / "projects"
    project, published = _published_project(projects_root, train_runtime=True)
    backend = _SchedulerTrainBackend()
    monkeypatch.setattr(
        backend,
        "query_task",
        lambda **_: TrainTaskState(status="failed", message="Train failed."),
    )
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={
            "train_runtime": TrainRuntimeExecutor(
                projects_root=str(projects_root), backend=backend
            ),
        },
    )
    started = scheduler.start(
        project=project,
        published=published,
        request_id="retry-input-provenance",
        inputs=_run_inputs("batch-1"),
    )
    assert scheduler.recover(project=project, run_id=started.run_id).status == "paused"
    changed = _run_inputs("batch-1")
    changed["data"] = {"train_uri": "storage://datasets/new-train.jsonl"}
    with pytest.raises(RuntimeSchedulerError, match="already consumed by dataset"):
        scheduler.retry(
            project=project,
            published=published,
            run_id=started.run_id,
            inputs=changed,
        )
    assert backend.submit_count == 1


def test_train_poll_without_request_credentials_stays_waiting(tmp_path, monkeypatch):
    projects_root = tmp_path / "projects"
    project, published = _published_project(
        projects_root,
        train_runtime=True,
    )
    executor = _CredentialWaitingTrainExecutor()
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={"train_runtime": executor},
    )

    waiting = scheduler.start(
        project=project,
        published=published,
        request_id="manual-credential-wait",
        inputs=_run_inputs("batch-1"),
    )

    assert waiting.status == "waiting_external"
    submitted = _stage_for_node(scheduler, project, waiting.run_id, "train").task_runs[
        0
    ]
    assert not submitted.requires_request_credentials
    advanced = scheduler.advance(
        project=project,
        published=published,
        run_id=waiting.run_id,
    )
    assert advanced.status == "waiting_external"
    assert advanced.failure is None
    train_stage = _stage_for_node(
        scheduler,
        project,
        waiting.run_id,
        "train",
    )
    assert train_stage.status == "waiting_external"
    assert "request credentials" in train_stage.message
    assert [task.status for task in train_stage.task_runs] == ["submitted"]

    scheduler.callback(
        project=project,
        published=published,
        run_id=waiting.run_id,
        provider_run_id="train-credential-wait",
        status="Failed",
    )
    assert (
        _stage_for_node(scheduler, project, waiting.run_id, "train")
        .task_runs[0]
        .provider_reported_status
        is None
    )

    for status, expected in (
        ("Running", "running"),
        ("Pending", "running"),
        ("Succeeded", "succeeded"),
        ("Running", "succeeded"),
        ("Failed", "succeeded"),
    ):
        observed = scheduler.callback_by_provider_task(
            provider_run_id="train-credential-wait",
            status=status,
            conversation_id=project.project.conversation_id,
            task_kind="echomind_train",
        )
        assert observed is not None
        assert observed.status == "waiting_external"
        assert observed.failure is None
        train_stage = _stage_for_node(scheduler, project, waiting.run_id, "train")
        assert train_stage.status == "waiting_external"
        assert not train_stage.output_refs
        task = train_stage.task_runs[0]
        assert task.status == "submitted"
        assert task.finished_at is None
        assert task.provider_reported_status == expected
        assert task.requires_request_credentials
        standalone = scheduler.graph_run_store.load_task_run(
            project_id=waiting.project_id,
            graph_run_id=waiting.run_id,
            task_run_id=task.task_run_id,
        )
        assert standalone == task

    monkeypatch.setattr(
        executor,
        "poll",
        lambda *_: StagePollResult(status="running", message="Training."),
    )
    scheduler.advance(project=project, published=published, run_id=waiting.run_id)
    resumed = _stage_for_node(scheduler, project, waiting.run_id, "train").task_runs[0]
    assert resumed.task_run_id == submitted.task_run_id
    assert not resumed.requires_request_credentials
    assert (
        scheduler.graph_run_store.load_task_run(
            project_id=waiting.project_id,
            graph_run_id=waiting.run_id,
            task_run_id=resumed.task_run_id,
        )
        == resumed
    )


def test_retry_reuses_provider_task_after_legacy_credential_poll_failure(
    tmp_path,
):
    projects_root = tmp_path / "projects"
    project, published = _published_project(
        projects_root,
        train_runtime=True,
    )
    backend = _CredentialFailureOnceTrainBackend()
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={
            "train_runtime": TrainRuntimeExecutor(
                projects_root=str(projects_root),
                backend=backend,
            )
        },
    )

    waiting = scheduler.start(
        project=project,
        published=published,
        request_id="manual-legacy-credential-retry",
        inputs=_run_inputs("batch-1"),
    )
    waiting_stage = _stage_for_node(
        scheduler,
        project,
        waiting.run_id,
        "train",
    )
    failed_at = datetime.now(UTC)
    failed_task = waiting_stage.task_runs[0].model_copy(
        update={
            "status": "failed",
            "error": (
                "RuntimeCredentialUnavailableError: current request credentials "
                "are unavailable"
            ),
            "finished_at": failed_at,
        }
    )
    failed_stage = waiting_stage.model_copy(
        update={
            "status": "failed",
            "message": "Train failed.",
            "error": f"infrastructure: {failed_task.error}",
            "task_runs": [failed_task],
            "finished_at": failed_at,
        }
    )
    scheduler.stage_run_store.save(failed_stage)
    scheduler.graph_run_store.save_task_run(
        project_id=project.project.project_id,
        graph_run_id=waiting.run_id,
        task_run=failed_task,
    )
    assert failed_stage.error is not None
    paused = scheduler.graph_run_store.save(
        waiting.model_copy(
            update={
                "status": "paused",
                "current_stage_ids": [failed_stage.run_id],
                "failure": RuntimeFailure(
                    category="infrastructure",
                    message=failed_stage.error,
                    details={"node_id": "train"},
                ),
            }
        )
    )

    reconnected = scheduler.retry(
        project=project,
        published=published,
        run_id=paused.run_id,
        inputs=waiting.inputs,
    )

    assert reconnected.status == "waiting_external"
    assert backend.submit_count == 1
    train_stages = [
        stage
        for stage in scheduler.stage_run_store.list_for_graph_run(
            project_id=project.project.project_id,
            graph_run_id=paused.run_id,
        )
        if stage.node_id == "train"
    ]
    assert [(stage.attempt, stage.status) for stage in train_stages] == [
        (1, "failed"),
        (2, "waiting_external"),
    ]
    assert {stage.task_runs[0].provider_run_id for stage in train_stages} == {
        "train-existing"
    }
    latest_task = scheduler.graph_run_store.find_task_run_by_provider_id(
        project_id=paused.project_id,
        graph_run_id=paused.run_id,
        provider_run_id="train-existing",
    )
    assert latest_task is not None
    assert latest_task.task_run_id == train_stages[-1].task_runs[0].task_run_id

    completed = scheduler.advance(
        project=project,
        published=published,
        run_id=paused.run_id,
    )

    assert completed.status == "succeeded"
    assert backend.submit_count == 1
    train_stages = [
        stage
        for stage in scheduler.stage_run_store.list_for_graph_run(
            project_id=project.project.project_id,
            graph_run_id=paused.run_id,
        )
        if stage.node_id == "train"
    ]
    assert [(stage.attempt, stage.status) for stage in train_stages] == [
        (1, "failed"),
        (2, "succeeded"),
    ]
    assert {stage.task_runs[0].provider_run_id for stage in train_stages} == {
        "train-existing"
    }


def test_paused_validation_run_retries_with_corrected_inputs(tmp_path):
    projects_root = tmp_path / "projects"
    project, published = _published_project(
        projects_root,
        dataset_runtime=True,
    )
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "train_uri": "/datasets/train.jsonl",
                "val_uri": "/datasets/val.jsonl",
                "train_samples": 3,
                "val_samples": 2,
            }
        ),
        encoding="utf-8",
    )
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={"dataset_runtime": DatasetRuntimeExecutor()},
    )

    paused = scheduler.start(
        project=project,
        published=published,
        request_id="manual-retry-validation",
        inputs={
            "data": "/datasets/raw",
            "base_model": "storage://models/base",
        },
    )

    assert paused.status == "paused"
    assert paused.failure is not None
    assert paused.failure.category == "validation"
    first_dataset = _stage_for_node(
        scheduler,
        project,
        paused.run_id,
        "dataset",
    )
    assert first_dataset.attempt == 1
    assert first_dataset.status == "failed"

    completed = scheduler.retry(
        project=project,
        published=published,
        run_id=paused.run_id,
        inputs={
            "data": str(manifest),
            "base_model": "storage://models/base",
        },
    )

    assert completed.status == "succeeded"
    assert completed.inputs["data"] == str(manifest)
    stages = scheduler.stage_run_store.list_for_graph_run(
        project_id=project.project.project_id,
        graph_run_id=paused.run_id,
    )
    dataset_stages = [stage for stage in stages if stage.node_id == "dataset"]
    assert [(stage.attempt, stage.status) for stage in dataset_stages] == [
        (1, "failed"),
        (2, "succeeded"),
    ]


def test_evaluator_runtime_waits_recovers_and_finishes_graph(tmp_path):
    projects_root = tmp_path / "projects"
    project, published = _published_project(
        projects_root,
        evaluator_runtime=True,
    )
    backend = _SchedulerEvaluatorBackend()
    scheduler = RuntimeGraphScheduler(
        projects_root=projects_root,
        executors={
            "evaluator_runtime": EvaluatorRuntimeExecutor(
                projects_root=str(projects_root),
                backend=backend,
            )
        },
    )

    started = scheduler.start(
        project=project,
        published=published,
        request_id="manual-evaluator-runtime",
        inputs=_run_inputs("batch-1"),
    )

    assert started.status == "waiting_external"
    assert backend.submit_count == 1
    evaluator_stage = _stage_for_node(
        scheduler,
        project,
        started.run_id,
        "evaluator",
    )
    assert evaluator_stage.status == "waiting_external"
    assert [task.status for task in evaluator_stage.task_runs] == ["submitted"]

    running = scheduler.callback_by_provider_task(
        provider_run_id="evaluator-task-1",
        status="Succeeded",
    )

    assert running is not None
    assert running.status == "waiting_external"
    evaluator_stage = _stage_for_node(
        scheduler,
        project,
        started.run_id,
        "evaluator",
    )
    assert [task.status for task in evaluator_stage.task_runs] == ["running"]
    assert backend.submit_count == 1

    completed = scheduler.callback_by_provider_task(
        provider_run_id="evaluator-task-1",
        status="Succeeded",
    )

    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.outcome == "candidate_recorded"
    assert completed.promotion_result["decision"] == "awaiting_user_selection"
    assert backend.submit_count == 1
    stages = scheduler.stage_run_store.list_for_graph_run(
        project_id=project.project.project_id,
        graph_run_id=started.run_id,
    )
    assert [stage.status for stage in stages] == [
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    evaluator_stage = next(stage for stage in stages if stage.node_id == "evaluator")
    assert [task.status for task in evaluator_stage.task_runs] == ["succeeded"]


class _ReadyBatchSource:
    def __init__(self, batches: list[ReadyBatch]) -> None:
        self.batches = batches

    def list_ready_batches(
        self,
        *,
        project_id: str,
        tracer_resource_id: str,
    ) -> list[ReadyBatch]:
        del project_id
        return [
            batch
            for batch in self.batches
            if batch.tracer_resource_id == tracer_resource_id
        ]


class _CredentialWaitingTrainExecutor:
    def prepare(self, context):
        del context
        return StagePreparationResult(
            status="waiting_external",
            message="Train ready.",
            external_task=ExternalTaskIntent(
                provider="pyromind.studio.training",
            ),
        )

    def submit(self, context, task_run):
        del context, task_run
        return StageSubmissionResult(
            status="submitted",
            message="Train submitted.",
            provider_run_id="train-credential-wait",
        )

    def poll(self, context, task_run):
        del context, task_run
        raise RuntimeCredentialUnavailableError(
            "current request credentials are unavailable"
        )

    def cancel(self, context, task_run):
        del context, task_run
        return StageCancellationResult(
            status="canceled",
            message="Train canceled.",
        )


class _CredentialFailureOnceTrainBackend:
    def __init__(self) -> None:
        self.submit_count = 0
        self.poll_count = 0

    def find_task_by_name(self, *, name, spec, request_context):
        del name, spec, request_context
        return None

    def submit_workflow(self, *, spec, request_context):
        del spec, request_context
        self.submit_count += 1
        return "train-existing"

    def query_task(self, *, task_id, spec, request_context):
        del task_id, spec, request_context
        self.poll_count += 1
        if self.poll_count == 1:
            raise RuntimeCredentialUnavailableError(
                "current request credentials are unavailable"
            )
        return TrainTaskState(
            status="succeeded",
            message="Train completed.",
            output_uri="storage://models/candidate-1",
        )

    def cancel_task(self, *, task_id, spec, request_context):
        del task_id, spec, request_context


class _SchedulerTrainBackend:
    def __init__(self) -> None:
        self.submit_count = 0
        self.poll_count = 0

    def find_task_by_name(self, *, name, spec, request_context):
        del name, spec, request_context
        return None

    def submit_workflow(self, *, spec, request_context):
        del spec, request_context
        self.submit_count += 1
        return "train-task-1"

    def query_task(self, *, task_id, spec, request_context):
        del task_id, spec, request_context
        self.poll_count += 1
        if self.poll_count == 1:
            return TrainTaskState(status="running", message="Train is running.")
        return TrainTaskState(
            status="succeeded",
            message="Train completed.",
            output_uri="storage://models/candidate-1",
        )

    def cancel_task(self, *, task_id, spec, request_context):
        del task_id, spec, request_context


class _SchedulerEvaluatorBackend:
    def __init__(self) -> None:
        self.submit_count = 0
        self.poll_count = 0

    def find_task_by_name(self, *, name, spec, request_context):
        del name, spec, request_context
        return None

    def submit_workflow(self, *, spec, request_context):
        del spec, request_context
        self.submit_count += 1
        return "evaluator-task-1"

    def query_task(self, *, task_id, spec, request_context):
        del task_id, spec, request_context
        self.poll_count += 1
        if self.poll_count == 1:
            return EvaluatorTaskState(
                status="running",
                message="Evaluation is running.",
            )
        return EvaluatorTaskState(
            status="succeeded",
            message="Evaluation completed.",
            output_uri="storage://evaluations/candidate-1/report.json",
        )

    def cancel_task(self, *, task_id, spec, request_context):
        del task_id, spec, request_context


def _published_project(
    projects_root,
    *,
    with_runtime_slot: bool = False,
    dataset_runtime: bool = False,
    train_runtime: bool = False,
    evaluator_runtime: bool = False,
    development: bool = False,
) -> tuple[ProjectInfo, PublishedGraph]:
    project_id = str(uuid4())
    digest = canonical_sha256({"executor": "contract_probe", "version": 1})
    nodes = [
        ProjectNode(
            id="dataset",
            type="dataset",
            title="Dataset",
            executor="dataset_runtime" if dataset_runtime else "contract_probe",
            implementation_ref=(
                "builtin://echomind/dataset_runtime/v1"
                if dataset_runtime
                else "builtin://echomind/contract_probe/v1"
            ),
            implementation_digest=(
                canonical_sha256({"executor": "dataset_runtime", "version": 1})
                if dataset_runtime
                else digest
            ),
            inputs={
                "source_batch": ProjectNodeInput(
                    artifact_type="dataset.source",
                    schema_ref="echomind://schemas/dataset-source/v1",
                    binding=ProjectNodeInputBinding(
                        kind="run_input",
                        name="data",
                    ),
                )
            },
            outputs={
                "train_messages": ProjectNodeOutput(
                    artifact_type="dataset.messages.train",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                ),
                "val_messages": ProjectNodeOutput(
                    artifact_type="dataset.messages.validation",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                ),
            },
        ),
        ProjectNode(
            id="train",
            type="train",
            title="Train",
            executor="train_runtime" if train_runtime else "contract_probe",
            implementation_ref=(
                "builtin://echomind/train_runtime/v1"
                if train_runtime
                else "builtin://echomind/contract_probe/v1"
            ),
            implementation_digest=canonical_sha256(
                {
                    "executor": (
                        "train_runtime" if train_runtime else "contract_probe"
                    ),
                    "version": 1,
                }
            ),
            inputs={
                "train_dataset": ProjectNodeInput(
                    artifact_type="dataset.messages.train",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                    binding=ProjectNodeInputBinding(
                        kind="node_output",
                        node_id="dataset",
                        port="train_messages",
                    ),
                ),
                "validation_dataset": ProjectNodeInput(
                    artifact_type="dataset.messages.validation",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                    binding=ProjectNodeInputBinding(
                        kind="node_output",
                        node_id="dataset",
                        port="val_messages",
                    ),
                ),
                "base_model": ProjectNodeInput(
                    artifact_type="model.checkpoint",
                    schema_ref="echomind://schemas/model/v1",
                    binding=ProjectNodeInputBinding(
                        kind="run_input",
                        name="base_model",
                    ),
                ),
            },
            outputs={
                "trained_model": ProjectNodeOutput(
                    artifact_type="model.checkpoint",
                    schema_ref="echomind://schemas/model/v1",
                )
            },
            config=(
                {
                    "workflow": {
                        "name": "train",
                        "nodes": [
                            {
                                "data": {
                                    "config": {
                                        "train_data_path": "",
                                        "val_data_path": "",
                                    }
                                }
                            }
                        ],
                    },
                    "workflow_bindings": {
                        "nodes[0].data.config.train_data_path": "train_dataset",
                        "nodes[0].data.config.val_data_path": ("validation_dataset"),
                    },
                }
                if train_runtime
                else {}
            ),
        ),
        ProjectNode(
            id="evaluator",
            type="evaluator",
            title="Evaluator",
            executor=("evaluator_runtime" if evaluator_runtime else "contract_probe"),
            implementation_ref=(
                "builtin://echomind/evaluator_runtime/v1"
                if evaluator_runtime
                else "builtin://echomind/contract_probe/v1"
            ),
            implementation_digest=canonical_sha256(
                {
                    "executor": (
                        "evaluator_runtime" if evaluator_runtime else "contract_probe"
                    ),
                    "version": 1,
                }
            ),
            inputs={
                "candidate_model": ProjectNodeInput(
                    artifact_type="model.checkpoint",
                    schema_ref="echomind://schemas/model/v1",
                    binding=ProjectNodeInputBinding(
                        kind="node_output",
                        node_id="train",
                        port="trained_model",
                    ),
                ),
                "validation_dataset": ProjectNodeInput(
                    artifact_type="dataset.messages.validation",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                    binding=ProjectNodeInputBinding(
                        kind="node_output",
                        node_id="dataset",
                        port="val_messages",
                    ),
                ),
            },
            outputs={
                "eval_report": ProjectNodeOutput(
                    artifact_type="evaluation.report",
                    schema_ref="echomind://schemas/evaluation-report/v1",
                )
            },
            policies={
                "promotion": {"metric": "accuracy", "direction": "maximize"},
                "candidate_inference": {
                    "create": "evaluator",
                    "health_check": "required",
                    "cleanup": "finalizer",
                },
            },
            config=(
                {
                    "workflow": {
                        "name": "evaluation",
                        "nodes": [
                            {
                                "data": {
                                    "config": {
                                        "model_path": "",
                                        "dataset_path": "",
                                    }
                                }
                            }
                        ],
                    },
                    "workflow_bindings": {
                        "nodes[0].data.config.model_path": "candidate_model",
                        "nodes[0].data.config.dataset_path": ("validation_dataset"),
                    },
                }
                if evaluator_runtime
                else {}
            ),
        ),
    ]
    edges = [
        ProjectEdge(source="dataset", target="train"),
        ProjectEdge(source="train", target="evaluator"),
    ]
    release_id = str(uuid4())
    published = PublishedGraph.build(
        release_id=release_id,
        project_id=project_id,
        source_graph_id=str(uuid4()),
        source_graph_digest=canonical_sha256({"nodes": [node.id for node in nodes]}),
        runtime_requirements=RuntimeRequirements(),
        promotion_policy=PromotionPolicy(metric="accuracy"),
        nodes=nodes,
        edges=edges,
        published_by="42",
    )
    lifecycle = "development" if development else "frozen"
    project = ProjectInfo(
        project=ProjectRecord(
            project_id=project_id,
            name="Runtime scheduler test",
            description="Runtime scheduler test",
            skill="experiment-planning",
            owner_user_id="42",
            conversation_id="conversation-1",
            cluster="us-west-1#pre",
            lifecycle=lifecycle,
            active_release_id=None if development else release_id,
            frozen_at=None if development else datetime.now(UTC),
        ),
        graph=ProjectGraph(
            graph_id=published.source_graph_id,
            project_id=project_id,
            lifecycle=lifecycle,
            nodes=nodes,
            edges=edges,
        ),
    )
    ProjectStore(projects_root).save(project)
    publication_store = PublicationStore(projects_root)
    if development:
        publication_store.save(published)
    else:
        publication_store.activate(published)
    if with_runtime_slot:
        _save_ready_runtime_slot(projects_root, project_id, release_id)
    return project, published


def _save_ready_runtime_slot(
    projects_root,
    project_id: str,
    release_id: str,
) -> None:
    health = RuntimeHealth(status="healthy", checked_at=datetime.now(UTC))
    RuntimeSlotStore(projects_root).save(
        RuntimeSlot(
            project_id=project_id,
            status="ready",
            active_release_id=release_id,
            tracer=TracerRuntime(
                resource_id="tracer-1",
                endpoint="https://tracer.example.test",
                admin_secret_ref="secret://echomind/tracer/admin",
                proxy_secret_ref="secret://echomind/tracer/proxy",
                target_inference_resource_id="inference-a",
                buffer_config=TracerBufferConfig(
                    trajectory_buffer_size=100,
                    batch_size=10,
                    time_per_round_seconds=60,
                    training_save_path="/target-workspace/models",
                ),
                health=health,
            ),
            serving_inference=ServingInferenceRuntime(
                resource_id="inference-a",
                endpoint="https://serving.example.test",
                secret_ref="secret://echomind/inference/a",
                model_uri="storage://models/baseline",
                health=health,
            ),
        )
    )


def _run_inputs(batch_id: str) -> dict[str, object]:
    return {
        "data": {
            "train_uri": "storage://datasets/train.jsonl",
            "val_uri": "storage://datasets/val.jsonl",
            "train_samples": 8,
            "val_samples": 2,
        },
        "base_model": "storage://models/base",
        "tracer_batch": {
            "resource_id": "tracer-1",
            "batch_id": batch_id,
            "uri": f"tracer://tracer-1/batches/{batch_id}",
        },
    }


def _stage_for_node(
    scheduler: RuntimeGraphScheduler,
    project: ProjectInfo,
    graph_run_id: str,
    node_id: str,
):
    return next(
        stage
        for stage in scheduler.stage_run_store.list_for_graph_run(
            project_id=project.project.project_id,
            graph_run_id=graph_run_id,
        )
        if stage.node_id == node_id
    )


def test_graph_start_waits_for_manual_deployment_under_project_lease(tmp_path):
    root = tmp_path / "projects"
    project, published = _published_project(root, with_runtime_slot=True)
    project_id = project.project.project_id
    store = FileDeploymentStore(root)
    with store.lease(project_id):
        store.save(
            DeploymentRecord(
                deployment_id=uuid4().hex,
                project_id=project_id,
                request=DeploymentRequest(
                    request_id="deploy", candidate_id="candidate"
                ),
                candidate_id="candidate",
                model_uri="/workspace/new",
                previous=DeploymentTarget(
                    resource_id="inf-old", model_uri="/workspace/old"
                ),
                status="waiting_inference",
            )
        )
    scheduler = RuntimeGraphScheduler(projects_root=root)
    with pytest.raises(RuntimeSchedulerError, match="deployment is still in progress"):
        scheduler.start(
            project=project,
            published=published,
            request_id="batch",
            inputs=_run_inputs("batch"),
        )
    assert scheduler.graph_run_store.list(project_id) == []
