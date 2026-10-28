"""Background selection survives restarts without resubmitting platform work."""

from datetime import timedelta
from uuid import uuid4

import httpx
import pytest
from pyromind_agent_server.checkpoint_scheduler import CheckpointScheduler
from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_runtime.application.checkpoint_evaluations import (
    CheckpointEvaluationCoordinator,
    job_state,
)
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.checkpoint_evaluation import (
    CheckpointEvaluationRequest,
    CheckpointObservation,
    EvaluationExecution,
    TaskStatus,
    TrainingObservation,
)
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.training_evaluation import (
    CheckpointArtifact,
    StepEvaluation,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.checkpoint_evaluation import EvaluationCredentialsRequired

from .fake_adapter import FakeAdapter
from .test_product_api import _app


class FakePlatform:
    def __init__(self):
        self.training_status: TaskStatus = "running"
        self.training_step = 1000
        self.files = {100: "v1", 200: "v1"}
        self.complete = True
        self.submitted = []
        self.executions = {}
        self.stopped = []
        self.acceptance_unknown = False
        self.credentials_expired = False
        self.publish_error = False

    async def freeze(self, request, output_dir, context):
        return {
            "training_task_id": request.training_task_id,
            "training_node_id": request.training_node_id,
            "comparison_group": "frozen",
            "datasets": {"validation": {"path": "/val.jsonl", "sha256": "val"}},
            "template": {
                "inference": {},
                "evaluation_config": {},
                "input_path": "/val.jsonl",
            },
        }

    async def observe(self, definition, context):
        return TrainingObservation(
            status=self.training_status,
            training_step=self.training_step,
            checkpoints=[
                CheckpointObservation(
                    artifact=CheckpointArtifact(
                        step=step,
                        path=f"/training/checkpoint-{step}",
                        kind="full",
                        complete=self.complete,
                    ),
                    fingerprint=version,
                )
                for step, version in self.files.items()
            ],
        )

    async def prepare(self, parent, child, context) -> JsonObject:
        if self.credentials_expired:
            raise EvaluationCredentialsRequired()
        return {"workflow": "frozen"}

    async def submit(self, child, workflow, context):
        self.submitted.append(child)
        if self.acceptance_unknown:
            raise TimeoutError("accepted or not: unknown")
        return "task-" + str(child.run_id)

    async def inspect(self, parent, candidate, context):
        execution = self.executions.get(candidate.artifact.step)
        if execution == "success":
            return EvaluationExecution(
                status="succeeded",
                evaluation=StepEvaluation(
                    step=candidate.artifact.step,
                    run_id=str(candidate.run_id),
                    output_dir="/outputs/" + str(candidate.run_id),
                    checkpoint=candidate.artifact,
                    metrics={"pass_rate": candidate.artifact.step / 1000},
                    comparison_group="frozen",
                    eligible=True,
                ),
            )
        if execution == "weights":
            return EvaluationExecution(status="failed", weights_not_ready=True)
        if execution == "failed":
            return EvaluationExecution(status="failed", error="inference failed")
        return EvaluationExecution(status="running")

    async def verify_submission(self, child, task_id, context):
        if task_id != "verified-task":
            raise ValueError("wrong platform task")

    async def stop(self, task_id, context):
        self.stopped.append(task_id)

    async def publish(self, parent, context):
        if self.publish_error:
            raise ConnectionError()


@pytest.fixture
def checkpoint_job(tmp_path):
    scope = "a" * 32
    root = tmp_path / scope
    root.mkdir()
    store = FileProductStore(root)
    store.create(
        ConversationSnapshot(conversation_id=scope, capabilities=HarnessCapabilities()),
        user_id="42",
    )
    platform = FakePlatform()
    registry = WorkflowExternalTaskRegistry(tmp_path)
    coordinator = CheckpointEvaluationCoordinator(store, registry, platform)
    return scope, store, platform, registry, coordinator


async def register_job(fixture):
    scope, store, platform, registry, coordinator = fixture
    request = CheckpointEvaluationRequest(
        training_task_id="123", training_node_id="train", template={}
    )
    context = RequestContext(user_id="42", cookie="ephemeral-cookie")
    run = await coordinator.register(uuid4(), scope, request, context)
    return run, context


def selection(run):
    result = job_state(run).selection
    assert result is not None
    return result


def terminal_time(run):
    result = job_state(run).terminal_since
    assert result is not None
    return result


async def advance(coordinator, run, context, seconds=60):
    state = job_state(run)
    return await coordinator.tick(
        run, context, now=state.next_poll_at + timedelta(seconds=seconds - 60)
    )


async def stabilize(coordinator, run, context):
    for _ in range(3):
        run = await advance(coordinator, run, context)
    return run


async def test_checkpoint_evaluations_are_serial_and_finalize_after_training_and_drain(
    checkpoint_job,
):
    scope, store, platform, registry, coordinator = checkpoint_job
    run, context = await register_job(checkpoint_job)
    run = await stabilize(coordinator, run, context)
    assert len(platform.submitted) == 1
    assert job_state(run).candidates["100"].status == "running"
    assert job_state(run).candidates["200"].status == "ready"
    # A reconstructed coordinator reuses persisted child identity.
    coordinator = CheckpointEvaluationCoordinator(store, registry, platform)
    run = await advance(coordinator, run, context)
    assert len(platform.submitted) == 1
    platform.executions[100] = "success"
    run = await advance(coordinator, run, context)
    assert len(platform.submitted) == 2
    assert selection(run).selected_step == 100
    assert (
        platform.submitted[0].definition["comparison_group"]
        == platform.submitted[1].definition["comparison_group"]
    )
    platform.training_status = "succeeded"
    platform.executions[200] = "success"
    run = await advance(coordinator, run, context)
    assert job_state(run).status == "draining"
    assert selection(run).selected_step == 200
    run = await coordinator.tick(
        run, context, now=terminal_time(run) + timedelta(seconds=600)
    )
    assert job_state(run).status == "completed"
    assert len(platform.submitted) == 2
    assert (
        "ephemeral-cookie" not in (store.directory / "pipeline-runs.json").read_text()
    )


async def test_without_wandb_waits_five_minutes_and_metadata_change_resets_window(
    checkpoint_job,
):
    *_, platform, registry, coordinator = checkpoint_job
    run, context = await register_job(checkpoint_job)
    platform.training_step = None
    run = await stabilize(coordinator, run, context)
    assert not platform.submitted
    platform.files[100] = "v2"
    run = await advance(coordinator, run, context)
    assert job_state(run).candidates["100"].observations == 1
    run = await advance(coordinator, run, context)
    run = await advance(coordinator, run, context)
    assert len(platform.submitted) == 1
    run = await advance(coordinator, run, context)
    assert platform.submitted[0].definition["checkpoint"]["step"] == 200


async def test_uncertain_submission_never_retries_after_restart_or_cancel(
    checkpoint_job,
):
    scope, store, platform, registry, coordinator = checkpoint_job
    run, context = await register_job(checkpoint_job)
    platform.acceptance_unknown = True
    run = await stabilize(coordinator, run, context)
    assert job_state(run).status == "needs_attention"
    coordinator = CheckpointEvaluationCoordinator(store, registry, platform)
    for _ in range(3):
        run = await advance(coordinator, run, context)
    assert len(platform.submitted) == 1
    run = coordinator.cancel(run)
    run = await advance(coordinator, run, context)
    assert job_state(run).status == "needs_attention"
    assert not platform.stopped
    assert job_state(run).candidates["200"].status == "cancelled"


async def test_credentials_pause_and_preparation_can_resume_without_duplicate_children(
    checkpoint_job,
):
    scope, store, platform, registry, coordinator = checkpoint_job
    run, context = await register_job(checkpoint_job)
    run = await advance(coordinator, run, None)
    assert job_state(run).status == "waiting_credentials"
    platform.credentials_expired = True
    run = await stabilize(coordinator, run, context)
    assert job_state(run).status == "waiting_credentials"
    assert not platform.submitted
    platform.credentials_expired = False
    run = await advance(coordinator, run, context)
    assert len(platform.submitted) == 1
    assert len(store.load_pipeline_runs()) == 2


@pytest.mark.parametrize("training_status", ["succeeded", "failed", "stopped"])
async def test_terminal_training_keeps_polling_children_and_marks_unstable_files(
    checkpoint_job, training_status
):
    scope, store, platform, registry, coordinator = checkpoint_job
    run, context = await register_job(checkpoint_job)
    run = await stabilize(coordinator, run, context)
    platform.training_status = training_status
    platform.files.pop(200)
    run = await advance(coordinator, run, context)
    run = await coordinator.tick(
        run, context, now=terminal_time(run) + timedelta(seconds=600)
    )
    assert job_state(run).status == "draining"
    assert job_state(run).candidates["200"].status == "unavailable"
    platform.executions[100] = "success"
    run = await advance(coordinator, run, context)
    assert job_state(run).status == "completed"


async def test_cancel_stops_only_evaluation_and_waits_for_confirmation(checkpoint_job):
    scope, store, platform, registry, coordinator = checkpoint_job
    run, context = await register_job(checkpoint_job)
    run = await stabilize(coordinator, run, context)
    run = coordinator.cancel(run)
    run = await advance(coordinator, run, context)
    assert job_state(run).status == "cancelling"
    assert len(platform.stopped) == 1 and "123" not in platform.stopped
    platform.executions[100] = "failed"
    run = await advance(coordinator, run, context)
    assert job_state(run).status == "cancelled"


async def test_weight_loading_failure_retries_once_and_other_failures_dont_block_queue(
    checkpoint_job,
):
    scope, store, platform, registry, coordinator = checkpoint_job
    run, context = await register_job(checkpoint_job)
    run = await stabilize(coordinator, run, context)
    platform.executions[100] = "weights"
    run = await advance(coordinator, run, context)
    assert len(platform.submitted) == 2  # Other ready checkpoint can proceed.
    platform.executions[200] = "failed"
    run = await advance(coordinator, run, context)
    run = await advance(coordinator, run, context)
    assert len(platform.submitted) == 3
    run = await advance(coordinator, run, context)
    assert job_state(run).candidates["100"].status == "failed"
    assert len(platform.submitted) == 3


async def test_deleted_successful_checkpoint_is_not_selected(checkpoint_job):
    scope, store, platform, registry, coordinator = checkpoint_job
    run, context = await register_job(checkpoint_job)
    run = await stabilize(coordinator, run, context)
    platform.executions[100] = "success"
    run = await advance(coordinator, run, context)
    platform.files.pop(100)
    run = await advance(coordinator, run, context)
    assert selection(run).selected_step is None
    assert "checkpoint_deleted_after_evaluation" in selection(run).excluded["100"]


async def test_uncertain_reconciliation_verifies_platform_identity(checkpoint_job):
    scope, store, platform, registry, coordinator = checkpoint_job
    run, context = await register_job(checkpoint_job)
    platform.acceptance_unknown = True
    run = await stabilize(coordinator, run, context)
    with pytest.raises(ValueError, match="wrong platform task"):
        await coordinator.reconcile(run, 100, "wrong-task", context)
    run = await coordinator.reconcile(run, 100, "verified-task", context)
    assert job_state(run).candidates["100"].task_id == "verified-task"
    assert registry.owner("verified-task") == scope
    run = await advance(coordinator, run, context)
    assert len(platform.submitted) == 1


async def test_runtime_manages_children_without_waking_agent_and_recovers_credentials(
    checkpoint_job,
):
    scope, store, platform, registry, coordinator = checkpoint_job
    runtime = ConversationRuntime(
        store.conversation_dir.parent,
        FakeAdapter(),
        external_tasks=registry,
        checkpoint_platform=platform,
    )
    context = RequestContext(user_id="42", cookie="ephemeral")
    request_id = str(uuid4())
    args = {"training_task_id": "123", "training_node_id": "train", "template": {}}
    try:
        result = await runtime.checkpoint_evaluation_call(
            scope, "register", args, request_id, context
        )
        repeated = await runtime.checkpoint_evaluation_call(
            scope, "register", args, request_id, context
        )
        assert repeated["run_id"] == result["run_id"]
        other_scope = "b" * 32
        other_root = store.conversation_dir.parent / other_scope
        other_root.mkdir()
        other_store = FileProductStore(other_root)
        other_store.create(
            ConversationSnapshot(
                conversation_id=other_scope, capabilities=HarnessCapabilities()
            ),
            user_id="42",
        )
        other_job = await runtime.checkpoint_evaluation_call(
            other_scope, "register", args, request_id, context
        )
        assert other_job["run_id"] != result["run_id"]
        assert (
            registry.owner("checkpoint-selection:" + other_job["run_id"]) == other_scope
        )
        with pytest.raises(PermissionError):
            await runtime.checkpoint_evaluation_call(
                scope,
                "get",
                {"run_id": result["run_id"]},
                request_id,
                RequestContext(user_id="43"),
            )
        run = store.load_pipeline_runs()[result["run_id"]]
        run = await stabilize(coordinator, run, context)
        child = job_state(run).candidates["100"]
        assert child.task_id is not None
        await runtime.deliver_external_task_status(
            scope, task_id=child.task_id, status="succeeded"
        )
        task = next(
            t
            for t in store.load_snapshot().external_tasks
            if t.task_id == child.task_id
        )
        assert not task.resume_pending
        platform.executions[100] = "success"
        run = await advance(coordinator, run, context)
        await runtime.close()
        runtime = ConversationRuntime(
            store.conversation_dir.parent,
            FakeAdapter(),
            external_tasks=registry,
            checkpoint_platform=platform,
        )
        state = job_state(run)
        state.next_poll_at = state.next_poll_at - timedelta(days=1)
        coordinator.save(run, state)
        await runtime.poll_checkpoint_evaluations()
        assert (
            job_state(store.load_pipeline_runs()[result["run_id"]]).status
            == "waiting_credentials"
        )
        await runtime.get_snapshot(scope, context)
        await runtime.poll_checkpoint_evaluations()
        assert (
            job_state(store.load_pipeline_runs()[result["run_id"]]).status
            != "waiting_credentials"
        )
    finally:
        await runtime.close()


async def test_checkpoint_http_api_authorizes_and_returns_durable_job(
    checkpoint_job, tmp_path
):
    scope, store, platform, registry, coordinator = checkpoint_job
    runtime = ConversationRuntime(
        store.conversation_dir.parent,
        FakeAdapter(),
        external_tasks=registry,
        checkpoint_platform=platform,
    )
    transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
    url = f"/api/v2/pyromind/conversations/{scope}/checkpoint-evaluations"
    request_id = str(uuid4())
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"x-pyromind-debug-user-id": "42", "cookie": "ephemeral"},
        ) as client:
            response = await client.post(
                url,
                json={
                    "action": "register",
                    "request_id": request_id,
                    "arguments": {
                        "training_task_id": "123",
                        "training_node_id": "train",
                        "template": {},
                    },
                },
            )
            assert response.status_code == 200, response.text
            run_id = response.json()["run_id"]
            assert run_id != request_id
            assert response.json()["checkpoint_evaluation"]["status"] == "watching"
            other = await client.post(
                url,
                headers={"x-pyromind-debug-user-id": "43"},
                json={
                    "action": "get",
                    "request_id": request_id,
                    "arguments": {"run_id": run_id},
                },
            )
            assert other.status_code == 404
            cancelled = await client.post(
                url,
                json={
                    "action": "cancel",
                    "request_id": request_id,
                    "arguments": {"run_id": run_id},
                },
            )
            assert cancelled.json()["checkpoint_evaluation"]["cancel_requested"]
    finally:
        await runtime.close()


async def test_scheduler_owns_one_lock_and_releases_it_on_shutdown(checkpoint_job):
    scope, store, platform, registry, coordinator = checkpoint_job
    runtime = ConversationRuntime(
        store.conversation_dir.parent,
        FakeAdapter(),
        external_tasks=registry,
        checkpoint_platform=platform,
    )
    lock = store.conversation_dir.parent / "scheduler.lock"
    first, second = (
        CheckpointScheduler(runtime, lock),
        CheckpointScheduler(runtime, lock),
    )
    try:
        first.start()
        first.start()
        with pytest.raises(RuntimeError, match="one server worker"):
            second.start()
        await first.close()
        second.start()
    finally:
        await first.close()
        await second.close()
        await runtime.close()
