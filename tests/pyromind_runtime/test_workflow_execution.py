from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from harness_adapter.pi_adapter.business_tool_host import (
    PyromindBusinessToolHost,
    ToolExecutionContext,
)
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import (
    UserMessageCommand,
)
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.storage_paths import resolve_conversation_storage_path
from pyromind_runtime.domain.task_correlation import TaskCorrelation, TaskOrigin
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter


class FakeSubmission:
    def __init__(self, *, training: bool = True) -> None:
        self.training = training
        self.workflow = {"name": "sft", "nodes": [{"data": {"nodeType": "SFTTrain"}}]}
        self.submissions: list[tuple[dict, str]] = []
        self.correlations: list[TaskCorrelation | None] = []
        self.prepared_paths: list[str] = []

    async def prepare(self, workflow_path: str, context: RequestContext) -> dict:
        self.prepared_paths.append(workflow_path)
        return {
            "workflow": dict(self.workflow),
            "fingerprint": "sha256",
            "summary": {
                "name": "sft",
                "workflow_path": workflow_path,
                "has_training_node": self.training,
            },
        }

    async def submit(
        self,
        workflow: dict,
        conversation_id: str,
        context: RequestContext,
        *,
        correlation: TaskCorrelation | None = None,
    ) -> str:
        self.submissions.append((workflow, conversation_id))
        self.correlations.append(correlation)
        return f"task-{122 + len(self.submissions)}"


@pytest.mark.parametrize(
    "path",
    ["../workflow.py", "storage/workflow.py", "/target-workspace/workflow.py"],
)
def test_conversation_storage_path_rejects_non_conversation_paths(path):
    with pytest.raises(ValueError):
        resolve_conversation_storage_path("conversation-1", path, field="workflow_path")


async def _runtime(tmp_path, *, training: bool = True, harness_id: str = "pi"):
    adapter = FakeAdapter(harness_id)
    submission = FakeSubmission(training=training)
    runtime = ConversationRuntime(
        tmp_path,
        {harness_id: adapter},
        default_harness_id=harness_id,
        workflow_submission=submission,
    )
    context = RequestContext(user_id="42", cookie="auth_token=ephemeral")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="workflow-conversation",
            user_id="42",
            workspace_root=str(tmp_path / "workflow-conversation"),
        ),
        context,
    )
    return runtime, adapter, submission, context


async def test_conversation_workflow_path_is_resolved_only_for_preparation(tmp_path):
    runtime, _, submission, context = await _runtime(tmp_path)
    conversation_id = "workflow-conversation"
    workflow_path = "public_data/workflow_canvas/workflow.py"

    prepared = await runtime.prepare_workflow_execution(
        conversation_id,
        workflow_path,
        context,
        request_id="request-1",
        workflow_storage_path=(
            f"/workspace/.pyromind-agent/{conversation_id}/{workflow_path}"
        ),
    )

    assert prepared["summary"]["workflow_path"] == workflow_path
    assert submission.prepared_paths == [
        f"/workspace/.pyromind-agent/{conversation_id}/{workflow_path}"
    ]
    await runtime.close()


@pytest.mark.parametrize("status", ["submitting", "uncertain", "submitted"])
async def test_native_workflow_reuses_receipt_across_tool_calls_and_restart(
    tmp_path, status
):
    runtime, _, submission, context = await _runtime(tmp_path)
    identity = "workflow-conversation"
    origin = TaskOrigin(type="subagent", subagent_id="original-context")
    first = await runtime.prepare_workflow_execution(
        identity, "/train.py", context, request_id="first", origin=origin
    )
    store = FileProductStore(tmp_path / identity)
    request = store.load_workflow_execution_requests()["first"]
    store.save_workflow_execution_request(
        request.model_copy(
            update={
                "status": status,
                "task_id": "task-existing" if status == "submitted" else None,
                "revision": request.revision + 1,
            }
        ),
        expected_revision=request.revision,
    )
    await runtime.close()
    runtime = ConversationRuntime(
        tmp_path,
        {"pi": FakeAdapter("pi")},
        default_harness_id="pi",
        workflow_submission=submission,
    )
    try:
        replies = await asyncio.gather(
            *(
                runtime.prepare_workflow_execution(
                    identity,
                    "/train.py",
                    context,
                    request_id=f"new-call-{index}",
                    origin=TaskOrigin(type="subagent", subagent_id=f"fresh-{index}"),
                )
                for index in range(3)
            )
        )
        assert all(reply["request_id"] == first["request_id"] for reply in replies)
        assert all(reply["status"] == status and reply["reused"] for reply in replies)
        assert all(
            reply["task_id"] == (None if status != "submitted" else "task-existing")
            for reply in replies
        )
        assert len(store.load_workflow_execution_requests()) == 1
        assert not store.load_snapshot().pending_permissions
        assert len(submission.submissions) == 1
    finally:
        await runtime.close()


@pytest.mark.parametrize("same_request_id", [False, True])
async def test_duplicate_workflow_submissions_reuse_the_same_task(
    tmp_path, same_request_id
):
    runtime, _, submission, context = await _runtime(tmp_path)
    identity = "workflow-conversation"
    store = FileProductStore(tmp_path / identity)
    original_prepare = submission.prepare

    async def prepare(*args, **kwargs):
        await asyncio.sleep(0)
        return await original_prepare(*args, **kwargs)

    submission.prepare = prepare
    try:
        results = await asyncio.gather(
            *(
                runtime.prepare_workflow_execution(
                    identity,
                    "/train.py",
                    context,
                    request_id="same" if same_request_id else key,
                    origin=TaskOrigin(type="subagent", subagent_id="worker"),
                )
                for key in ("first", "second")
            )
        )
        assert len(submission.submissions) == 1
        assert all(reply["task_id"] == "task-123" for reply in results)
        assert results[1]["reused"] is True
        assert not store.load_snapshot().pending_permissions
    finally:
        await runtime.close()


async def test_native_submission_keeps_changed_files_and_non_child_requests_separate(
    tmp_path,
):
    runtime, _, submission, context = await _runtime(tmp_path)
    identity = "workflow-conversation"
    origin = TaskOrigin(type="subagent", subagent_id="one")
    try:
        for key, selected in (("child", origin), ("main", TaskOrigin())):
            reply = await runtime.prepare_workflow_execution(
                identity,
                "/train.py",
                context,
                request_id=key,
                origin=selected,
            )
            assert reply["request_id"] == key
        original = submission.prepare

        async def changed(workflow_path: str, context: RequestContext):
            return {
                **await original(workflow_path, context),
                "fingerprint": "new-content",
            }

        submission.prepare = changed
        reply = await runtime.prepare_workflow_execution(
            identity,
            "/train.py",
            context,
            request_id="changed",
            origin=origin,
        )
        assert reply["request_id"] == "changed"
    finally:
        await runtime.close()


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("harness_id", ["pi", "openhands"])
async def test_workflow_submits_directly_once_and_resumes_followup(
    tmp_path, training, harness_id
):
    runtime, adapter, submission, context = await _runtime(
        tmp_path, training=training, harness_id=harness_id
    )
    conversation_id = "workflow-conversation"
    prepared = await runtime.prepare_workflow_execution(
        conversation_id,
        "/workspace/.pyromind-agent/workflow.py",
        context,
        request_id="request-1",
    )
    assert prepared["status"] == "submitted"
    assert prepared["task_id"] == "task-123"
    assert not (
        await runtime.get_snapshot(conversation_id, context)
    ).pending_permissions
    assert not any(
        event.type.startswith("permission.")
        for event in FileProductStore(tmp_path / conversation_id).replay()
    )

    submission.workflow["name"] = "changed-after-prepare"
    repeated = await runtime.prepare_workflow_execution(
        conversation_id,
        "/workspace/.pyromind-agent/workflow.py",
        context,
        request_id="request-1",
    )
    assert repeated["task_id"] == "task-123"
    assert submission.submissions == [
        (
            {"name": "sft", "nodes": [{"data": {"nodeType": "SFTTrain"}}]},
            conversation_id,
        )
    ]
    assert adapter.sent == []
    assert len(adapter.external_task_notifications) == 1
    submitted = adapter.external_task_notifications[0][1]
    assert submitted.task_id == "task-123"
    assert submitted.trigger_turn is False
    assert submitted.visible_text is None
    reference = json.loads(submitted.hidden_text.split(": ", 1)[1])
    assert reference == {
        "request_id": "request-1",
        "task_id": "task-123",
        "workflow_path": "/workspace/.pyromind-agent/workflow.py",
        "status": "pending",
    }
    assert (
        await runtime.get_snapshot(conversation_id, context)
    ).pending_permissions == ()

    await runtime.deliver_external_task_status(
        conversation_id, task_id="task-123", status="Succeeded"
    )
    notification = adapter.external_task_notifications[-1][1]
    assert notification.trigger_turn is True
    if training:
        assert "training-analysis skill" in notification.hidden_text
        assert "inference-evaluation" in notification.hidden_text
        assert "evaluation was explicitly excluded" in notification.hidden_text
        assert "same-step complete checkpoints" in notification.hidden_text
        assert "do not start extra GPU inference" in notification.hidden_text
        assert "Do not rerun automatically" in notification.hidden_text
    else:
        assert "inference-evaluation" not in notification.hidden_text
        assert "without a training node" in notification.hidden_text
    assert "task-123" in notification.hidden_text
    product_text = (
        tmp_path / conversation_id / "product" / "workflow-execution-requests.json"
    ).read_text()
    assert "auth_token=ephemeral" not in product_text
    assert "permission_id" not in product_text
    assert json.loads(product_text)["request-1"]["task_id"] == "task-123"
    await runtime.close()


@pytest.mark.parametrize("fail_initial_sync", [False, True])
async def test_workflow_task_reference_is_recovered_before_next_user_turn(
    tmp_path, monkeypatch, fail_initial_sync
):
    runtime, adapter, submission, context = await _runtime(tmp_path)
    original_notify = adapter.notify_external_task

    async def unavailable(*args, **kwargs):
        raise RuntimeError("harness temporarily unavailable")

    if fail_initial_sync:
        monkeypatch.setattr(adapter, "notify_external_task", unavailable)
    result = await runtime.prepare_workflow_execution(
        "workflow-conversation",
        "/workspace/project/workflow.py",
        context,
        request_id="r",
    )
    assert result["status"] == "submitted"
    assert len(submission.submissions) == 1
    if fail_initial_sync:
        assert adapter.external_task_notifications == []
        monkeypatch.setattr(adapter, "notify_external_task", original_notify)
    else:
        await runtime.close()
        adapter = FakeAdapter("pi")
        runtime = ConversationRuntime(
            tmp_path, {"pi": adapter}, default_harness_id="pi"
        )

    await runtime.submit_command(
        "workflow-conversation",
        UserMessageCommand(
            command_id="check-task",
            content=(TextContent(text="查看执行情况"),),
        ),
        context,
    )
    assert len(adapter.external_task_notifications) == 1
    reference = adapter.external_task_notifications[0][1]
    assert reference.trigger_turn is False
    assert reference.task_id == "task-123"
    assert "/workspace/project/workflow.py" in reference.hidden_text
    assert adapter.sent[-1][1].command_id == "check-task"
    await runtime.close()


async def test_workflow_submission_failure_is_uncertain_and_not_retried(tmp_path):
    runtime, adapter, submission, context = await _runtime(tmp_path)
    attempts = 0

    async def failed_submit(
        workflow: dict,
        conversation_id: str,
        context: RequestContext,
        *,
        correlation: TaskCorrelation | None = None,
    ) -> str:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("platform create response lost")

    submission.submit = failed_submit
    for _ in range(2):
        result = await runtime.prepare_workflow_execution(
            "workflow-conversation",
            "/workspace/project/workflow.py",
            context,
            request_id="r",
        )
        assert result["status"] == "uncertain"
    assert attempts == 1
    assert submission.submissions == []
    assert not (
        await runtime.get_snapshot("workflow-conversation", context)
    ).pending_permissions
    await runtime.close()


async def test_failed_workflow_callback_routes_to_failure_analysis(tmp_path):
    runtime, adapter, _, context = await _runtime(tmp_path, training=False)
    await runtime.prepare_workflow_execution(
        "workflow-conversation",
        "/workspace/project/workflow.py",
        context,
        request_id="r",
    )
    await runtime.deliver_external_task_status(
        "workflow-conversation", task_id="task-123", status="Failed"
    )
    notification = adapter.external_task_notifications[-1][1]
    assert notification.trigger_turn is True
    assert "inspect_workflow_task" in notification.hidden_text
    assert "include_log=true" in notification.hidden_text
    assert "continue the user's task" in notification.hidden_text
    assert "with submit_workflow" in notification.hidden_text
    assert "user confirmation" not in notification.hidden_text
    assert "Do not edit or rerun" not in notification.hidden_text
    assert "then stop" not in notification.hidden_text
    assert (
        FileProductStore(tmp_path / "workflow-conversation")
        .load_snapshot()
        .external_tasks[0]
        .status
        == "failed"
    )
    await runtime.close()


async def test_callback_arriving_during_create_waits_for_task_registration(tmp_path):
    runtime, adapter, submission, context = await _runtime(tmp_path)
    callback: asyncio.Task | None = None

    async def submit_with_fast_callback(
        workflow: dict,
        conversation_id: str,
        context: RequestContext,
        *,
        correlation: TaskCorrelation | None = None,
    ) -> str:
        nonlocal callback
        callback = asyncio.create_task(
            runtime.deliver_external_task_status(
                conversation_id, task_id="task-123", status="Succeeded"
            )
        )
        await asyncio.sleep(0)
        return "task-123"

    submission.submit = submit_with_fast_callback
    await runtime.prepare_workflow_execution(
        "workflow-conversation",
        "/workspace/project/workflow.py",
        context,
        request_id="r",
    )
    assert callback is not None
    await callback
    assert (
        "training-analysis skill"
        in adapter.external_task_notifications[-1][1].hidden_text
    )
    await runtime.close()


async def test_pi_tool_exposes_direct_workflow_submission(tmp_path):
    skills = Path(__file__).parents[2] / ".agents" / "skills"
    host = PyromindBusinessToolHost(
        [skills / "data-processing", skills / "training-analysis"]
    )
    calls: list[tuple[str, str, str, str | None]] = []

    async def prepare(
        conversation_id, workflow_path, context, *, request_id, workflow_storage_path
    ):
        calls.append(
            (conversation_id, workflow_path, workflow_storage_path, request_id)
        )
        return {"request_id": request_id, "status": "submitted", "task_id": "task-123"}

    host.set_workflow_preparer(prepare)
    assert "submit_workflow" in {spec["name"] for spec in host.specs()}
    result = await host.execute(
        "submit_workflow",
        {"workflow_path": "public_data/workflow_canvas/workflow.py"},
        ToolExecutionContext(
            conversation_id="conversation-1",
            workspace_root=tmp_path,
            request_context=RequestContext(user_id="42"),
            model_configuration={},
        ),
        tool_call_id="call-1",
    )
    assert result["details"]["status"] == "submitted"
    assert result["details"]["task_id"] == "task-123"
    assert calls == [
        (
            "conversation-1",
            "public_data/workflow_canvas/workflow.py",
            "/workspace/.pyromind-agent/conversation-1/"
            "public_data/workflow_canvas/workflow.py",
            "call-1",
        )
    ]
