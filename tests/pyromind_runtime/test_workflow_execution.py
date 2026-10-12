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
from pyromind_runtime.domain.commands import PermissionResponseCommand
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter


class FakeSubmission:
    def __init__(self, *, training: bool = True) -> None:
        self.training = training
        self.workflow = {"name": "sft", "nodes": [{"data": {"nodeType": "SFTTrain"}}]}
        self.submissions: list[tuple[dict, str]] = []

    async def prepare(self, workflow_path: str, context: RequestContext) -> dict:
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
        self, workflow: dict, conversation_id: str, context: RequestContext
    ) -> str:
        self.submissions.append((workflow, conversation_id))
        return "task-123"


async def _runtime(tmp_path, *, training: bool = True):
    adapter = FakeAdapter("pi")
    submission = FakeSubmission(training=training)
    runtime = ConversationRuntime(
        tmp_path,
        {"pi": adapter},
        default_harness_id="pi",
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


async def test_confirmed_workflow_submits_once_and_resumes_training_analysis(tmp_path):
    runtime, adapter, submission, context = await _runtime(tmp_path)
    conversation_id = "workflow-conversation"
    prepared = await runtime.prepare_workflow_execution(
        conversation_id,
        "storage/.pyromind-agent/workflow.py",
        context,
        request_id="request-1",
    )
    assert prepared["status"] == "awaiting_confirmation"
    assert submission.submissions == []
    assert (await runtime.get_snapshot(conversation_id, context)).pending_permissions

    # Approval runs the definition that the user reviewed, even if the source
    # workflow changes in Storage while the confirmation dialog is open.
    submission.workflow["name"] = "changed-after-prepare"
    receipt = await runtime.submit_command(
        conversation_id,
        PermissionResponseCommand(
            command_id="confirm-1",
            permission_id="workflow-run:request-1",
            decision="allow_once",
        ),
        context,
    )
    assert receipt.response["task_id"] == "task-123"
    assert submission.submissions == [
        (
            {"name": "sft", "nodes": [{"data": {"nodeType": "SFTTrain"}}]},
            conversation_id,
        )
    ]
    assert adapter.sent == []
    assert (
        await runtime.get_snapshot(conversation_id, context)
    ).pending_permissions == ()

    repeated = await runtime.submit_command(
        conversation_id,
        PermissionResponseCommand(
            command_id="confirm-2",
            permission_id="workflow-run:request-1",
            decision="allow_once",
        ),
        context,
    )
    assert repeated.response["task_id"] == "task-123"
    assert len(submission.submissions) == 1

    await runtime.deliver_external_task_status(
        conversation_id, task_id="task-123", status="Succeeded"
    )
    notification = adapter.external_task_notifications[-1][1]
    assert notification.trigger_turn is True
    assert "training-analysis skill" in notification.hidden_text
    assert "task-123" in notification.hidden_text
    product_text = (
        tmp_path / conversation_id / "product" / "workflow-execution-requests.json"
    ).read_text()
    assert "auth_token=ephemeral" not in product_text
    assert json.loads(product_text)["request-1"]["task_id"] == "task-123"
    await runtime.close()


@pytest.mark.parametrize("decision", ["deny", "allow_once"])
async def test_workflow_confirmation_cancel_or_failure(tmp_path, decision):
    runtime, adapter, submission, context = await _runtime(tmp_path)
    await runtime.prepare_workflow_execution(
        "workflow-conversation", "storage/project/workflow.py", context, request_id="r"
    )
    if decision == "allow_once":

        async def failed_submit(
            workflow: dict, conversation_id: str, context: RequestContext
        ) -> str:
            raise TimeoutError("platform create response lost")

        submission.submit = failed_submit
    receipt = await runtime.submit_command(
        "workflow-conversation",
        PermissionResponseCommand(
            command_id="answer", permission_id="workflow-run:r", decision=decision
        ),
        context,
    )
    assert receipt.response["status"] == (
        "cancelled" if decision == "deny" else "uncertain"
    )
    assert submission.submissions == []
    assert not (
        await runtime.get_snapshot("workflow-conversation", context)
    ).pending_permissions
    await runtime.close()


async def test_failed_workflow_callback_routes_to_failure_analysis(tmp_path):
    runtime, adapter, _, context = await _runtime(tmp_path, training=False)
    await runtime.prepare_workflow_execution(
        "workflow-conversation", "storage/project/workflow.py", context, request_id="r"
    )
    await runtime.submit_command(
        "workflow-conversation",
        PermissionResponseCommand(
            command_id="answer", permission_id="workflow-run:r", decision="allow_once"
        ),
        context,
    )
    await runtime.deliver_external_task_status(
        "workflow-conversation", task_id="task-123", status="Failed"
    )
    notification = adapter.external_task_notifications[-1][1]
    assert notification.trigger_turn is True
    assert "analyze_task_failure" in notification.hidden_text
    assert "Do not edit or rerun" in notification.hidden_text
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
    await runtime.prepare_workflow_execution(
        "workflow-conversation", "storage/project/workflow.py", context, request_id="r"
    )
    callback: asyncio.Task | None = None

    async def submit_with_fast_callback(
        workflow: dict, conversation_id: str, context: RequestContext
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
    await runtime.submit_command(
        "workflow-conversation",
        PermissionResponseCommand(
            command_id="answer", permission_id="workflow-run:r", decision="allow_once"
        ),
        context,
    )
    assert callback is not None
    await callback
    assert (
        "training-analysis skill"
        in adapter.external_task_notifications[-1][1].hidden_text
    )
    await runtime.close()


async def test_pi_tool_exposes_confirmed_workflow_entry(tmp_path):
    skills = Path(__file__).parents[2] / ".agents" / "skills"
    host = PyromindBusinessToolHost(
        [skills / "data-processing", skills / "training-analysis"]
    )
    calls: list[tuple[str, str, str | None]] = []

    async def prepare(conversation_id, workflow_path, context, *, request_id):
        calls.append((conversation_id, workflow_path, request_id))
        return {"request_id": request_id, "status": "awaiting_confirmation"}

    host.set_workflow_preparer(prepare)
    assert "submit_workflow" in {spec["name"] for spec in host.specs()}
    result = await host.execute(
        "submit_workflow",
        {"workflow_path": "storage/project/workflow.py"},
        ToolExecutionContext(
            conversation_id="conversation-1",
            workspace_root=tmp_path,
            request_context=RequestContext(user_id="42"),
            model_configuration={},
        ),
        tool_call_id="call-1",
    )
    assert result["details"]["status"] == "awaiting_confirmation"
    assert calls == [("conversation-1", "storage/project/workflow.py", "call-1")]
