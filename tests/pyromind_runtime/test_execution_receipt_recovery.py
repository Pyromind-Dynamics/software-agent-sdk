"""Recover persisted submissions after a server restart without live credentials."""

import asyncio
from dataclasses import replace
from uuid import UUID

import pytest
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from .fake_adapter import FakeAdapter
from .test_standalone_experiments import claim, experiment as experiment, request


class ReceiptAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.receipt_reads = []

    async def read_submission_receipt(self, handle, tool_name, operation_id, context):
        self.receipt_reads.append((handle.session_id, tool_name, operation_id, context))
        return {"task_id": f"platform-{handle.session_id}", "output_dir": "/outputs"}


@pytest.fixture
async def recovery(tmp_path, experiment):
    service, owner, _ = experiment
    adapter = ReceiptAdapter()
    queried = asyncio.Queue()

    class TaskQuery:
        async def query(self, task_id: str, context: RequestContext) -> JsonObject:
            queried.put_nowait((task_id, context))
            return {"status": "running"}

    runtime = ConversationRuntime(
        tmp_path,
        adapter,
        experiment_plan_store=service.plans,
        experiment_execution_store=service.store,
        external_task_query=TaskQuery(),
    )
    try:
        yield runtime, adapter, service, owner, queried
    finally:
        await runtime.close()


def pending_submission(runtime, service, owner):
    directory = runtime.conversation_root / owner.conversation_id
    directory.mkdir()
    product = FileProductStore(directory)
    product.create(
        ConversationSnapshot(
            conversation_id=owner.conversation_id,
            capabilities=FakeAdapter.capabilities,
        ),
        user_id=owner.user_id,
        profile_id=owner.profile_id,
    )
    run = service.request(owner, request(), ("eval",))
    scope = claim(service, owner, run)
    service.store.managed_submission(
        owner, run["id"], scope, {"operation": "inference"}
    )
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
    return product, service.store.list(owner)[0]


async def activate(runtime, owner):
    context = RequestContext(
        user_id=owner.user_id,
        profile_id=owner.profile_id,
        authorization="Bearer renewed-test-token",
    )
    await runtime.submit_command(
        owner.conversation_id,
        UserMessageCommand(
            command_id="continue", content=(TextContent(text="Continue"),)
        ),
        context,
    )
    return context


async def test_refresh_defers_dormant_receipt_and_resumes_after_authorized_command(
    recovery, caplog
):
    runtime, adapter, service, dormant, queried = recovery
    pending_submission(runtime, service, dormant)
    active = dormant.model_copy(update={"conversation_id": "active-conversation"})
    pending_submission(runtime, service, active)
    active_context = await activate(runtime, active)

    # Startup marks interrupted submissions uncertain before the first refresh.
    await runtime.start_execution_events()
    assert await asyncio.wait_for(queried.get(), 3) == (
        f"platform-{active.conversation_id}",
        active_context,
    )
    assert service.store.list(dormant)[0]["state"] == "submission_uncertain"
    assert service.store.list(active)[0]["state"] == "waiting_platform"
    assert adapter.attached == [active.conversation_id]
    assert [read[0] for read in adapter.receipt_reads] == [active.conversation_id]

    dormant_context = await activate(runtime, dormant)
    assert await asyncio.wait_for(queried.get(), 3) == (
        f"platform-{dormant.conversation_id}",
        dormant_context,
    )
    assert service.store.list(dormant)[0]["state"] == "waiting_platform"
    assert [read[0] for read in adapter.receipt_reads] == [
        active.conversation_id,
        dormant.conversation_id,
    ]
    assert "Execution refresh delivery failed" not in caplog.text


@pytest.mark.parametrize("source", ["pipeline", "workflow"])
async def test_persisted_receipt_recovers_without_activating_session(recovery, source):
    runtime, adapter, service, owner, _ = recovery
    product, run = pending_submission(runtime, service, owner)
    service.store.recover_submissions()
    submission = run["submissions"][0]
    if source == "pipeline":
        product.save_pipeline_run(
            PipelineRun(
                run_id=UUID(submission["evidence"]["pipeline_run_id"]),
                conversation_id=owner.conversation_id,
                task_run_id=run["id"],
                definition={},
                fingerprint="persisted-definition",
                task_id="persisted-platform-task",
                output_dir="/outputs/persisted",
                submission_status="submitted",
            ),
            expected_revision=None,
        )
    else:
        from pyromind_runtime.infrastructure.sql_experiment_execution import (
            submissions,
        )

        with service.store.engine.begin() as connection:
            connection.execute(
                submissions.update()
                .where(submissions.c.id == submission["id"])
                .values(
                    evidence={
                        **submission["evidence"],
                        "workflow_request_id": "workflow-request",
                    }
                )
            )
        product.save_workflow_execution_request(
            WorkflowExecutionRequest(
                request_id="workflow-request",
                conversation_id=owner.conversation_id,
                permission_id="workflow-permission",
                workflow_path="/workflow.py",
                fingerprint="persisted-definition",
                workflow={},
                summary={},
                task_run_id=run["id"],
                task_id="persisted-platform-task",
                status="submitted",
            ),
            expected_revision=None,
        )

    await runtime._recover_execution_receipts()
    await runtime._recover_execution_receipts()

    recovered = service.store.list(owner)[0]
    assert recovered["state"] == "waiting_platform"
    assert recovered["submission"]["task_id"] == "persisted-platform-task"
    assert adapter.attached == []
    assert adapter.receipt_reads == []


@pytest.mark.parametrize("field", ["user_id", "profile_id"])
async def test_receipt_wait_does_not_allow_another_owner_to_activate(recovery, field):
    runtime, adapter, service, owner, _ = recovery
    pending_submission(runtime, service, owner)
    service.store.recover_submissions()
    await runtime._recover_execution_receipts()

    with pytest.raises(PermissionError):
        await activate(runtime, owner.model_copy(update={field: "other"}))

    assert service.store.list(owner)[0]["state"] == "submission_uncertain"
    assert adapter.attached == []
    assert adapter.receipt_reads == []


@pytest.mark.parametrize("field", ["user_id", "profile_id"])
async def test_receipt_reader_rejects_a_mismatched_active_context(recovery, field):
    runtime, adapter, service, owner, _ = recovery
    pending_submission(runtime, service, owner)
    service.store.recover_submissions()
    context = await activate(runtime, owner)
    runtime._active[owner.conversation_id].context = replace(
        context, **{field: "other"}
    )

    with pytest.raises(PermissionError, match="delegated work owner mismatch"):
        await runtime._recover_execution_receipts()

    assert service.store.list(owner)[0]["state"] == "submission_uncertain"
    assert adapter.receipt_reads == []
