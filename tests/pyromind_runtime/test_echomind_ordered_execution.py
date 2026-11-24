import asyncio
import json
from hashlib import sha256

import pytest

from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.echomind_chat import completed_chat_turns
from pyromind_runtime.application.echomind_notices import (
    candidate_for,
    manager_messages,
    public_blocks,
    resolved_submission_notices,
    visible_chat_event,
)
from pyromind_runtime.application.plan_execution import PLAN_PATH, next_approved_task
from pyromind_runtime.application.preparation import Preparations
from pyromind_runtime.domain.commands import CreateProjectCommand, UserMessageCommand
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_chat import CHAT_PROFILE
from pyromind_runtime.domain.events import HarnessEvent, ProductEvent, ProductEventType
from pyromind_runtime.domain.preparation import ProjectView, SetPreparationPlan
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    ExecutionNoticeLink,
    ExternalTaskState,
    TimelineMessage,
    TimelineOperation,
)
from pyromind_runtime.ports.harness import DelegatedRole, SessionSpec

from .fake_adapter import FakeAdapter
from .test_preparation import preparation_setup as preparation_setup


async def wait_for(check):
    async with asyncio.timeout(8):
        while not check():
            await asyncio.sleep(0.02)


def plan_request():
    return SetPreparationPlan.model_validate(
        {
            "entry_task_id": "data",
            "plan": {
                "goal": "Prepare data and use it in a baseline",
                "resources": [
                    {
                        "id": "source",
                        "name": "source",
                        "storage_path": "/workspace/source.jsonl",
                    }
                ],
                "stages": [{"id": "stage", "kind": "measure", "name": "Test"}],
                "tasks": [
                    {
                        "id": "data",
                        "stage_id": "stage",
                        "kind": "score",
                        "name": "Prepare",
                        "inputs": [{"name": "dataset", "resource_id": "source"}],
                        "outputs": [{"name": "samples"}],
                    },
                    {
                        "id": "baseline",
                        "stage_id": "stage",
                        "kind": "score",
                        "name": "Baseline",
                        "inputs": [
                            {
                                "name": "dataset",
                                "task_id": "data",
                                "output_name": "samples",
                            }
                        ],
                    },
                ],
            },
        }
    )


def local_completion():
    return {
        "id": "data",
        "stage_id": "stage",
        "business_status": "completed",
        "verified_outputs": {"samples": "/workspace/prepared.jsonl"},
    }


def test_rejected_submission_can_retry_without_requesting_a_user_decision():
    event = ProductEvent(
        conversation_id="session",
        type="experiment_execution.changed",
        payload={"execution": {"state": "failed", "run_id": "baseline"}},
    )
    snapshot = ConversationSnapshot.model_validate(
        {
            "conversation_id": "session",
            "capabilities": {},
            "status": "running",
        }
    )
    assert candidate_for(event, snapshot, set()) is None
    settled = snapshot.model_copy(update={"status": "idle"})
    assert candidate_for(event, settled, set()) is not None


def test_stage_and_final_messages_use_public_boundaries():
    report = "[评测报告](https://example.org/r?x=1&y=2)\n\n格式示例 `<think>`。"

    def blocks(text: str):
        return (TextContent(text=text),)

    for final in (False, True):
        assert not public_blocks(
            blocks('private prefix with "<think>example" and private notes</think>\n'),
            publish=final,
        )
        assert not public_blocks(blocks("private prefix</think>"), publish=final)
        assert not public_blocks(blocks("<think>unfinished"), publish=final)
    assert not public_blocks(
        blocks("arbitrary unmarked tool commentary"), publish=False
    )
    assert public_blocks(blocks("private</think>" + report), publish=True) == blocks(
        report
    )
    assert public_blocks(blocks(report), publish=True) == blocks(report)
    sample = report + '\n~~~json\n{"format":"<think>example</think>"}\n~~~\n'
    assert public_blocks(blocks(sample), publish=True) == blocks(sample)
    event = ProductEvent(
        conversation_id="chat",
        type="message.completed",
        payload={
            "content": [{"type": "text", "text": "old leaked copy"}],
            "execution_notice": {"source_message_id": "legacy"},
        },
    )
    assert visible_chat_event(event).payload["content"] == []


def test_approved_continuation_requires_proven_local_outputs():
    project = ProjectView(
        project_id="project",
        conversation_id="session",
        plan_sequence=1,
        creation_command_id="create",
        plan=plan_request().plan,
    )
    done = local_completion()
    ready = next_approved_task(project, [done], "data")
    assert ready is not None and ready[0].id == "baseline"
    assert ready[1] == {"dataset": "/workspace/prepared.jsonl"}
    assert (
        next_approved_task(
            project, [{**done, "verified_outputs": {"other": "x"}}], "data"
        )
        is None
    )
    assert (
        next_approved_task(
            project, [{**done, "platform_task_id": "already-submitted"}], "data"
        )
        is None
    )
    assert (
        next_approved_task(
            project, [done, {"id": "baseline", "business_status": "blocked"}], "data"
        )
        is None
    )


async def test_sandbox_read_recovery_requires_login_and_notifies_original_turn_once(
    preparation_setup, tmp_path
):
    preparations, owner, policy = preparation_setup
    cid = owner.conversation_id
    viewer = RequestContext(user_id=owner.user_id, profile_id="echomind")
    signed_in = RequestContext(
        user_id=owner.user_id, profile_id="echomind", authorization="Bearer fixture"
    )
    refreshed = []

    class SandboxAdapter(FakeAdapter):
        def can_recover_on_read(self, context):
            return bool(context.authorization)

        def refresh_execution_context(self, handle, context):
            refreshed.append((handle.session_id, context.authorization))

    adapter = SandboxAdapter()
    runtime = ConversationRuntime(tmp_path, adapter, preparation_store=preparations)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid,
                user_id=owner.user_id,
                profile_id="echomind",
                workspace_root=str(tmp_path / cid),
                preparation_policy=policy,
                delegated_roles=(DelegatedRole(name="eval"),),
            ),
            signed_in,
        )
        chat = await runtime.echomind_chat.open(cid, {}, signed_in)
        await runtime._release(cid, reason="test")
        store = runtime._store(cid)
        task = "project-start:fixture"
        runtime._queue_task(
            store, cid, task, "project_start", "Inspect", origin_turn_id="turn"
        )
        await runtime.get_snapshot(cid, viewer)
        await runtime._read_store(cid, viewer)
        assert cid not in runtime._active
        await runtime.get_snapshot(cid, signed_in)
        await wait_for(
            lambda: cid in runtime._active and runtime._active[cid].ready.is_set()
        )
        await runtime.get_snapshot(cid, signed_in)
        assert (cid, "Bearer fixture") in refreshed
        for seq in range(2):
            event, _ = store.append(
                ProductEvent(
                    conversation_id=cid,
                    task_id=task,
                    run_id="run",
                    type="operation.failed",
                    payload={
                        "operation_id": f"term-{seq}",
                        "name": "terminal",
                        "error_code": "sandbox_authorization_required",
                    },
                )
            )
            runtime._publish(event)
        await wait_for(lambda: not runtime.echomind_notices.workers)
        notices = [
            m
            for m in runtime._store(chat.conversation_id).load_snapshot().timeline
            if isinstance(m, TimelineMessage) and m.execution_notice
        ]
        assert len(notices) == 1
        notice = notices[0]
        assert (
            notice.execution_notice and notice.execution_notice.origin_turn_id == "turn"
        )
        block = notice.content[0]
        assert isinstance(block, TextContent) and "沙箱需要登录" in block.text
    finally:
        await runtime.close()


@pytest.fixture
async def started_echomind_project(preparation_setup, tmp_path):
    preparations, owner, old_policy = preparation_setup
    owner = owner.model_copy(
        update={"profile_id": "echomind", "conversation_id": "recover-plan"}
    )
    preparations.initialize(owner)
    policy = old_policy.model_copy(
        update={
            "allowed_roles": (),
            "workflow_nodes": tuple(
                node.model_copy(update={"roles": ()})
                for node in old_policy.workflow_nodes
            ),
        }
    )
    cid = owner.conversation_id
    context = RequestContext(
        user_id=owner.user_id,
        profile_id=owner.profile_id,
        x_cluster="test",
        cookie="auth_token=test",
    )
    adapter = FakeAdapter()
    runtime = ConversationRuntime(tmp_path, adapter, preparation_store=preparations)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid,
                user_id=context.user_id,
                profile_id=context.profile_id,
                workspace_root=str(tmp_path / cid),
                preparation_policy=policy,
                command_allowlist=("create_project", "enqueue_task"),
            ),
            context,
        )
        runtime._queue_task(
            runtime._store(cid),
            cid,
            "initial",
            "requirement",
            "Initial plan",
            origin_turn_id="original",
        )
        runtime._mark_task_dispatched(runtime._store(cid), cid, "initial")
        runtime._store(cid).append(
            ProductEvent(
                conversation_id=cid,
                type="status.changed",
                task_id="initial",
                run_id="initial-run",
                payload={"status": "idle"},
            )
        )
        plan = Preparations(preparations).set_plan(owner, plan_request(), policy, 0)
        receipt = await runtime.submit_command(
            cid,
            CreateProjectCommand(
                command_id="create",
                expected_preparation_revision=plan.revision,
            ),
            context,
        )
        await wait_for(lambda: bool(adapter.external_task_notifications))
        first = adapter.external_task_notifications[0][1]
        assert first.kind == "project_start"
        started = HarnessEvent(
            session_id=cid,
            type="status.changed",
            run_id="first-run",
            task_id=first.task_id,
            payload={"status": "running"},
        )
        adapter.queues[cid].put_nowait(started)
        store = runtime._store(cid)
        await wait_for(lambda: store.load_snapshot().status == "running")
        yield (
            runtime,
            adapter,
            store,
            context,
            first.task_id,
            receipt.response["project_id"],
        )
    finally:
        await runtime.close()


async def test_local_project_continuation_precedes_a_later_requirement(
    started_echomind_project, tmp_path
):
    runtime, adapter, store, _, first_id, project_id = started_echomind_project
    cid = store.load_snapshot().conversation_id
    runtime._queue_task(
        store, cid, "requirement:later", "requirement", "Inspect something new"
    )
    (tmp_path / cid / PLAN_PATH).write_text(json.dumps({"tasks": [local_completion()]}))
    adapter.queues[cid].put_nowait(
        HarnessEvent(
            session_id=cid,
            type="run.finished",
            run_id="first-run",
            task_id=first_id,
            payload={"status": "idle"},
        )
    )
    await wait_for(lambda: len(adapter.external_task_notifications) >= 2)
    continuation = adapter.external_task_notifications[1][1]
    assert continuation.task_id == f"project-continue:{project_id}:baseline"
    assert "/workspace/prepared.jsonl" in continuation.hidden_text
    assert adapter.external_task_notifications[1][1].kind == "project_continue"
    await wait_for(lambda: len(adapter.external_task_notifications) >= 3)
    assert adapter.external_task_notifications[2][1].task_id == "requirement:later"


async def test_authenticated_read_recovers_a_missed_wrapped_plan_continuation(
    started_echomind_project, tmp_path
):
    runtime, adapter, store, context, first_id, project_id = started_echomind_project
    cid = store.load_snapshot().conversation_id
    # Older versions wrote the final status without projecting this layout.
    store.append(
        ProductEvent(
            conversation_id=cid,
            run_id="first-run",
            task_id=first_id,
            type="status.changed",
            payload={"status": "idle"},
        )
    )
    (tmp_path / cid / PLAN_PATH).write_text(
        json.dumps(
            {
                "plan": {
                    "tasks": [
                        {
                            "id": "data",
                            "stage_id": "stage",
                            "outputs": [{"name": "samples"}],
                        }
                    ]
                },
                "task_states": {
                    "data": {
                        "business_status": "completed",
                        "verified_outputs": ["/workspace/samples.jsonl"],
                    }
                },
            }
        )
    )
    for _ in range(2):
        await runtime.get_snapshot(cid, context)
        await wait_for(lambda: not runtime._read_recoveries)
    continuations = [
        n
        for _, n, _ in adapter.external_task_notifications
        if n.kind == "project_continue"
    ]
    assert len(continuations) == 1
    assert continuations[0].task_id == f"project-continue:{project_id}:baseline"
    assert "/workspace/samples.jsonl" in continuations[0].hidden_text


async def test_historical_pre_submission_blocker_is_corrected_on_the_same_bubble(
    started_echomind_project,
):
    _, _, store, _, _, _ = started_echomind_project
    snapshot = store.load_snapshot()
    step = snapshot.experiment_executions[0]
    mid = "execution-notice:" + sha256(f"run:{step.run_id}:failed".encode()).hexdigest()
    message = TimelineMessage(
        item_id=mid,
        started_seq=1,
        role="assistant",
        content=(TextContent(text="请决定是否重试。"),),
        execution_notice=ExecutionNoticeLink(origin_turn_id="original"),
    )
    chat = snapshot.model_copy(update={"timeline": (message,)})
    assert not resolved_submission_notices(snapshot, chat)
    manager = snapshot.model_copy(
        update={
            "experiment_executions": (
                step.model_copy(update={"platform_task_id": "123"}),
            )
        }
    )
    [update] = resolved_submission_notices(manager, chat)
    assert update.payload["message_id"] == mid
    assert update.payload["execution_notice"] == {"origin_turn_id": "original"}
    corrected = message.model_copy(
        update={
            "content": (TextContent(text="此前提交遇到问题，之后已成功提交到平台。"),)
        }
    )
    assert not resolved_submission_notices(
        manager, chat.model_copy(update={"timeline": (corrected,)})
    )


async def test_direct_project_resume_requires_a_receipt_bound_to_the_original_project(
    started_echomind_project,
):
    _, _, store, _, _, _ = started_echomind_project
    snapshot = store.load_snapshot()
    step = snapshot.experiment_executions[0].model_copy(
        update={"platform_task_id": "1294"}
    )
    direct = "confirmed-resume"
    op = TimelineOperation(
        run_id=direct,
        item_id="submit",
        started_seq=4,
        completed_seq=5,
        status="completed",
        name="df_submit_evaluation",
        category="tool",
        details={"task_id": "1294"},
    )
    message = TimelineMessage(
        run_id=direct,
        item_id="result",
        started_seq=6,
        completed_seq=7,
        role="assistant",
        status="completed",
        content=(TextContent(text="已提交评测任务 1294。"),),
    )
    event = ProductEvent(
        conversation_id=snapshot.conversation_id,
        seq=8,
        type="status.changed",
        run_id=direct,
        payload={"status": "idle"},
    )

    def notice(operation):
        return candidate_for(
            event,
            snapshot.model_copy(
                update={
                    "timeline": (operation, message),
                    "experiment_executions": (step,),
                }
            ),
            set(),
        )

    result = notice(op)
    assert result and result.link.origin_turn_id == "original"
    chat = snapshot.model_copy(
        update={"conversation_id": "echomind-chat-test", "timeline": ()}
    )
    original, _ = manager_messages(
        snapshot.model_copy(update={"timeline": (op, message)}),
        chat,
        7,
        0,
        frozenset({direct}),
    )
    assert original[7].payload["content"] == [message.content[0].model_dump()]
    assert notice(op.model_copy(update={"details": {"task_id": "another"}})) is None
    assert notice(op.model_copy(update={"status": "failed"})) is None


async def test_saved_successful_callback_projects_its_output_before_dispatch(
    started_echomind_project, tmp_path
):
    runtime, adapter, store, context, _, project_id = started_echomind_project
    cid = store.load_snapshot().conversation_id
    run = store.load_snapshot().experiment_executions[0]
    runtime._record_execution(
        store, run.model_copy(update={"platform_task_id": "1293", "state": "preparing"})
    )
    store.append(
        ProductEvent(
            conversation_id=cid,
            type="status.changed",
            run_id="first-run",
            task_id=f"project-start:{project_id}",
            payload={"status": "idle"},
        )
    )
    receipt = ExternalTaskState(
        task_id="1293",
        kind="workflow_run",
        status="pending",
        submitted_at="2026-11-23T00:00:00Z",
        updated_at="2026-11-23T00:01:00Z",
    )
    for status in ("pending", "succeeded"):
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="external_task.submitted"
                if status == "pending"
                else "external_task.completed",
                payload={**receipt.model_dump(mode="json"), "status": status},
            )
        )
    for status in ("running", "idle"):
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="status.changed",
                task_id="1293",
                run_id="callback-run",
                payload={"status": status},
            )
        )
    plan_path = tmp_path / cid / PLAN_PATH
    row = {**local_completion(), "actual_platform_task_id": "1293"}
    plan_path.write_text(
        json.dumps({"tasks": [{**row, "business_status": "in_progress"}]})
    )
    await runtime.get_snapshot(cid, context)
    await wait_for(lambda: not runtime._read_recoveries)
    assert not any(
        n.kind == "project_continue" for _, n, _ in adapter.external_task_notifications
    )
    # The plan file has now caught up to the successful external callback.
    plan_path.write_text(json.dumps({"tasks": [row]}))
    run_seq, retry_after = runtime._plan_recovery_checks[cid]
    assert retry_after < float("inf")
    runtime._plan_recovery_checks[cid] = (run_seq, 0)
    for _ in range(2):
        await runtime.get_snapshot(cid, context)
        await wait_for(lambda: not runtime._read_recoveries)
    assert [
        n.task_id
        for _, n, _ in adapter.external_task_notifications
        if n.kind == "project_continue"
    ] == [f"project-continue:{project_id}:baseline"]


async def test_successful_bound_platform_callback_continues_only_with_verified_output(
    started_echomind_project,
):
    runtime, adapter, store, context, _, project_id = started_echomind_project
    cid = store.load_snapshot().conversation_id
    project = await runtime.get_project(cid, context)
    assert project
    active = runtime._active[cid]
    row = {**local_completion(), "actual_platform_task_id": "1290"}
    registered = (
        store.load_snapshot()
        .experiment_executions[0]
        .model_copy(update={"platform_task_id": "1290", "state": "succeeded"})
    )
    runtime._record_execution(store, registered)
    for status in ("pending", "succeeded"):
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="external_task.submitted"
                if status == "pending"
                else "external_task.completed",
                payload=ExternalTaskState(
                    task_id="1290",
                    submitted_at="2026-11-23T00:00:00Z",
                    updated_at="2026-11-23T00:01:00Z",
                    kind="data_preparation",
                    status=status,
                    run_id="platform-pipeline-id",
                ).model_dump(mode="json"),
            )
        )
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="status.changed",
                run_id="callback-run",
                task_id="1290",
                payload={"status": "running"},
            )
        )
        await runtime._continue_approved_project(
            active,
            store,
            "callback-run",
            "1290",
            [{**row, "actual_platform_task_id": "different"}],
        )
        assert not store.load_snapshot().experiment_queue
        await runtime._continue_approved_project(
            active, store, "callback-run", "1290", [row]
        )
        queued = list(store.load_snapshot().experiment_queue or ())
        if status == "pending":
            assert not queued
        else:
            assert [q["task_id"] for q in queued] == [
                f"project-continue:{project_id}:baseline"
            ]
            assert (
                queued[0]["origin_turn_id"]
                == store.load_snapshot().task_origins[f"project-start:{project_id}"]
            )


async def test_adapter_queued_receipt_keeps_oldest_requirement_at_head(tmp_path):
    class Adapter(FakeAdapter):
        ready_to_start = False

        async def notify_external_task(
            self, handle, notification, context
        ) -> JsonObject:
            await super().notify_external_task(handle, notification, context)
            return {"accepted": True, "queued": not self.ready_to_start}

    adapter = Adapter()
    runtime = ConversationRuntime(tmp_path, adapter)
    cid = "queue-order"
    context = RequestContext(user_id="owner")
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid, user_id="owner", workspace_root=str(tmp_path / cid)
            ),
            context,
        )
        store = runtime._store(cid)
        active = runtime._active[cid]
        for task in ("older", "newer"):
            runtime._queue_task(store, cid, task, "requirement", task)
        await runtime._dispatch_queued_tasks(active, store)
        await runtime._dispatch_queued_tasks(active, store)
        assert [n.task_id for _, n, _ in adapter.external_task_notifications] == [
            "older",
            "older",
        ]
        assert [
            item["task_id"] for item in store.load_snapshot().experiment_queue or ()
        ] == ["older", "newer"]
        adapter.ready_to_start = True
        await runtime._dispatch_queued_tasks(active, store)
        assert [
            item["task_id"] for item in store.load_snapshot().experiment_queue or ()
        ] == ["newer"]
    finally:
        await runtime.close()


async def test_chat_accepts_messages_durably_and_starts_one_at_a_time(tmp_path):
    cid = "echomind-chat-order"
    context = RequestContext(user_id="owner", profile_id=CHAT_PROFILE)
    adapter = FakeAdapter()
    runtime = ConversationRuntime(tmp_path, adapter)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid,
                user_id="owner",
                profile_id=CHAT_PROFILE,
                workspace_root=str(tmp_path / cid),
                command_allowlist=("user_message",),
            ),
            context,
        )
        store = runtime._store(cid)
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="execution.policy.updated",
                payload={"automatic_turns_suspended": True},
            )
        )
        for request in ("A", "B", "C"):
            receipt = await runtime.submit_command(
                cid,
                UserMessageCommand(
                    command_id=request, content=(TextContent(text=request),)
                ),
                context,
            )
            assert receipt.response["queued"]
        store = runtime._store(cid)
        assert [cmd.command_id for _, cmd, _ in adapter.sent] == ["A"]
        assert not store.load_snapshot().automatic_turns_suspended
        assert [
            item["task_id"] for item in store.load_snapshot().experiment_queue or ()
        ] == [
            "chat-message:A",
            "chat-message:B",
            "chat-message:C",
        ]
        for index, request in enumerate(("A", "B", "C"), 1):
            await wait_for(lambda: len(adapter.sent) == index)
            adapter.emit(
                cid,
                "status.changed",
                {"status": "running"},
                event_id=f"{request}:start",
                run_id=request,
            )
            adapter.emit(
                cid,
                "message.started",
                {
                    "role": "user",
                    "command_id": request,
                    "message_id": f"message:{request}",
                },
                event_id=f"{request}:message",
                run_id=request,
            )
            await wait_for(
                lambda: all(
                    item["task_id"] != f"chat-message:{request}"
                    for item in store.load_snapshot().experiment_queue or ()
                )
            )
            assert len(adapter.sent) == index
            adapter.emit(
                cid,
                "run.finished",
                {"status": "idle"},
                event_id=f"{request}:done",
                run_id=request,
            )
        assert [cmd.command_id for _, cmd, _ in adapter.sent] == ["A", "B", "C"]
    finally:
        await runtime.close()


async def test_chat_recovery_uses_recorded_user_message_as_dispatch_ack(tmp_path):
    cid = "echomind-chat-recovery"
    context = RequestContext(user_id="owner", profile_id=CHAT_PROFILE)
    adapter = FakeAdapter()
    runtime = ConversationRuntime(tmp_path, adapter)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid,
                user_id="owner",
                profile_id=CHAT_PROFILE,
                workspace_root=str(tmp_path / cid),
                command_allowlist=("user_message",),
            ),
            context,
        )
        for request in ("A", "B"):
            await runtime.submit_command(
                cid,
                UserMessageCommand(
                    command_id=request, content=(TextContent(text=request),)
                ),
                context,
            )
        store = runtime._store(cid)
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="message.completed",
                run_id="A",
                turn_id="A",
                payload={
                    "role": "user",
                    "message_id": "message:A",
                    "command_id": "A",
                    "content": [{"type": "text", "text": "A"}],
                },
            )
        )
    finally:
        await runtime.close()

    restored_adapter = FakeAdapter()
    restored = ConversationRuntime(tmp_path, restored_adapter)
    try:
        await restored.get_snapshot(cid, context)
        await wait_for(lambda: bool(restored_adapter.sent))
        assert [cmd.command_id for _, cmd, _ in restored_adapter.sent] == ["B"]
        assert [
            item["task_id"] for item in store.load_snapshot().experiment_queue or ()
        ] == ["chat-message:B"]
    finally:
        await restored.close()


async def test_unacknowledged_platform_callback_is_retried_before_new_requirement(
    tmp_path,
):
    class Adapter(FakeAdapter):
        available = False

        async def notify_external_task(
            self, handle, notification, context
        ) -> JsonObject:
            await super().notify_external_task(handle, notification, context)
            return {"accepted": self.available}

    adapter = Adapter()
    runtime = ConversationRuntime(tmp_path, adapter)
    cid = "callback-order"
    context = RequestContext(user_id="owner")
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid, user_id="owner", workspace_root=str(tmp_path / cid)
            ),
            context,
        )
        store = runtime._store(cid)
        task = ExternalTaskState(
            task_id="platform-result",
            kind="workflow_run",
            status="succeeded",
            submitted_at="2026-11-23T00:00:00Z",
            updated_at="2026-11-23T00:00:01Z",
            resume_pending=True,
            resume_auto_run=True,
        )
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="external_task.submitted",
                payload=task.model_copy(
                    update={"status": "pending", "resume_pending": False}
                ).model_dump(mode="json"),
            )
        )
        store.append(
            ProductEvent(
                conversation_id=cid,
                type="external_task.completed",
                payload=task.model_dump(mode="json"),
            )
        )
        runtime._queue_task(store, cid, "newer", "requirement", "Run later")
        active = runtime._active[cid]
        await runtime._resume_notifications(active, store)
        assert [n.task_id for _, n, _ in adapter.external_task_notifications] == [
            "platform-result"
        ]
        adapter.available = True
        await runtime._resume_notifications(active, store)
        assert [n.task_id for _, n, _ in adapter.external_task_notifications] == [
            "platform-result",
            "platform-result",
            "newer",
        ]
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "placeholder",
    (
        "本轮执行已结束。可查看原任务中的执行详情。",
        "本轮执行已结束，执行记录中没有可展示的结果正文。",
        "This round ended without a recorded user-facing result.",
        "此前经过裁剪的报告正文",
    ),
)
async def test_ordinary_execution_completion_delivers_one_readable_result(
    preparation_setup, tmp_path, placeholder, monkeypatch
):
    preparations, owner, policy = preparation_setup
    cid = owner.conversation_id
    context = RequestContext(user_id=owner.user_id, profile_id="echomind")
    outcomes = []

    class Adapter(FakeAdapter):
        async def compose_execution_notice(self, handle, facts):
            outcomes.append(facts)
            return '{"message":null}'

    adapter = Adapter()
    runtime = ConversationRuntime(tmp_path, adapter, preparation_store=preparations)
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=cid,
                user_id=context.user_id,
                profile_id=context.profile_id,
                workspace_root=str(tmp_path / cid),
                preparation_policy=policy,
                delegated_roles=(DelegatedRole(name="eval"),),
            ),
            context,
        )
        chat = await runtime.echomind_chat.open(cid, {}, context)
        manager = runtime._store(cid)
        chat_store = runtime._store(chat.conversation_id)
        task = "requirement:chat:live"
        run = "analysis"
        runtime._queue_task(
            manager, cid, task, "requirement", "Inspect", origin_turn_id="user-turn"
        )
        evidence = "  ## 交付物\n[HTML](https://example.org/report?signature=abc)\n\n```json\n{}\n```\n"
        blocks: list = [
            {"type": "text", "text": evidence},
            {"type": "image_url", "url": "https://example.org/figure.png"},
        ]

        def publish(kind: ProductEventType, payload: JsonObject):
            event, _ = manager.append(
                ProductEvent(
                    conversation_id=cid,
                    type=kind,
                    run_id=run,
                    task_id=task,
                    payload=payload,
                )
            )
            runtime._publish(event)
            runtime._publish(event)

        publish("status.changed", {"status": "running"})
        source = "echomind-update:analysis:milestone"
        publish("message.delta", {"message_id": "draft", "text": "internal only"})
        for fragment in ("  ## ", "交付物"):
            publish(
                "message.delta",
                {"message_id": source, "text": fragment, "public_delta": fragment},
            )
            await wait_for(lambda: not runtime.echomind_notices.workers)
        [preview] = [
            i
            for i in chat_store.load_snapshot().timeline
            if isinstance(i, TimelineMessage)
        ]
        assert preview.status == "streaming" and preview.content == (
            TextContent(text="  ## 交付物"),
        )
        assert (
            preview.execution_notice
            and preview.execution_notice.origin_turn_id == "user-turn"
        )
        # Reconnect after writing a preview but before saving the worker cursor.
        chat_store.save_execution_notice_cursor(1)
        runtime.echomind_notices.schedule(cid)
        await wait_for(lambda: not runtime.echomind_notices.workers)
        assert [
            e.payload["text"] for e in chat_store.replay() if e.type == "message.delta"
        ] == ["  ## ", "交付物"]
        publish(
            "message.completed",
            {"role": "assistant", "message_id": "empty", "content": []},
        )
        publish(
            "message.completed",
            {
                "role": "assistant",
                "message_id": "echomind-update:analysis:milestone",
                "content": blocks,
            },
        )
        await wait_for(lambda: not runtime.echomind_notices.workers)
        [first] = [
            i
            for i in chat_store.load_snapshot().timeline
            if isinstance(i, TimelineMessage)
        ]
        assert manager.load_snapshot().status == "running"
        assert [block.model_dump() for block in first.content] == blocks
        assert (
            first.execution_notice
            and first.execution_notice.source_message_id
            == "echomind-update:analysis:milestone"
        )
        publish(
            "message.delta",
            {
                "message_id": "final",
                "text": "[[echomind:reply]]最终",
                "public_delta": "最终",
            },
        )
        publish(
            "message.delta",
            {
                "message_id": "echomind-update:analysis:aborted",
                "text": "incomplete",
                "public_delta": "incomplete",
            },
        )
        await wait_for(lambda: not runtime.echomind_notices.workers)
        publish(
            "message.completed",
            {
                "role": "assistant",
                "message_id": "final",
                "content": [{"type": "text", "text": "[[echomind:reply]]最终结论"}],
            },
        )
        publish("status.changed", {"status": "idle"})
        await wait_for(lambda: not runtime.echomind_notices.workers)
        messages = {
            i.execution_notice.source_message_id: i
            for i in chat_store.load_snapshot().timeline
            if isinstance(i, TimelineMessage) and i.execution_notice
        }
        assert messages[source].item_id == preview.item_id
        assert messages["final"].content == (TextContent(text="最终结论"),)
        assert messages["echomind-update:analysis:aborted"].status == "completed"
        assert not messages["echomind-update:analysis:aborted"].content
        run = "historical-run"
        origin = "historical-turn"
        task = "requirement:chat:historical"
        runtime._queue_task(
            manager, cid, task, "requirement", "Inspect", origin_turn_id=origin
        )
        manager.append(
            ProductEvent(
                conversation_id=cid,
                task_id=task,
                run_id=run,
                type="message.completed",
                payload={
                    "role": "assistant",
                    "message_id": run,
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "## 目录概览\n样本使用 `<think>` 标签。\n"
                                "图片共有 15 张。"
                            ),
                        }
                    ],
                },
            )
        )
        ended, _ = manager.append(
            ProductEvent(
                conversation_id=cid,
                task_id=task,
                run_id=run,
                type="status.changed",
                payload={"status": "idle"},
            )
        )
        identity = "execution-notice:" + sha256(f"round:{run}".encode()).hexdigest()
        chat_store.append(
            ProductEvent(
                conversation_id=chat.conversation_id,
                event_id=identity,
                run_id=run,
                type="message.completed",
                payload={
                    "role": "assistant",
                    "message_id": identity,
                    "content": [
                        {
                            "type": "text",
                            "text": placeholder,
                        }
                    ],
                    "execution_notice": {"origin_turn_id": origin},
                },
            )
        )
        chat_store.save_execution_notice_cursor(ended.seq)

        def unexpected_plan_lookup(snapshot):
            raise AssertionError("An old round must not consult the current plan")

        monkeypatch.setattr(
            "pyromind_runtime.application.echomind_notices.plan_ready",
            unexpected_plan_lookup,
        )
        for _ in range(2):
            await runtime.echomind_chat.snapshot(cid, context)
            await wait_for(lambda: not runtime.echomind_notices.workers)
        visible = await runtime.echomind_chat.snapshot(cid, context)
        assert visible
        messages = [
            item for item in visible.timeline if isinstance(item, TimelineMessage)
        ]
        assert len(messages) == 3
        assert messages[-1].item_id == identity
        assert messages[-1].content == (
            TextContent(
                text="## 目录概览\n样本使用 `<think>` 标签。\n图片共有 15 张。"
            ),
        )
        assert (
            messages[-1].execution_notice
            and messages[-1].execution_notice.origin_turn_id == origin
        )
        assert all(
            item.execution_notice
            and item.execution_notice.origin_turn_id == "user-turn"
            for item in messages[:-1]
        )
        assert messages[1].content == (TextContent(text="最终结论"),)
        assert not outcomes
    finally:
        await runtime.close()


def test_chat_model_pairs_late_results_with_original_questions():
    rows = []

    def message(role, turn, text, *, run=None, notice=None, identity=None):
        seq = len(rows) + 1
        rows.append(
            dict(
                kind="message",
                item_id=identity or str(seq),
                turn_id=turn,
                run_id=run or turn,
                role=role,
                content=[{"type": "text", "text": text}],
                started_seq=seq,
                completed_seq=seq,
                status="completed",
                execution_notice=notice,
            )
        )

    message("user", "a", "Inspect A")
    rows.append(
        dict(
            kind="operation",
            item_id="queue",
            turn_id="a",
            started_seq=2,
            completed_seq=2,
            name="queue_echomind_requirement",
            category="tool",
            status="completed",
        )
    )
    message("assistant", "a", "Queue receipt")
    message("user", "b", "Explain B")
    message("assistant", "b", "Answer B")
    message(
        "assistant",
        None,
        "Old A",
        run="old",
        notice={"origin_turn_id": "a"},
        identity="execution-notice:" + sha256(b"round:old").hexdigest(),
    )
    message(
        "assistant",
        None,
        "A result",
        notice={"origin_turn_id": "a", "has_result": True},
    )
    message(
        "assistant",
        None,
        "Status only",
        notice={"origin_turn_id": "a", "has_result": False},
    )
    message("assistant", None, "Orphan", notice={"has_result": True})
    message("user", "c", "Current question")
    message("assistant", "c", "Current intermediate step")
    snapshot = ConversationSnapshot.model_validate(
        {"conversation_id": "chat", "capabilities": {}, "timeline": rows}
    )
    assert completed_chat_turns(snapshot, "c") == [
        {"question": "Inspect A", "answer": "Old A\n\nA result"},
        {"question": "Explain B", "answer": "Answer B"},
    ]
