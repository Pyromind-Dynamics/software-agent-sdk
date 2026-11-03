"""Regressions for automatic-execution suspension coverage (A/B/C).

A: graph resume and queued graph execution must not bypass the persisted
   suspension policy, no matter when the cancellation lands.
B: an accepted user message actively delivers saved-up results without an
   extra snapshot read.
C: the conversation list coordinates leftover inputs exactly like the detail
   entry, after authorization.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import (
    InputResponseCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter


QUESTIONS = [
    {
        "id": "duplicate_policy",
        "prompt": "如何处理重复行？",
        "options": [
            {"id": "keep_first", "label": "保留第一条"},
            {"id": "drop_all", "label": "全部删除"},
        ],
    },
]


class RecordingBackend:
    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict]] = []
        self.executions: list[str] = []

    async def invoke(
        self, scope: str, action: str, arguments: JsonObject, request_id: str
    ) -> JsonObject:
        self.invocations.append((action, dict(arguments)))
        if action == "resume":
            return {"id": arguments["run_id"], "status": "queued"}
        return {"status": "ok"}

    async def execute(self, scope: str, run_id: str, emit: object) -> JsonObject:
        self.executions.append(run_id)
        return {"id": run_id, "status": "succeeded", "result": {"outputs": {}}}

    async def close(self) -> None:
        return None


def _input_adapter() -> FakeAdapter:
    adapter = FakeAdapter()
    adapter.capabilities = HarnessCapabilities(
        resume=True,
        cancel=True,
        permission_reply=True,
        partial_message=True,
        fork=True,
        workflow_rollback=True,
        external_task_resume=True,
        user_input_reply=True,
    )
    return adapter


async def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


async def _graph_runtime(
    tmp_path,
) -> tuple[
    ConversationRuntime, FakeAdapter, RecordingBackend, RequestContext, FileProductStore
]:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = _input_adapter()
    backend = RecordingBackend()
    runtime = ConversationRuntime(conversations, adapter, workflows=backend)
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="conversation-1",
            user_id="42",
            workspace_root=str(conversations / "conversation-1"),
            user_input_enabled=True,
        ),
        context,
    )
    adapter.emit(
        "conversation-1",
        "input.requested",
        {
            "request_id": "input-1",
            "operation_id": "call-1",
            "questions": QUESTIONS,
        },
        run_id="run-1",
    )
    store = runtime._store("conversation-1")
    await _wait_for(lambda: store.load_snapshot().pending_inputs)
    return runtime, adapter, backend, context, store


def _graph_task_payload(task_id: str = "task-1") -> dict:
    return {
        "task_id": task_id,
        "kind": "graph_await",
        "run_id": "run-1",
        "graph_run_id": "run-g",
        "status": "running",
        "submitted_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


async def _cancel_pending(runtime, context) -> None:
    await runtime.submit_command(
        "conversation-1",
        InputResponseCommand.model_validate(
            {
                "command_id": f"cancel-{uuid4_hex()}",
                "request_id": "input-1",
                "status": "cancelled",
            }
        ),
        context,
    )


def uuid4_hex() -> str:
    from uuid import uuid4

    return uuid4().hex


async def _continue_message(runtime, context, command_id: str = "msg-1") -> None:
    await runtime.submit_command(
        "conversation-1",
        UserMessageCommand(command_id=command_id, content=(TextContent(text="继续"),)),
        context,
    )


# -- A: graph must not bypass the persisted suspension policy -----------------


async def test_graph_resume_blocked_when_cancel_lands_during_callback(
    tmp_path,
) -> None:
    """A1: the callback read the policy before the await; the decision must not."""
    runtime, adapter, backend, context, store = await _graph_runtime(tmp_path)
    try:
        runtime.register_external_task("conversation-1", _graph_task_payload())

        original = runtime.reconcile_versions

        async def cancel_mid_callback(conversation_id, context=None):
            # The request is cancelled during callback version reconciliation; the
            # suspension policy is persisted during this interval.
            await runtime._user_input.interrupt_pending(
                store, conversation_id, "input_cancelled"
            )
            return await original(conversation_id, context)

        runtime.reconcile_versions = cancel_mid_callback
        await runtime.deliver_external_task_status(
            "conversation-1", task_id="task-1", status="succeeded"
        )

        assert not any(action == "resume" for action, _ in backend.invocations), (
            "graph must not resume while the persisted policy is suspended"
        )
        assert adapter.external_task_notifications == []
        task = next(
            t for t in store.load_snapshot().external_tasks if t.task_id == "task-1"
        )
        assert task.resume_pending is True
    finally:
        await runtime.close()


async def test_queued_graph_waits_for_suspension_before_executing(tmp_path) -> None:
    """A2: a queued graph executes only after the user lifts the suspension."""
    runtime, adapter, backend, context, store = await _graph_runtime(tmp_path)
    try:
        runtime.register_external_task("conversation-1", _graph_task_payload())
        await runtime.deliver_external_task_status(
            "conversation-1", task_id="task-1", status="succeeded"
        )
        await _wait_for(
            lambda: any(action == "resume" for action, _ in backend.invocations)
        )
        # The old turn ends here (the pump releases it); the user cancelled the
        # question before it ended.
        await _cancel_pending(runtime, context)
        assert runtime._workflows is not None
        runtime._workflows.release("conversation-1", None)
        await asyncio.sleep(0.05)
        assert backend.executions == [], "queued graph must not execute while suspended"

        # The queued graph runs only after the user explicitly resumes.
        await _continue_message(runtime, context)
        await _wait_for(lambda: backend.executions)
        assert backend.executions == ["run-g"]
    finally:
        await runtime.close()


async def test_paused_admission_consumes_stale_wakeup(tmp_path, monkeypatch) -> None:
    runtime, _adapter, _backend, context, _store = await _graph_runtime(tmp_path)
    admission = None
    try:
        await _cancel_pending(runtime, context)
        wakeup = runtime._auto_execution_wakeups.setdefault(
            "conversation-1", asyncio.Event()
        )
        wakeup.set()
        wait = wakeup.wait
        calls = 0

        async def guarded_wait():
            nonlocal calls
            calls += 1
            assert calls <= 3, "paused admission is spinning on a stale wakeup"
            return await wait()

        monkeypatch.setattr(wakeup, "wait", guarded_wait)
        admission = asyncio.create_task(runtime._await_auto_execution("conversation-1"))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(admission), 0.03)
        await _continue_message(runtime, context)
        await asyncio.wait_for(admission, 1)
    finally:
        if admission is not None:
            admission.cancel()
            await asyncio.gather(admission, return_exceptions=True)
        await runtime.close()


async def test_cancel_during_start_event_blocks_graph_execution(tmp_path) -> None:
    runtime, _adapter, backend, context, _store = await _graph_runtime(tmp_path)
    assert runtime._workflows is not None
    workflows = runtime._workflows
    workflows.complete = None
    publish = workflows.publish
    cancelled = asyncio.Event()

    async def cancel_during_publish(*args):
        await publish(*args)
        if args[2] == "operation.started":
            await _cancel_pending(runtime, context)
            cancelled.set()

    workflows.publish = cancel_during_publish
    try:
        runtime.register_external_task("conversation-1", _graph_task_payload())
        await runtime.deliver_external_task_status(
            "conversation-1", task_id="task-1", status="succeeded"
        )
        await _wait_for(lambda: workflows.busy("conversation-1"))
        workflows.release("conversation-1")
        await asyncio.wait_for(cancelled.wait(), 1)
        assert backend.executions == []
        await _continue_message(runtime, context)
        await _wait_for(lambda: backend.executions)
        assert backend.executions == ["run-g"]
    finally:
        await runtime.close()


async def test_blocked_graph_backlog_is_retried_after_continue(tmp_path) -> None:
    runtime, adapter, backend, context, store = await _graph_runtime(tmp_path)
    assert runtime._workflows is not None
    runtime._workflows.complete = None
    runtime.register_external_task(
        "conversation-1",
        {
            **_graph_task_payload("plain"),
            "kind": "workflow_run",
            "graph_run_id": None,
            "status": "succeeded",
            "resume_pending": True,
        },
    )
    runtime.register_external_task(
        "conversation-1",
        {**_graph_task_payload("graph"), "status": "succeeded", "resume_pending": True},
    )
    notify = adapter.notify_external_task

    async def cancel_during_notification(handle, notification, context):
        result = await notify(handle, notification, context)
        await runtime._user_input.interrupt_pending(
            store, "conversation-1", "cancel_during_notification"
        )
        return result

    adapter.notify_external_task = cancel_during_notification
    try:
        await runtime._resume_pending_external_tasks(
            runtime._active["conversation-1"], store
        )
        task = next(
            t for t in store.load_snapshot().external_tasks if t.task_id == "graph"
        )
        assert task.resume_pending
        assert not any(action == "resume" for action, _ in backend.invocations)
        adapter.notify_external_task = notify
        await _continue_message(runtime, context)
        assert sum(action == "resume" for action, _ in backend.invocations) == 1
        task = next(
            t for t in store.load_snapshot().external_tasks if t.task_id == "graph"
        )
        assert not task.resume_pending
    finally:
        await runtime.close()


async def test_parked_await_rebinds_a_terminal_task_and_resumes_once(tmp_path) -> None:
    runtime, _adapter, backend, _context, store = await _graph_runtime(tmp_path)
    runtime.register_external_task(
        "conversation-1",
        {
            "task_id": "platform-1",
            "kind": "workflow_run",
            "run_id": "submission-1",
            "status": "succeeded",
            "submitted_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:01:00Z",
        },
    )
    runtime.register_external_task(
        "conversation-1",
        {
            "task_id": "graph-run",
            "kind": "historical_experience",
            "run_id": "agent-run-1",
            "status": "waiting",
            "submitted_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:01:00Z",
            "completion_result": {
                "id": "graph-run",
                "status": "waiting",
                "waiting": {
                    "kind": "external",
                    "node": "chain.wait",
                    "task_id": "platform-1",
                    "output_dir": "/.pyromind-agent/conversation-1/data/run-1",
                },
            },
        },
    )
    try:
        await runtime._repair_parked_await_bindings("conversation-1")

        platform = next(
            task
            for task in store.load_snapshot().external_tasks
            if task.task_id == "platform-1"
        )
        assert platform.kind == "graph_await"
        assert platform.graph_run_id == "graph-run"
        assert platform.status == "succeeded"
        assert backend.invocations == [("resume", {"run_id": "graph-run"})]
    finally:
        await runtime.close()


async def test_callback_keeps_failed_graph_resume_pending_until_retry(tmp_path) -> None:
    runtime, _adapter, backend, context, store = await _graph_runtime(tmp_path)
    assert runtime._workflows is not None
    runtime._workflows.complete = None
    invoke = backend.invoke
    attempted = asyncio.Event()

    async def fail_first_resume(scope, action, arguments, request_id):
        if action == "resume" and not attempted.is_set():
            attempted.set()
            raise RuntimeError("temporary graph service failure")
        return await invoke(scope, action, arguments, request_id)

    backend.invoke = fail_first_resume
    try:
        runtime.register_external_task("conversation-1", _graph_task_payload())
        await runtime.deliver_external_task_status(
            "conversation-1", task_id="task-1", status="succeeded"
        )
        await asyncio.wait_for(attempted.wait(), 1)
        assert store.load_snapshot().external_tasks[0].resume_pending
        assert backend.invocations == []

        await runtime._user_input.interrupt_pending(
            store, "conversation-1", "input_cancelled"
        )
        await _continue_message(runtime, context)
        assert sum(action == "resume" for action, _ in backend.invocations) == 1
        assert not store.load_snapshot().external_tasks[0].resume_pending
    finally:
        await runtime.close()


async def test_no_active_session_persisted_suspension_blocks_resume(
    tmp_path,
) -> None:
    """A3: the policy is persisted; a live adapter session is not required."""
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    (conversations / "conversation-1").mkdir()
    store = FileProductStore(conversations / "conversation-1")
    store.create(
        ConversationSnapshot(
            conversation_id="conversation-1",
            capabilities=HarnessCapabilities(user_input_reply=True),
        ),
        user_id="42",
    )
    store.append(
        ProductEvent.model_validate(
            {
                "conversation_id": "conversation-1",
                "type": "execution.policy.updated",
                "payload": {"automatic_turns_suspended": True},
            }
        )
    )
    backend = RecordingBackend()
    runtime = ConversationRuntime(conversations, FakeAdapter(), workflows=backend)
    try:
        runtime.register_external_task("conversation-1", _graph_task_payload())
        await runtime.deliver_external_task_status(
            "conversation-1", task_id="task-1", status="succeeded"
        )
        assert not any(action == "resume" for action, _ in backend.invocations), (
            "persisted suspension must block the graph resume without a session"
        )
        task = next(
            t for t in store.load_snapshot().external_tasks if t.task_id == "task-1"
        )
        assert task.resume_pending is True
    finally:
        await runtime.close()


async def test_suspended_graph_resume_keeps_pending_until_continue(tmp_path) -> None:
    """A2b: a blocked resume stays deliverable; the user's continue resumes it."""
    runtime, adapter, backend, context, store = await _graph_runtime(tmp_path)
    try:
        runtime.register_external_task("conversation-1", _graph_task_payload())
        await _cancel_pending(runtime, context)
        # A callback during suspension does not resume the graph; keep it pending.
        await runtime.deliver_external_task_status(
            "conversation-1", task_id="task-1", status="succeeded"
        )
        assert not any(action == "resume" for action, _ in backend.invocations)
        # Resume the graph and mark it consumed after the user explicitly continues.
        await _continue_message(runtime, context)
        await _wait_for(
            lambda: any(action == "resume" for action, _ in backend.invocations)
        )
        task = next(
            t for t in store.load_snapshot().external_tasks if t.task_id == "task-1"
        )
        assert task.resume_pending is False
    finally:
        await runtime.close()


# -- B: accepted message delivers the backlog without a snapshot read --------


async def test_continue_message_delivers_backlog_without_snapshot_read(
    tmp_path,
) -> None:
    runtime, adapter, _backend, context, store = await _graph_runtime(tmp_path)
    try:
        runtime.register_external_task(
            "conversation-1",
            {
                "task_id": "task-1",
                "kind": "workflow_run",
                "status": "pending",
                "submitted_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )
        await _cancel_pending(runtime, context)
        await runtime.deliver_external_task_status(
            "conversation-1", task_id="task-1", status="succeeded"
        )
        assert adapter.external_task_notifications == []
        # The accepted message itself must deliver the backlog without another read.
        await _continue_message(runtime, context)
        await _wait_for(lambda: adapter.external_task_notifications)
        assert adapter.external_task_notifications[0][1].task_id == "task-1"
        task = next(
            t for t in store.load_snapshot().external_tasks if t.task_id == "task-1"
        )
        assert task.resume_pending is False
    finally:
        await runtime.close()


async def test_rejected_message_does_not_deliver_backlog(tmp_path) -> None:
    runtime, adapter, _backend, context, _store = await _graph_runtime(tmp_path)
    try:
        runtime.register_external_task(
            "conversation-1",
            {
                "task_id": "task-1",
                "kind": "workflow_run",
                "status": "pending",
                "submitted_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )
        # A message rejected during a pending question must not deliver any backlog.
        with pytest.raises(ProductRuntimeError) as excinfo:
            await _continue_message(runtime, context)
        assert excinfo.value.code == "input_pending"
        assert adapter.external_task_notifications == []
    finally:
        await runtime.close()


async def test_new_suspension_blocks_backlog_delivery(tmp_path) -> None:
    runtime, adapter, _backend, context, store = await _graph_runtime(tmp_path)
    try:
        runtime.register_external_task(
            "conversation-1",
            {
                "task_id": "task-1",
                "kind": "workflow_run",
                "status": "pending",
                "submitted_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )
        await _cancel_pending(runtime, context)
        await runtime.deliver_external_task_status(
            "conversation-1", task_id="task-1", status="succeeded"
        )
        # The accepted message releases suspension, but delivery creates a new one.
        original = runtime.reconcile_versions

        async def suspend_mid_delivery(conversation_id, context=None):
            await runtime._user_input.suspend(
                store, conversation_id, "test_new_suspension"
            )
            return await original(conversation_id, context)

        runtime.reconcile_versions = suspend_mid_delivery
        await _continue_message(runtime, context, command_id="msg-lift")
        await asyncio.sleep(0.05)
        assert adapter.external_task_notifications == []
        task = next(
            t for t in store.load_snapshot().external_tasks if t.task_id == "task-1"
        )
        assert task.resume_pending is True
        assert store.load_snapshot().automatic_turns_suspended is True
    finally:
        await runtime.close()


async def test_concurrent_reads_do_not_double_deliver_backlog(tmp_path) -> None:
    runtime, adapter, _backend, context, store = await _graph_runtime(tmp_path)
    try:
        runtime.register_external_task(
            "conversation-1",
            {
                "task_id": "task-1",
                "kind": "workflow_run",
                "status": "pending",
                "submitted_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )
        await _cancel_pending(runtime, context)
        await runtime.deliver_external_task_status(
            "conversation-1", task_id="task-1", status="succeeded"
        )
        # Concurrent snapshot reads during delivery must not duplicate the notification.
        await asyncio.gather(
            _continue_message(runtime, context),
            runtime.get_snapshot("conversation-1", context),
        )
        await _wait_for(lambda: adapter.external_task_notifications)
        await asyncio.sleep(0.05)
        delivered = [
            notification
            for _sid, notification, _ctx in adapter.external_task_notifications
            if notification.task_id == "task-1"
        ]
        assert len(delivered) == 1
    finally:
        await runtime.close()


# -- C: the list entry coordinates leftovers like the detail entry -----------


async def _pending_conversation_state(tmp_path):
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = _input_adapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="conversation-1",
            user_id="42",
            workspace_root=str(conversations / "conversation-1"),
            user_input_enabled=True,
        ),
        context,
    )
    adapter.emit(
        "conversation-1",
        "input.requested",
        {
            "request_id": "input-1",
            "operation_id": "call-1",
            "questions": QUESTIONS,
        },
        run_id="run-1",
    )
    store = runtime._store("conversation-1")
    await _wait_for(lambda: store.load_snapshot().pending_inputs)
    await runtime.close()
    return conversations, context


async def test_restarted_list_entry_coordinates_leftover_inputs(tmp_path) -> None:
    conversations, context = await _pending_conversation_state(tmp_path)
    runtime = ConversationRuntime(conversations, _input_adapter())
    try:
        # The first list access after restart must not report stale pending questions.
        snapshots = await runtime.list_snapshots(context)
        entry = snapshots[0]
        assert entry.conversation_id == "conversation-1"
        assert entry.pending_inputs == ()
        assert entry.automatic_turns_suspended is True
        # Repeated list requests must not append duplicate interruptions.
        again = await runtime.list_snapshots(context)
        assert [item.conversation_id for item in again] == ["conversation-1"]
        store = runtime._store("conversation-1")
        interrupted = [
            event
            for event in store.replay()
            if event.type == "input.resolved"
            and event.payload.get("request_id") == "input-1"
        ]
        assert len(interrupted) == 1
    finally:
        await runtime.close()


async def test_list_keeps_live_pending_input_of_active_session(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = _input_adapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="conversation-1",
                user_id="42",
                workspace_root=str(conversations / "conversation-1"),
                user_input_enabled=True,
            ),
            context,
        )
        adapter.emit(
            "conversation-1",
            "input.requested",
            {
                "request_id": "input-1",
                "operation_id": "call-1",
                "questions": QUESTIONS,
            },
            run_id="run-1",
        )
        store = runtime._store("conversation-1")
        await _wait_for(lambda: store.load_snapshot().pending_inputs)
        snapshots = await runtime.list_snapshots(context)
        entry = snapshots[0]
        # Preserve a live session's pending question when listing conversations.
        assert entry.pending_inputs[0].request_id == "input-1"
        assert entry.status == "waiting_for_input"
    finally:
        await runtime.close()


async def test_list_of_foreign_user_reads_and_writes_nothing(tmp_path) -> None:
    conversations, _owner = await _pending_conversation_state(tmp_path)
    runtime = ConversationRuntime(conversations, _input_adapter())
    try:
        foreign = RequestContext(user_id="43")
        snapshots = await runtime.list_snapshots(foreign)
        assert snapshots == ()
        store = runtime._store("conversation-1")
        snapshot = store.load_snapshot()
        assert snapshot.pending_inputs[0].request_id == "input-1"
        interrupted = [
            event
            for event in store.replay()
            if event.type == "input.resolved"
            and event.payload.get("request_id") == "input-1"
        ]
        assert interrupted == []
    finally:
        await runtime.close()


async def test_list_leaves_unenabled_conversation_untouched(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="plain-1",
                user_id="42",
                workspace_root=str(conversations / "plain-1"),
            ),
            context,
        )
        before = runtime._store("plain-1").load_snapshot()
        snapshots = await runtime.list_snapshots(context)
        assert snapshots[0].conversation_id == "plain-1"
        assert snapshots[0].status == before.status
        assert snapshots[0].pending_inputs == ()
        assert snapshots[0].automatic_turns_suspended is False
    finally:
        await runtime.close()
