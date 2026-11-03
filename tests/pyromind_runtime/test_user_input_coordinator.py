from __future__ import annotations

import asyncio
import threading
import time

import pytest
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import (
    CancelCommand,
    InputResponseCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import (
    ProductRuntimeError,
    UserInputValidationError,
)
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.user_input import UserInputResolution
from pyromind_runtime.infrastructure.file_product_store import (
    CommandConflictError,
    FileProductStore,
)
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
    {"id": "output_name", "prompt": "输出文件名？", "options": []},
]


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


async def _create(
    runtime: ConversationRuntime,
    conversations,
    context: RequestContext,
    conversation_id: str = "conversation-1",
) -> ConversationSnapshot:
    return await runtime.create_conversation(
        SessionSpec(
            conversation_id=conversation_id,
            user_id=context.user_id,
            workspace_root=str(conversations),
        ),
        context,
    )


def _request_input(adapter: FakeAdapter, conversation_id: str) -> None:
    adapter.emit(
        conversation_id,
        "input.requested",
        {
            "request_id": "input-1",
            "operation_id": "call-input-1",
            "questions": QUESTIONS,
        },
        run_id="run-1",
    )
    adapter.emit(
        conversation_id, "status.changed", {"status": "running"}, run_id="run-1"
    )


async def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def _answer_command(
    command_id: str = "answer-1",
    request_id: str = "input-1",
    status: str = "answered",
    answers: list | None = None,
) -> InputResponseCommand:
    if answers is None:
        answers = (
            []
            if status == "cancelled"
            else [
                {"question_id": "duplicate_policy", "option_id": "keep_first"},
                {"question_id": "output_name", "text": "cleaned_data.csv"},
            ]
        )
    return InputResponseCommand.model_validate(
        {
            "command_id": command_id,
            "request_id": request_id,
            "status": status,
            **({"answers": answers} if answers else {}),
        }
    )


async def _runtime_with_pending_request(tmp_path):
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = _input_adapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")
    await _create(runtime, conversations, context)
    _request_input(adapter, "conversation-1")
    store = runtime._store("conversation-1")
    await _wait_for(lambda: store.load_snapshot().pending_inputs)
    return runtime, adapter, context, store


async def test_answer_flow_resolves_request_and_returns_run(tmp_path) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    try:
        receipt = await runtime.submit_command(
            "conversation-1", _answer_command(), context
        )
        assert receipt.status == "completed"
        assert receipt.response["status"] == "answered"
        assert [(sid, command.status) for sid, command in adapter.input_responses] == [
            ("conversation-1", "answered")
        ]
        snapshot = store.load_snapshot()
        assert snapshot.pending_inputs == ()
        assert snapshot.status == "running"
        assert snapshot.active_run_id == "run-1"
    finally:
        await runtime.close()


async def test_cancel_flow_pauses_and_suspends(tmp_path) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    try:
        receipt = await runtime.submit_command(
            "conversation-1",
            _answer_command(command_id="cancel-1", status="cancelled"),
            context,
        )
        assert receipt.status == "completed"
        snapshot = store.load_snapshot()
        assert snapshot.pending_inputs == ()
        assert snapshot.status == "paused"
        assert snapshot.automatic_turns_suspended is True
    finally:
        await runtime.close()


async def test_answer_retry_is_idempotent(tmp_path) -> None:
    runtime, adapter, context, _ = await _runtime_with_pending_request(tmp_path)
    try:
        first = await runtime.submit_command(
            "conversation-1", _answer_command(), context
        )
        second = await runtime.submit_command(
            "conversation-1", _answer_command(), context
        )
        assert second == first
        assert len(adapter.input_responses) == 1
    finally:
        await runtime.close()


async def test_same_command_id_with_different_content_conflicts(tmp_path) -> None:
    runtime, _, context, _ = await _runtime_with_pending_request(tmp_path)
    try:
        await runtime.submit_command("conversation-1", _answer_command(), context)
        with pytest.raises(CommandConflictError):
            await runtime.submit_command(
                "conversation-1",
                _answer_command(
                    answers=[
                        {"question_id": "duplicate_policy", "option_id": "drop_all"},
                        {"question_id": "output_name", "text": "other.csv"},
                    ]
                ),
                context,
            )
    finally:
        await runtime.close()


async def test_second_command_for_same_request_loses(tmp_path) -> None:
    runtime, _, context, _ = await _runtime_with_pending_request(tmp_path)
    try:
        await runtime.submit_command("conversation-1", _answer_command(), context)
        with pytest.raises(ProductRuntimeError) as excinfo:
            await runtime.submit_command(
                "conversation-1", _answer_command(command_id="answer-2"), context
            )
        assert excinfo.value.code == "input_not_pending"
    finally:
        await runtime.close()


async def test_invalid_answers_keep_request_pending(tmp_path) -> None:
    runtime, _, context, store = await _runtime_with_pending_request(tmp_path)
    try:
        incomplete = _answer_command(
            command_id="try-1",
            answers=[{"question_id": "duplicate_policy", "option_id": "keep_first"}],
        )
        with pytest.raises(UserInputValidationError):
            await runtime.submit_command("conversation-1", incomplete, context)
        unknown_option = _answer_command(
            command_id="try-2",
            answers=[
                {"question_id": "duplicate_policy", "option_id": "nope"},
                {"question_id": "output_name", "text": "x.csv"},
            ],
        )
        with pytest.raises(UserInputValidationError):
            await runtime.submit_command("conversation-1", unknown_option, context)
        snapshot = store.load_snapshot()
        assert snapshot.pending_inputs[0].request_id == "input-1"
        # A corrected command can still be answered.
        receipt = await runtime.submit_command(
            "conversation-1", _answer_command(command_id="answer-fixed"), context
        )
        assert receipt.status == "completed"
        assert store.load_snapshot().pending_inputs == ()
    finally:
        await runtime.close()


async def test_user_message_rejected_while_pending(tmp_path) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    try:
        with pytest.raises(ProductRuntimeError) as excinfo:
            await runtime.submit_command(
                "conversation-1",
                UserMessageCommand(
                    command_id="msg-1", content=(TextContent(text="hi"),)
                ),
                context,
            )
        assert excinfo.value.code == "input_pending"
        # Reject before side effects: no adapter message or workflow sync occurs.
        assert all(
            not isinstance(command, UserMessageCommand)
            for _, command, _ in adapter.sent
        )
        assert store.load_snapshot().pending_inputs[0].request_id == "input-1"
    finally:
        await runtime.close()


async def test_suspension_lifted_only_by_accepted_message(tmp_path) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    try:
        await runtime.submit_command(
            "conversation-1",
            _answer_command(command_id="cancel-1", status="cancelled"),
            context,
        )
        assert store.load_snapshot().automatic_turns_suspended is True
        await runtime.submit_command(
            "conversation-1",
            UserMessageCommand(command_id="msg-1", content=(TextContent(text="继续"),)),
            context,
        )
        snapshot = store.load_snapshot()
        assert snapshot.automatic_turns_suspended is False
        assert any(
            isinstance(command, UserMessageCommand) for _, command, _ in adapter.sent
        )
    finally:
        await runtime.close()


async def test_delivery_failure_keeps_answer_and_suspends(tmp_path) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    adapter.fail_input_delivery = True
    try:
        receipt = await runtime.submit_command(
            "conversation-1", _answer_command(), context
        )
        # The receipt confirms the answer was accepted and stored; delivery failure
        # does not alter the stored answer.
        assert receipt.status == "completed"
        snapshot = store.load_snapshot()
        assert snapshot.pending_inputs == ()
        assert snapshot.automatic_turns_suspended is True
        resolved = [
            event
            for event in store.replay()
            if event.type == "input.resolved"
            and event.payload.get("request_id") == "input-1"
        ]
        assert len(resolved) == 1
        assert resolved[0].payload["outcome"] == "answered"
        notices = [
            item
            for item in snapshot.timeline
            if item.kind == "notice" and item.code == "input_delivery_failed"
        ]
        assert notices
    finally:
        await runtime.close()


async def test_recovery_interrupts_leftover_request(tmp_path) -> None:
    runtime, adapter, context, _ = await _runtime_with_pending_request(tmp_path)
    await runtime.close()

    conversations = tmp_path / "conversations"
    restarted = ConversationRuntime(conversations, _input_adapter())
    try:
        # Coordinate leftover inputs on the read path without starting the harness.
        snapshot = await restarted.get_snapshot("conversation-1", context)
        assert snapshot.pending_inputs == ()
        assert snapshot.status == "paused"
        assert snapshot.automatic_turns_suspended is True
        events = restarted._store("conversation-1").replay()
        interrupted = [
            event
            for event in events
            if event.type == "input.resolved"
            and event.payload.get("request_id") == "input-1"
        ]
        assert len(interrupted) == 1
        assert interrupted[0].payload["outcome"] == "interrupted"
    finally:
        await restarted.close()


async def test_recovery_reconciles_command_receipts(tmp_path) -> None:
    from pyromind_runtime.domain.events import ProductEvent

    conversations = tmp_path / "conversations" / "conversation-1"
    conversations.mkdir(parents=True)
    store = FileProductStore(conversations)
    store.create(
        ConversationSnapshot(
            conversation_id="conversation-1",
            capabilities=HarnessCapabilities(user_input_reply=True),
        ),
        user_id="42",
    )

    def _request_event(request_id: str, operation_id: str) -> ProductEvent:
        return ProductEvent.model_validate(
            {
                "conversation_id": "conversation-1",
                "type": "input.requested",
                "run_id": "run-1",
                "payload": {
                    "request_id": request_id,
                    "operation_id": operation_id,
                    "questions": QUESTIONS,
                },
            }
        )

    store.append(_request_event("input-1", "call-input-1"))
    # Claimed with no resolution: execution was interrupted.
    claimed, won = store.claim_command(_answer_command(command_id="stuck"))
    assert won and claimed.status == "accepted"
    store.append(_request_event("input-2", "call-input-2"))
    # The answered resolution is stored, but its completion receipt is missing;
    # rebuild it from events.
    claimed2, won2 = store.claim_command(
        _answer_command(command_id="delivered", request_id="input-2")
    )
    assert won2
    resolved, _ = store.resolve_pending_input(
        UserInputResolution(
            request_id="input-2",
            outcome="answered",
            answers=(),
            command_id="delivered",
        )
    )
    assert resolved is not None

    runtime = ConversationRuntime(tmp_path / "conversations", _input_adapter())
    context = RequestContext(user_id="42")
    try:
        await runtime.get_snapshot("conversation-1", context)
        receipts = store.load_input_command_receipts()
        assert receipts["stuck"][0].status == "failed"
        assert receipts["delivered"][0].status == "completed"
        assert receipts["delivered"][0].response["status"] == "answered"
        snapshot = store.load_snapshot()
        assert snapshot.automatic_turns_suspended is True
    finally:
        await runtime.close()


async def test_recovery_suspends_only_unconsumed_answers(tmp_path) -> None:
    from pyromind_runtime.domain.events import ProductEvent

    conversations = tmp_path / "conversations" / "conversation-1"
    conversations.mkdir(parents=True)
    store = FileProductStore(conversations)
    store.create(
        ConversationSnapshot(
            conversation_id="conversation-1",
            capabilities=HarnessCapabilities(user_input_reply=True),
        ),
        user_id="42",
    )
    for request_id, operation_id in [
        ("input-consumed", "call-consumed"),
        ("input-lost", "call-lost"),
    ]:
        command = _answer_command(command_id=f"cmd-{request_id}", request_id=request_id)
        claimed, won = store.claim_command(command)
        assert won
        store.append(
            ProductEvent.model_validate(
                {
                    "conversation_id": "conversation-1",
                    "type": "input.requested",
                    "run_id": "run-1",
                    "payload": {
                        "request_id": request_id,
                        "operation_id": operation_id,
                        "questions": QUESTIONS,
                    },
                }
            )
        )
        resolved, _ = store.resolve_pending_input(
            UserInputResolution(
                request_id=request_id,
                outcome="answered",
                answers=command.answers,
                command_id=command.command_id,
            )
        )
        assert resolved is not None
        store.complete_command(
            claimed.model_copy(
                update={"status": "completed", "response": {"status": "answered"}}
            )
        )
    # Consumed: the tool result exists. Unconsumed: execution stopped before the result.
    store.append(
        ProductEvent.model_validate(
            {
                "conversation_id": "conversation-1",
                "type": "operation.completed",
                "run_id": "run-1",
                "payload": {
                    "operation_id": "call-consumed",
                    "name": "ask_user",
                    "output": [{"type": "text", "text": "answered"}],
                },
            }
        )
    )

    runtime = ConversationRuntime(tmp_path / "conversations", _input_adapter())
    context = RequestContext(user_id="42")
    try:
        snapshot = await runtime.get_snapshot("conversation-1", context)
        assert snapshot.automatic_turns_suspended is True
    finally:
        await runtime.close()


async def test_recovery_skips_healthy_consumed_conversation(tmp_path) -> None:
    from pyromind_runtime.domain.events import ProductEvent

    conversations = tmp_path / "conversations" / "conversation-1"
    conversations.mkdir(parents=True)
    store = FileProductStore(conversations)
    store.create(
        ConversationSnapshot(
            conversation_id="conversation-1",
            capabilities=HarnessCapabilities(user_input_reply=True),
        ),
        user_id="42",
    )
    command = _answer_command()
    claimed, won = store.claim_command(command)
    assert won
    store.append(
        ProductEvent.model_validate(
            {
                "conversation_id": "conversation-1",
                "type": "input.requested",
                "run_id": "run-1",
                "payload": {
                    "request_id": "input-1",
                    "operation_id": "call-input-1",
                    "questions": QUESTIONS,
                },
            }
        )
    )
    store.resolve_pending_input(
        UserInputResolution(
            request_id="input-1",
            outcome="answered",
            answers=command.answers,
            command_id=command.command_id,
        )
    )
    store.complete_command(
        claimed.model_copy(
            update={"status": "completed", "response": {"status": "answered"}}
        )
    )
    store.append(
        ProductEvent.model_validate(
            {
                "conversation_id": "conversation-1",
                "type": "operation.completed",
                "run_id": "run-1",
                "payload": {
                    "operation_id": "call-input-1",
                    "name": "ask_user",
                    "output": [{"type": "text", "text": "answered"}],
                },
            }
        )
    )
    store.append(
        ProductEvent.model_validate(
            {
                "conversation_id": "conversation-1",
                "type": "status.changed",
                "run_id": "run-1",
                "payload": {"status": "idle"},
            }
        )
    )

    runtime = ConversationRuntime(tmp_path / "conversations", _input_adapter())
    context = RequestContext(user_id="42")
    try:
        snapshot = await runtime.get_snapshot("conversation-1", context)
        assert snapshot.automatic_turns_suspended is False
        assert snapshot.status == "idle"
    finally:
        await runtime.close()


async def test_external_task_callback_deferred_while_suspended(tmp_path) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
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
        await runtime.submit_command(
            "conversation-1",
            _answer_command(command_id="cancel-1", status="cancelled"),
            context,
        )
        await runtime.deliver_external_task_status(
            "conversation-1", task_id="task-1", status="succeeded"
        )
        # The result is stored, but the Agent has not been resumed.
        assert adapter.external_task_notifications == []
        snapshot = store.load_snapshot()
        task = next(t for t in snapshot.external_tasks if t.task_id == "task-1")
        assert task.status == "succeeded"
        assert task.resume_pending is True
        # After the user resumes, deliver the pending notification. The accepted
        # message itself triggers delivery without requiring another snapshot read.
        await runtime.submit_command(
            "conversation-1",
            UserMessageCommand(command_id="msg-1", content=(TextContent(text="继续"),)),
            context,
        )
        await _wait_for(lambda: adapter.external_task_notifications)
        notification = adapter.external_task_notifications[0][1]
        assert notification.task_id == "task-1"
    finally:
        await runtime.close()


async def test_fork_blocked_while_input_pending(tmp_path) -> None:
    runtime, _, context, _ = await _runtime_with_pending_request(tmp_path)
    try:
        with pytest.raises(ProductRuntimeError) as excinfo:
            await runtime.fork_conversation(
                "conversation-1", event_id="missing", title=None, context=context
            )
        assert excinfo.value.code == "conversation_busy"
    finally:
        await runtime.close()


async def test_create_rejects_unsupported_input_capability(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()  # user_input_reply 缺省为 False
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")
    try:
        with pytest.raises(ProductRuntimeError) as excinfo:
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id="conversation-1",
                    user_id="42",
                    workspace_root=str(conversations),
                    user_input_enabled=True,
                ),
                context,
            )
        assert excinfo.value.code == "capability_not_supported"
        assert adapter.closed == ["conversation-1"]
    finally:
        await runtime.close()


async def test_unauthorized_recovery_read_does_not_write(tmp_path) -> None:
    runtime, _, _, store = await _runtime_with_pending_request(tmp_path)
    await runtime.close()
    before = store.events_path.read_bytes()
    restarted = ConversationRuntime(tmp_path / "conversations", _input_adapter())
    try:
        with pytest.raises(PermissionError):
            await restarted.get_snapshot(
                "conversation-1", RequestContext(user_id="other")
            )
        assert store.events_path.read_bytes() == before
        assert store.load_snapshot().pending_inputs
    finally:
        await restarted.close()


async def test_repeated_cancel_after_resume_suspends_again(tmp_path) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    try:
        await runtime.submit_command(
            "conversation-1", _answer_command(status="cancelled"), context
        )
        await runtime.submit_command(
            "conversation-1",
            UserMessageCommand(
                command_id="resume", content=(TextContent(text="continue"),)
            ),
            context,
        )
        assert not store.load_snapshot().automatic_turns_suspended
        adapter.emit(
            "conversation-1",
            "input.requested",
            {
                "request_id": "input-2",
                "operation_id": "call-2",
                "questions": QUESTIONS,
            },
            event_id="input-2:requested",
            run_id="run-2",
        )
        await _wait_for(lambda: store.load_snapshot().pending_inputs)
        await runtime.submit_command(
            "conversation-1",
            _answer_command(
                command_id="cancel-2", request_id="input-2", status="cancelled"
            ),
            context,
        )
        assert (
            FileProductStore(store.conversation_dir)
            .load_snapshot()
            .automatic_turns_suspended
        )
        policies = [e for e in store.replay() if e.type == "execution.policy.updated"]
        assert [e.payload["automatic_turns_suspended"] for e in policies] == [
            True,
            False,
            True,
        ]
    finally:
        await runtime.close()


async def test_global_stop_resolves_input_and_allows_next_message(tmp_path) -> None:
    runtime, _, context, store = await _runtime_with_pending_request(tmp_path)
    try:
        await runtime.submit_command(
            "conversation-1", CancelCommand(command_id="stop"), context
        )
        assert not store.load_snapshot().pending_inputs
        assert store.load_snapshot().automatic_turns_suspended
        receipt = await runtime.submit_command(
            "conversation-1",
            UserMessageCommand(
                command_id="next", content=(TextContent(text="continue"),)
            ),
            context,
        )
        assert receipt.status == "completed"
    finally:
        await runtime.close()


async def test_execution_input_interruption_suspends_and_ignores_stale_run(
    tmp_path,
) -> None:
    runtime, adapter, _, store = await _runtime_with_pending_request(tmp_path)
    try:
        adapter.emit(
            "conversation-1",
            "input.resolved",
            {"request_id": "input-1", "outcome": "interrupted"},
            run_id="old-run",
        )
        adapter.emit(
            "conversation-1", "status.changed", {"status": "idle"}, run_id="old-run"
        )
        await asyncio.sleep(0.05)
        assert store.load_snapshot().pending_inputs
        adapter.emit(
            "conversation-1",
            "input.resolved",
            {"request_id": "input-1", "outcome": "interrupted"},
            run_id="run-1",
        )
        await _wait_for(lambda: not store.load_snapshot().pending_inputs)
        assert store.load_snapshot().automatic_turns_suspended
    finally:
        await runtime.close()


async def test_input_persistence_failure_closes_execution(
    tmp_path, monkeypatch
) -> None:
    runtime, adapter, _, store = await _runtime_with_pending_request(tmp_path)
    append = FileProductStore.append

    def fail_request(self, event):
        if event.type == "input.requested":
            raise OSError("injected persistence failure")
        return append(self, event)

    try:
        monkeypatch.setattr(FileProductStore, "append", fail_request)
        adapter.emit(
            "conversation-1",
            "input.requested",
            {
                "request_id": "invisible",
                "operation_id": "call-invisible",
                "questions": QUESTIONS,
            },
            run_id="run-1",
        )
        await _wait_for(lambda: "conversation-1" not in runtime._active)
        assert adapter.closed == ["conversation-1"]
        assert not store.load_snapshot().pending_inputs
        assert store.load_snapshot().automatic_turns_suspended
    finally:
        await runtime.close()


async def test_receipt_write_failure_stops_and_recovery_preserves_answer(
    tmp_path, monkeypatch
) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    complete = FileProductStore.complete_command

    def fail_receipt(self, receipt):
        raise OSError("injected receipt failure")

    try:
        monkeypatch.setattr(FileProductStore, "complete_command", fail_receipt)
        with pytest.raises(OSError):
            await runtime.submit_command("conversation-1", _answer_command(), context)
        assert adapter.closed == ["conversation-1"]
        assert adapter.input_responses == []
        monkeypatch.setattr(FileProductStore, "complete_command", complete)
        await runtime.get_snapshot("conversation-1", context)
        assert store.load_input_command_receipts()["answer-1"][0].status == "completed"
        resolutions = [e for e in store.replay() if e.type == "input.resolved"]
        assert len(resolutions) == 1 and resolutions[0].payload["outcome"] == "answered"
        await runtime.submit_command(
            "conversation-1",
            UserMessageCommand(
                command_id="manual-continue", content=(TextContent(text="continue"),)
            ),
            context,
        )
        assert not store.load_snapshot().automatic_turns_suspended
        await runtime.close()
        restarted = ConversationRuntime(tmp_path / "conversations", _input_adapter())
        try:
            snapshot = await restarted.get_snapshot("conversation-1", context)
            assert not snapshot.automatic_turns_suspended
        finally:
            await restarted.close()
    finally:
        await runtime.close()


async def test_concurrent_answer_and_cancel_have_one_resolution(tmp_path) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    try:
        results = await asyncio.gather(
            runtime.submit_command("conversation-1", _answer_command(), context),
            runtime.submit_command(
                "conversation-1",
                _answer_command(command_id="cancel", status="cancelled"),
                context,
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(result, ProductRuntimeError) for result in results) == 1
        assert len(adapter.input_responses) == 1
        assert len([e for e in store.replay() if e.type == "input.resolved"]) == 1
    finally:
        await runtime.close()


async def test_http_cancellation_does_not_lose_a_saved_answer(
    tmp_path, monkeypatch
) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    original = FileProductStore.resolve_pending_input
    saved = threading.Event()
    release = threading.Event()

    def save_then_wait(self, resolution):
        result = original(self, resolution)
        saved.set()
        assert release.wait(3)
        return result

    monkeypatch.setattr(FileProductStore, "resolve_pending_input", save_then_wait)
    request = asyncio.create_task(
        runtime.submit_command("conversation-1", _answer_command(), context)
    )
    try:
        await _wait_for(saved.is_set)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        release.set()
        await _wait_for(lambda: adapter.input_responses)
        assert store.load_input_command_receipts()["answer-1"][0].status == "completed"
        assert len([e for e in store.replay() if e.type == "input.resolved"]) == 1
    finally:
        release.set()
        await runtime.close()


async def test_new_message_waits_until_failed_execution_is_closed(
    tmp_path, monkeypatch
) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    adapter.fail_input_delivery = True
    removed = asyncio.Event()
    allow_close = asyncio.Event()
    close = adapter.close

    async def slow_close(handle):
        await close(handle)
        removed.set()
        await allow_close.wait()

    monkeypatch.setattr(adapter, "close", slow_close)
    answer = asyncio.create_task(
        runtime.submit_command("conversation-1", _answer_command(), context)
    )
    message = None
    try:
        await asyncio.wait_for(removed.wait(), 3)
        message = asyncio.create_task(
            runtime.submit_command(
                "conversation-1",
                UserMessageCommand(
                    command_id="after-failure", content=(TextContent(text="continue"),)
                ),
                context,
            )
        )
        await asyncio.sleep(0.05)
        assert not message.done(), (
            "a new message must not attach while the old execution closes"
        )
        allow_close.set()
        await answer
        receipt = await asyncio.wait_for(message, 3)
        assert receipt.status == "completed"
        assert "conversation-1" in runtime._active
        assert not runtime._active["conversation-1"].task.done()
        assert not store.load_snapshot().automatic_turns_suspended
    finally:
        allow_close.set()
        await asyncio.gather(
            answer, *([message] if message is not None else []), return_exceptions=True
        )
        await runtime.close()


async def test_accepted_message_cannot_clear_a_newer_suspension(
    tmp_path, monkeypatch
) -> None:
    runtime, adapter, context, store = await _runtime_with_pending_request(tmp_path)
    send = adapter.send
    try:
        await runtime.submit_command(
            "conversation-1", _answer_command(status="cancelled"), context
        )

        async def new_interruption(handle, command, context):
            if isinstance(command, UserMessageCommand):
                await runtime._user_input.suspend(
                    store, "conversation-1", "new_input_interrupted"
                )
            return await send(handle, command, context)

        monkeypatch.setattr(adapter, "send", new_interruption)
        await runtime.submit_command(
            "conversation-1",
            UserMessageCommand(
                command_id="resume", content=(TextContent(text="continue"),)
            ),
            context,
        )
        assert store.load_snapshot().automatic_turns_suspended
    finally:
        await runtime.close()
