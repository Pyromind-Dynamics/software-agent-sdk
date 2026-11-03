from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import harness_adapter.pi_adapter.adapter as pi_adapter_module
import pytest
from harness_adapter.pi_adapter import PiAdapter
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.runner import PiRunnerExit, PlannedPiRunnerExitReason
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import (
    CancelCommand,
    InputResponseCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import (
    AttachRecovery,
    InterruptedInputRecord,
    SessionSpec,
)


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


class _RecordingRunner:
    instances: ClassVar[list[_RecordingRunner]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.exit_handler = kwargs["exit_handler"]
        self.request_handler = kwargs["request_handler"]
        self.running = False
        self.pid = 7000 + len(_RecordingRunner.instances)
        self.requests: list[tuple[str, dict[str, Any]]] = []
        _RecordingRunner.instances.append(self)

    async def start(self, _config: dict[str, Any]) -> None:
        self.running = True
        self.start_params = _config

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        self.requests.append((method, params))
        if method == "run.suspend":
            return {"suspended": True}
        if method == "fork":
            branch_path = Path(params["target_session_dir"]) / "branched-session.jsonl"
            branch_path.write_text("", encoding="utf-8")
            return {"session_path": str(branch_path)}
        return {"accepted": True}

    async def close(self, *, reason: PlannedPiRunnerExitReason = "shutdown") -> None:
        if not self.running:
            return
        self.running = False
        await self.exit_handler(PiRunnerExit(returncode=0, reason=reason))

    async def terminate(
        self, *, reason: PlannedPiRunnerExitReason = "shutdown"
    ) -> None:
        self.running = False


@pytest.fixture(autouse=True)
def _fake_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingRunner.instances.clear()
    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", _RecordingRunner)
    # Attach restores the model configuration and reads the API key from the
    # environment.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


async def _new_session(
    adapter: PiAdapter,
    conversations,
    *,
    enabled: bool = True,
) -> tuple[Any, Any]:
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="conversation-1",
            user_id="42",
            workspace_root=str(conversations / "conversation-1"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
            user_input_enabled=enabled,
        ),
        RequestContext(user_id="42"),
    )
    session = adapter._sessions[handle.session_id]
    return handle, session


async def _drain(session) -> list[Any]:
    events = []
    while not session.queue.empty():
        events.append(session.queue.get_nowait())
    return events


async def test_handle_capabilities_reflect_session_enablement(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="disabled-session",
            user_id="42",
            workspace_root=str(conversations / "disabled-session"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
            user_input_enabled=False,
        ),
        RequestContext(user_id="42"),
    )
    session = adapter._sessions[handle.session_id]
    assert session.user_input_enabled is False
    assert adapter._handle(session).capabilities.user_input_reply is False
    assert (
        PiSessionFiles(conversations / handle.session_id).load_session()[
            "user_input_enabled"
        ]
        is False
    )
    await adapter.close(handle)

    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="enabled-session",
            user_id="42",
            workspace_root=str(conversations / "enabled-session"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
            user_input_enabled=True,
        ),
        RequestContext(user_id="42"),
    )
    session = adapter._sessions[handle.session_id]
    assert adapter._handle(session).capabilities.user_input_reply is True
    assert (
        PiSessionFiles(conversations / handle.session_id).load_session()[
            "user_input_enabled"
        ]
        is True
    )
    # The enabled setting remains persisted after reattaching.
    await adapter.close(handle)
    attached = await adapter.attach_session(
        "enabled-session",
        RequestContext(user_id="42"),
        recovery=AttachRecovery(automatic_turns_suspended=True),
    )
    try:
        assert adapter._sessions["enabled-session"].user_input_enabled is True
        assert (
            adapter._handle(
                adapter._sessions["enabled-session"]
            ).capabilities.user_input_reply
            is True
        )
    finally:
        await adapter.close(attached)


async def test_input_request_emits_event_and_answer_resolves_wait(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle, session = await _new_session(adapter, conversations)
    try:
        session.files.save_inflight({"run_id": "run-1"})
        wait_task = asyncio.ensure_future(
            adapter._runner_request(
                session,
                "input.request",
                {
                    "request_id": "input:1",
                    "tool_call_id": "call-1",
                    "questions": QUESTIONS,
                },
            )
        )
        await asyncio.sleep(0.01)
        events = await _drain(session)
        requested = next(event for event in events if event.type == "input.requested")
        assert requested.run_id == "run-1"
        assert requested.payload["operation_id"] == "call-1"
        assert requested.payload["questions"][0]["id"] == "duplicate_policy"
        # After the answer, ask_user continues in the same run until completion.
        await adapter.send(
            handle,
            InputResponseCommand.model_validate(
                {
                    "command_id": "answer-1",
                    "request_id": "input:1",
                    "status": "answered",
                    "answers": [
                        {"question_id": "duplicate_policy", "option_id": "keep_first"},
                        {"question_id": "output_name", "text": "cleaned_data.csv"},
                    ],
                }
            ),
            session.context,
        )
        resolved = await asyncio.wait_for(wait_task, timeout=3)
        assert resolved["status"] == "answered"
        assert resolved["answers"][0]["option_id"] == "keep_first"
        assert resolved["answers"][1]["text"] == "cleaned_data.csv"
        assert session.pending_user_inputs == {}
    finally:
        await adapter.close(handle)


async def test_input_request_rejects_disabled_session(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    _, session = await _new_session(adapter, conversations, enabled=False)
    try:
        with pytest.raises(ValueError, match="not enabled"):
            await adapter._runner_request(
                session,
                "input.request",
                {
                    "request_id": "input:1",
                    "tool_call_id": "call-1",
                    "questions": QUESTIONS,
                },
            )
    finally:
        await adapter.close(adapter._handle(session))


async def test_input_cancel_cleans_waiting_request(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    _, session = await _new_session(adapter, conversations)
    try:
        assert await adapter._runner_request(
            session, "input.cancel", {"request_id": "unknown"}
        ) == {"cancelled": True}
        assert session.pending_user_inputs == {}
        assert session.input_calls == {}
        for index, request_id in enumerate(("unknown", "input:1", "input:1")):
            # Match JSONL dispatch: queue request and cancel without yielding.
            wait_task = asyncio.create_task(
                adapter._runner_request(
                    session,
                    "input.request",
                    {
                        "request_id": request_id,
                        "tool_call_id": f"call-{index}",
                        "questions": QUESTIONS,
                    },
                )
            )
            cancel_task = asyncio.create_task(
                adapter._runner_request(
                    session, "input.cancel", {"request_id": request_id}
                )
            )
            assert await cancel_task == {"cancelled": True}
            assert await wait_task == {"status": "cancelled"}
            assert session.pending_user_inputs == {}
            assert session.input_calls == {}
            assert await adapter._runner_request(
                session, "input.cancel", {"request_id": request_id}
            ) == {"cancelled": True}
            assert session.pending_user_inputs == {}
            assert session.input_calls == {}
            events = await _drain(session)
            assert [
                event.type for event in events if event.type.startswith("input.")
            ] == [
                "input.requested",
                "input.resolved",
            ]
    finally:
        await adapter.close(adapter._handle(session))


async def test_cancel_response_suspends_runner(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle, session = await _new_session(adapter, conversations)
    try:
        wait_task = asyncio.ensure_future(
            adapter._runner_request(
                session,
                "input.request",
                {
                    "request_id": "input:1",
                    "tool_call_id": "call-1",
                    "questions": QUESTIONS,
                },
            )
        )
        await asyncio.sleep(0.01)
        result = await adapter.send(
            handle,
            InputResponseCommand.model_validate(
                {
                    "command_id": "cancel-1",
                    "request_id": "input:1",
                    "status": "cancelled",
                }
            ),
            session.context,
        )
        assert result == {"suspended": True}
        assert await wait_task == {"status": "cancelled"}
        runner = session.runner
        assert runner is not None
        methods = [method for method, _ in runner.requests]
        # Disable scheduling before aborting; submitted platform tasks do not go
        # through stop_platform_task.
        assert methods[-1] == "run.suspend"
    finally:
        await adapter.close(handle)


async def test_unexpected_runner_exit_interrupts_waiting_inputs(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle = None
    handle, session = await _new_session(adapter, conversations)
    try:
        session.files.save_inflight({"run_id": "run-1"})
        wait_task = asyncio.ensure_future(
            adapter._runner_request(
                session,
                "input.request",
                {
                    "request_id": "input:1",
                    "tool_call_id": "call-1",
                    "questions": QUESTIONS,
                },
            )
        )
        await asyncio.sleep(0.01)
        runner = session.runner
        assert runner is not None
        # On unexpected exit, end the in-memory wait and emit
        # input.resolved(interrupted).
        await runner.exit_handler(PiRunnerExit(returncode=1, reason="unexpected"))
        resolved = await asyncio.wait_for(wait_task, timeout=3)
        assert resolved == {"status": "interrupted"}
        events = await _drain(session)
        interrupted_event = next(
            event
            for event in events
            if event.type == "input.resolved"
            and event.payload.get("request_id") == "input:1"
        )
        assert interrupted_event.payload["outcome"] == "interrupted"
        assert interrupted_event.source_event_id == "input:input:1:resolution"
    finally:
        if handle is not None:
            await adapter.close(handle)


async def test_attach_recovery_flags_forward_to_runner_start(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="conversation-1",
            user_id="42",
            workspace_root=str(conversations / "conversation-1"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
            user_input_enabled=True,
        ),
        RequestContext(user_id="42"),
    )
    session = adapter._sessions[handle.session_id]
    runner = session.runner
    assert runner is not None
    # _start_runner passes start configuration to the fake runner; verify the
    # session setting here.
    assert runner.running is True
    assert session.user_input_enabled is True
    await adapter.close(handle)

    # Verify the session retains recovery settings after restart and attach.
    restarted = PiAdapter(conversations, terminal_backend="os-sandbox")
    recovery = AttachRecovery(
        automatic_turns_suspended=True,
        interrupted_inputs=(
            InterruptedInputRecord(request_id="input:1", operation_id="call-1"),
        ),
    )
    attached = await restarted.attach_session(
        "conversation-1", RequestContext(user_id="42"), recovery=recovery
    )
    try:
        attached_session = restarted._sessions["conversation-1"]
        assert attached_session.attach_recovery is recovery
        assert attached_session.user_input_enabled is True
    finally:
        await restarted.close(attached)


async def test_fork_inherits_user_input_configuration(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    adapter = PiAdapter(
        conversations, terminal_backend="os-sandbox", knowledge_root=knowledge
    )
    handle = await adapter.create_session(
        SessionSpec(
            conversation_id="source",
            user_id="42",
            workspace_root=str(conversations / "source"),
            model_configuration={"model": "gpt-5", "api_key": "test-key"},
            user_input_enabled=True,
        ),
        RequestContext(user_id="42"),
    )
    session = adapter._sessions["source"]
    session.files.save_inflight({"run_id": "run-1"})
    from pyromind_runtime.domain.events import HarnessEvent

    session.queue.put_nowait(
        HarnessEvent(
            session_id="source",
            run_id="run-1",
            type="workflow.modified",
            payload={},
        )
    )
    session.files.save_inflight({"run_id": "run-1", "workflow_modified": True})
    # Construct a checkpoint directly to exercise the fork path.
    from pyromind_runtime.domain.snapshot import WorkflowState
    from pyromind_runtime.ports.harness import ForkSpec, ProductCheckpoint

    session.files.save_checkpoint_index({"event-1": "leaf-1"})
    checkpoint = ProductCheckpoint(
        event_id="event-1",
        through_seq=1,
        workflow=WorkflowState(
            resource_id="workflow", version="v1", dsl="workflow = InputNode()"
        ),
        adapter_checkpoint_ref="leaf-1",
    )
    target = await adapter.fork(
        handle,
        ForkSpec(
            source_conversation_id="source",
            target_conversation_id="target",
            event_id="event-1",
        ),
        checkpoint,
        RequestContext(user_id="42"),
    )
    try:
        forked = adapter._sessions["target"]
        assert forked.user_input_enabled is True
        # Fork does not copy pending requests or in-memory waiters.
        assert forked.pending_user_inputs == {}
        assert forked.attach_recovery is None
    finally:
        await adapter.close(target)
        await adapter.close(handle)


@pytest.mark.parametrize(
    "scenario", ["runner_exit", "persistence_failure", "global_stop"]
)
async def test_runtime_adapter_input_cleanup(tmp_path, monkeypatch, scenario) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(conversations, adapter, default_harness_id="pi")
    context = RequestContext(user_id="42")
    wait = None
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="conversation-1",
                user_id="42",
                workspace_root=str(conversations / "conversation-1"),
                model_configuration={"model": "gpt-5", "api_key": "test-key"},
                user_input_enabled=True,
            ),
            context,
        )
        session = adapter._sessions["conversation-1"]
        session.files.save_inflight({"run_id": "run-1"})
        store = runtime._store("conversation-1")
        if scenario == "persistence_failure":
            original = FileProductStore.append

            def fail_requested(self, event):
                if event.type == "input.requested":
                    raise OSError("input log unavailable")
                return original(self, event)

            monkeypatch.setattr(FileProductStore, "append", fail_requested)
        wait = asyncio.create_task(
            adapter._runner_request(
                session,
                "input.request",
                {
                    "request_id": "input-1",
                    "tool_call_id": "call-1",
                    "questions": QUESTIONS,
                },
            )
        )
        async with asyncio.timeout(3):
            if scenario == "persistence_failure":
                while "conversation-1" in runtime._active:
                    await asyncio.sleep(0.01)
                assert "conversation-1" not in adapter._sessions
            else:
                while not store.load_snapshot().pending_inputs:
                    await asyncio.sleep(0.01)
                if scenario == "runner_exit":
                    runner = _RecordingRunner.instances[-1]
                    assert runner is session.runner
                    await runner.exit_handler(
                        PiRunnerExit(returncode=1, reason="unexpected")
                    )
                else:
                    await runtime.submit_command(
                        "conversation-1", CancelCommand(command_id="stop"), context
                    )
                while store.load_snapshot().pending_inputs:
                    await asyncio.sleep(0.01)
            assert await wait == {"status": "interrupted"}
        assert not session.pending_user_inputs
        snapshot = store.load_snapshot()
        assert not snapshot.pending_inputs
        assert snapshot.automatic_turns_suspended
    finally:
        await runtime.close()
        if wait is not None:
            await asyncio.gather(wait, return_exceptions=True)


@pytest.mark.parametrize("python_waiting", [False, True])
async def test_realtime_input_rejection_precedes_workflow_sync(
    tmp_path, monkeypatch, python_waiting
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle, session = await _new_session(adapter, conversations)
    synced = []

    async def sync(*args):
        synced.append(True)

    async def reject(method, params):
        if method == "prompt.cancel":
            return {"released": True}
        assert method == "prompt.prepare"
        raise pi_adapter_module.PiRunnerError("input_pending", code="input_pending")

    monkeypatch.setattr(adapter, "_sync_xyflow", sync)
    monkeypatch.setattr(session.runner, "request", reject)
    if python_waiting:
        session.pending_user_inputs["pending"] = (
            asyncio.get_running_loop().create_future()
        )
    try:
        with pytest.raises(ProductRuntimeError) as error:
            await adapter.send(
                handle,
                UserMessageCommand(
                    command_id="message",
                    content=(TextContent(text="hi"),),
                    workflow_xyflow={"nodes": []},
                ),
                session.context,
            )
        assert error.value.code == "input_pending"
        assert synced == []
    finally:
        await adapter.close(handle)


async def test_failed_workflow_sync_releases_prompt_admission(
    tmp_path, monkeypatch
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle, session = await _new_session(adapter, conversations)
    runner = session.runner

    async def fail_sync(*args):
        raise ValueError("invalid workflow")

    monkeypatch.setattr(adapter, "_sync_xyflow", fail_sync)
    try:
        with pytest.raises(ValueError, match="invalid workflow"):
            await adapter.send(
                handle,
                UserMessageCommand(
                    command_id="message",
                    content=(TextContent(text="hi"),),
                    workflow_xyflow={"nodes": []},
                ),
                session.context,
            )
        assert [method for method, _ in runner.requests] == [
            "prompt.prepare",
            "prompt.cancel",
        ]
    finally:
        await adapter.close(handle)


async def test_disabled_input_preserves_workflow_sync_before_runner_start(
    tmp_path, monkeypatch
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = PiAdapter(conversations, terminal_backend="os-sandbox")
    handle, session = await _new_session(adapter, conversations, enabled=False)
    order = []
    ensure = adapter._ensure_runner

    async def record_ensure(session):
        order.append("ensure_runner")
        await ensure(session)

    async def record_sync(*args):
        order.append("sync")

    monkeypatch.setattr(adapter, "_ensure_runner", record_ensure)
    monkeypatch.setattr(adapter, "_sync_xyflow", record_sync)
    try:
        await adapter.send(
            handle,
            UserMessageCommand(
                command_id="message",
                content=(TextContent(text="hi"),),
                workflow_xyflow={"nodes": []},
            ),
            session.context,
        )
        assert order == ["sync", "ensure_runner"]
        assert [method for method, _ in session.runner.requests] == ["prompt"]
    finally:
        await adapter.close(handle)
