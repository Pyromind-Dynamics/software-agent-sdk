from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import (
    RollbackWorkflowCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter


def test_list_orders_by_activity_and_recovers_legacy_timestamps(tmp_path) -> None:
    runtime = ConversationRuntime(tmp_path, FakeAdapter())
    context = RequestContext(user_id="42")
    for conversation_id, day, count in [("old", 1, 3), ("recent", 2, 1)]:
        directory = tmp_path / conversation_id
        directory.mkdir()
        store = FileProductStore(directory)
        store.create(
            ConversationSnapshot(
                conversation_id=conversation_id,
                capabilities=HarnessCapabilities(),
            ),
            user_id="42",
        )
        for index in range(count):
            store.append(
                ProductEvent(
                    conversation_id=conversation_id,
                    type="status.changed",
                    payload={"status": "idle"},
                    occurred_at=datetime(2026, 9, day, tzinfo=UTC),
                )
            )
        legacy = json.loads(store.snapshot_path.read_text())
        legacy.pop("updated_at")
        store.snapshot_path.write_text(json.dumps(legacy))

    snapshots = runtime.list_snapshots(context)
    assert [item.conversation_id for item in snapshots] == ["recent", "old"]
    assert snapshots[0].updated_at == datetime(2026, 9, 2, tzinfo=UTC)
    assert runtime.list_snapshots(context) == snapshots


def _write_event_without_offset(path, index: int) -> None:
    """Rewrite one persisted event the way pre-backfill releases wrote it."""
    lines = path.read_text().splitlines()
    event = json.loads(lines[index])
    event["occurred_at"] = event["occurred_at"].removesuffix("Z")
    lines[index] = json.dumps(event)
    path.write_text("\n".join(lines) + "\n")


def test_list_recovers_legacy_naive_event_timestamps(tmp_path) -> None:
    """Events persisted without an offset must not abort the whole listing.

    ``max(snapshot.updated_at, event.occurred_at)`` compared an aware and a
    naive datetime and raised ``TypeError`` while replaying the legacy event,
    which surfaced as a 500 on ``GET /conversations``.
    """
    runtime = ConversationRuntime(tmp_path, FakeAdapter())
    context = RequestContext(user_id="42")
    directory = tmp_path / "legacy"
    directory.mkdir()
    store = FileProductStore(directory)
    store.create(
        ConversationSnapshot(
            conversation_id="legacy",
            capabilities=HarnessCapabilities(),
        ),
        user_id="42",
    )
    for day in (1, 2):
        store.append(
            ProductEvent(
                conversation_id="legacy",
                type="status.changed",
                payload={"status": "idle"},
                occurred_at=datetime(2026, 9, day, tzinfo=UTC),
            )
        )
    _write_event_without_offset(store.events_path, 1)
    legacy = json.loads(store.snapshot_path.read_text())
    legacy.pop("updated_at")
    store.snapshot_path.write_text(json.dumps(legacy))

    snapshots = runtime.list_snapshots(context)

    assert [item.conversation_id for item in snapshots] == ["legacy"]
    assert snapshots[0].updated_at == datetime(2026, 9, 2, tzinfo=UTC)


def test_list_skips_conversation_that_fails_to_load(tmp_path, monkeypatch) -> None:
    """A single unreadable conversation must not take the listing down."""
    runtime = ConversationRuntime(tmp_path, FakeAdapter())
    context = RequestContext(user_id="42")
    for conversation_id in ("healthy", "broken"):
        directory = tmp_path / conversation_id
        directory.mkdir()
        store = FileProductStore(directory)
        store.create(
            ConversationSnapshot(
                conversation_id=conversation_id,
                capabilities=HarnessCapabilities(),
            ),
            user_id="42",
        )
        store.append(
            ProductEvent(
                conversation_id=conversation_id,
                type="status.changed",
                payload={"status": "idle"},
                occurred_at=datetime(2026, 9, 1, tzinfo=UTC),
            )
        )
    load_snapshot = FileProductStore.load_snapshot

    def flaky(self: FileProductStore) -> ConversationSnapshot:
        if self.conversation_dir.name == "broken":
            raise RuntimeError("corrupted snapshot")
        return load_snapshot(self)

    monkeypatch.setattr(FileProductStore, "load_snapshot", flaky)

    snapshots = runtime.list_snapshots(context)

    assert [item.conversation_id for item in snapshots] == ["healthy"]


async def test_runtime_keeps_product_data_inside_conversation(tmp_path) -> None:
    conversations = tmp_path / "workspace" / "conversations"
    conversations.mkdir(parents=True)
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")
    snapshot = await runtime.create_conversation(
        SessionSpec(
            conversation_id="conversation-1",
            user_id="42",
            workspace_root=str(conversations),
            initial_message=(TextContent(text="hello"),),
        ),
        context,
    )

    conversation = conversations / snapshot.conversation_id
    assert (conversation / "public_data").is_dir()
    assert (conversation / "product" / "snapshot.json").is_file()
    assert not (tmp_path / "public_data").exists()
    assert not (tmp_path / "workspace" / "product_conversations").exists()
    assert [item.kind for item in snapshot.timeline] == ["message"]
    await runtime.close()


async def test_cached_session_reattaches_to_survive_harness_eviction(
    tmp_path,
) -> None:
    """Harnesses reclaim idle conversations behind the product layer's back.

    ``_ensure_active`` must re-attach on cache hits so an evicted harness service
    is re-activated rather than replayed into, which used to surface as a
    terminal ``inactive_service`` conflict on every later command.
    """
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="conversation-evicted",
            user_id="42",
            workspace_root=str(conversations),
        ),
        context,
    )
    assert adapter.attached == []

    await runtime.get_snapshot("conversation-evicted", context)
    await runtime.get_snapshot("conversation-evicted", context)

    assert adapter.attached == ["conversation-evicted", "conversation-evicted"]
    await runtime.close()


async def test_command_forwards_ephemeral_cookie_and_cluster(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(
        user_id="42",
        cookie="auth_token=secret",
        x_cluster="us-west-1#pre",
    )
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="conversation-2",
            user_id="42",
            workspace_root=str(conversations),
        ),
        context,
    )
    receipt = await runtime.submit_command(
        "conversation-2",
        UserMessageCommand(
            command_id="command-1",
            content=(TextContent(text="continue"),),
        ),
        context,
    )

    assert receipt.status == "completed"
    assert adapter.sent[0][2].cookie == "auth_token=secret"
    assert adapter.sent[0][2].x_cluster == "us-west-1#pre"
    product_files = (conversations / "conversation-2" / "product").iterdir()
    assert all(
        "secret" not in path.read_text(errors="ignore") for path in product_files
    )
    await runtime.close()


async def test_stream_replays_then_continues_without_sequence_gap(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")
    snapshot = await runtime.create_conversation(
        SessionSpec(
            conversation_id="conversation-3",
            user_id="42",
            workspace_root=str(conversations),
        ),
        context,
    )
    stream = runtime.stream_events("conversation-3", 0, context)
    first = await anext(stream)
    assert first.seq == 1
    assert first.type == "conversation.created"

    pending = asyncio.create_task(anext(stream))
    adapter.emit(
        "conversation-3",
        "status.changed",
        {"status": "running"},
        event_id="status-running",
    )
    second = await asyncio.wait_for(pending, timeout=1)
    assert second.seq == snapshot.through_seq + 1
    assert second.type == "status.changed"
    await stream.aclose()
    await runtime.close()


async def test_runtime_routes_existing_session_by_persisted_harness(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    creator = FakeAdapter("pi")
    runtime = ConversationRuntime(
        conversations,
        {"openhands": FakeAdapter(), "pi": creator},
        default_harness_id="pi",
    )
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="pi-conversation",
            user_id="42",
            workspace_root=str(conversations / "pi-conversation"),
        ),
        RequestContext(user_id="42"),
    )
    await runtime.close()

    openhands = FakeAdapter("openhands")
    pi = FakeAdapter("pi")
    restarted = ConversationRuntime(
        conversations,
        {"openhands": openhands, "pi": pi},
        default_harness_id="openhands",
    )
    snapshot = await restarted.get_snapshot(
        "pi-conversation", RequestContext(user_id="42")
    )
    assert snapshot.conversation_id == "pi-conversation"
    assert pi.queues == {} and openhands.queues == {}

    await restarted.submit_command(
        "pi-conversation",
        UserMessageCommand(
            command_id="command-1",
            content=(TextContent(text="continue"),),
        ),
        RequestContext(user_id="42"),
    )
    assert "pi-conversation" in pi.queues
    assert "pi-conversation" not in openhands.queues
    await restarted.close()


async def test_runtime_resumes_callback_through_owning_adapter(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversation = conversations / "callback-conversation"
    conversation.mkdir(parents=True)
    store = FileProductStore(conversation)
    store.create(
        ConversationSnapshot(
            conversation_id="callback-conversation",
            capabilities=HarnessCapabilities(cancel=True),
        ),
        user_id="42",
        harness_id="openhands",
    )
    store.append(
        ProductEvent(
            conversation_id="callback-conversation",
            type="external_task.submitted",
            payload={
                "task_id": "task-1",
                "kind": "data_cleaning",
                "run_id": "run-1",
                "status": "running",
                "output_dir": "/outputs/run-1",
                "submitted_at": "2026-08-24T00:00:00+00:00",
                "updated_at": "2026-08-24T00:00:00+00:00",
                "resume_pending": False,
            },
        )
    )
    adapter = FakeAdapter("openhands")
    runtime = ConversationRuntime(conversations, {"openhands": adapter})

    await runtime.deliver_external_task_status(
        "callback-conversation", task_id="task-1", status="Succeeded"
    )
    assert store.load_snapshot().external_tasks[0].resume_pending is True
    assert adapter.external_task_notifications == []

    snapshot = await runtime.get_snapshot(
        "callback-conversation", RequestContext(user_id="42")
    )

    assert snapshot.external_tasks[0].resume_pending is False
    assert adapter.external_task_notifications[0][1].status == "succeeded"
    await runtime.close()


async def test_runtime_logs_first_message_latency_metrics(tmp_path, caplog) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")

    with caplog.at_level(
        "INFO",
        logger="pyromind_runtime.application.conversation_runtime",
    ):
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="conversation-metrics",
                user_id="42",
                workspace_root=str(conversations),
            ),
            context,
        )
        await runtime.submit_command(
            "conversation-metrics",
            UserMessageCommand(
                command_id="first-command",
                content=(TextContent(text="hello"),),
            ),
            context,
        )
        adapter.emit(
            "conversation-metrics",
            "message.delta",
            {"message_id": "assistant-1", "text": "H"},
            event_id="assistant-1:delta:1",
            run_id="first-command",
        )
        for _ in range(10):
            if "first_delta_latency_ms=" in caplog.text:
                break
            await asyncio.sleep(0)

    assert "adapter.create_session_ms=" in caplog.text
    assert "runtime.ready_wait_ms=" in caplog.text
    assert "product.create.total_ms=" in caplog.text
    assert "first_command.accept_ms=" in caplog.text
    assert "first_delta_latency_ms=" in caplog.text
    assert "conversation_id=conversation-metrics" in caplog.text
    await runtime.close()


async def test_runtime_fork_replays_shared_product_history_to_checkpoint(
    tmp_path,
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter("pi")
    runtime = ConversationRuntime(
        conversations, {"pi": adapter}, default_harness_id="pi"
    )
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="fork-source",
            user_id="42",
            workspace_root=str(conversations / "fork-source"),
        ),
        context,
    )
    source_store = FileProductStore(conversations / "fork-source")
    source_store.append(
        ProductEvent(
            event_id="workflow-v1",
            conversation_id="fork-source",
            type="workflow.updated",
            source_event_id="pi-entry-1",
            payload={
                "resource_id": "pyromind_workflow",
                "version": "v1",
                "dsl": "workflow = InputNode()",
                "canvas": {"nodes": []},
            },
        )
    )
    source_store.append(
        ProductEvent(
            event_id="usage-after-checkpoint",
            conversation_id="fork-source",
            type="usage.updated",
            payload={"input_tokens": 10, "output_tokens": 5},
        )
    )

    target = await runtime.fork_conversation(
        "fork-source", event_id="workflow-v1", title="Forked", context=context
    )

    assert target.conversation_id != "fork-source"
    assert target.current_workflow is not None
    assert target.current_workflow.version == "v1"
    assert target.usage.input_tokens == 0
    assert adapter.forks[0][0].target_conversation_id == target.conversation_id
    assert adapter.forks[0][1].adapter_checkpoint_ref == "pi-entry-1"
    target_events = FileProductStore(conversations / target.conversation_id).replay()
    assert "usage-after-checkpoint" not in {event.event_id for event in target_events}
    await runtime.close()


async def test_runtime_rollback_uses_shared_checkpoint_and_is_idempotent(
    tmp_path,
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="rollback-source",
            user_id="42",
            workspace_root=str(conversations / "rollback-source"),
        ),
        context,
    )
    store = FileProductStore(conversations / "rollback-source")
    for event_id, version, dsl in (
        ("workflow-v1", "v1", "workflow = InputNode()"),
        ("workflow-v2", "v2", "workflow = OutputNode()"),
    ):
        store.append(
            ProductEvent(
                event_id=event_id,
                conversation_id="rollback-source",
                type="workflow.updated",
                source_event_id=f"native-{version}",
                payload={
                    "resource_id": "pyromind_workflow",
                    "version": version,
                    "dsl": dsl,
                    "canvas": {"nodes": []},
                },
            )
        )
    command = RollbackWorkflowCommand(
        command_id="rollback-command", event_id="workflow-v1"
    )

    first = await runtime.submit_command("rollback-source", command, context)
    second = await runtime.submit_command("rollback-source", command, context)

    assert first == second
    assert first.status == "completed"
    assert first.response["rolled_back_to_event_id"] == "workflow-v1"
    assert len(adapter.restores) == 1
    assert adapter.sent == []
    restored_workflow = store.load_snapshot().current_workflow
    assert restored_workflow is not None
    assert restored_workflow.version == "v1"
    await runtime.close()


async def test_runtime_records_busy_rollback_as_failed_command(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    runtime = ConversationRuntime(conversations, FakeAdapter())
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="busy-rollback",
            user_id="42",
            workspace_root=str(conversations / "busy-rollback"),
        ),
        context,
    )
    store = FileProductStore(conversations / "busy-rollback")
    store.append(
        ProductEvent(
            event_id="workflow-v1",
            conversation_id="busy-rollback",
            type="workflow.updated",
            payload={
                "resource_id": "pyromind_workflow",
                "version": "v1",
                "dsl": "workflow = InputNode()",
                "canvas": None,
            },
        )
    )
    store.append(
        ProductEvent(
            event_id="running",
            conversation_id="busy-rollback",
            type="status.changed",
            payload={"status": "running"},
        )
    )
    command = RollbackWorkflowCommand(command_id="busy-command", event_id="workflow-v1")

    with pytest.raises(ProductRuntimeError) as raised:
        await runtime.submit_command("busy-rollback", command, context)
    receipt = await runtime.submit_command("busy-rollback", command, context)

    assert raised.value.code == "conversation_busy"
    assert receipt.status == "failed"
    assert receipt.response["code"] == "conversation_busy"
    await runtime.close()


async def test_runtime_owns_workflow_debug_callback_policy(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter("pi")
    runtime = ConversationRuntime(
        conversations, {"pi": adapter}, default_harness_id="pi"
    )
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="debug-callback",
            user_id="42",
            workspace_root=str(conversations / "debug-callback"),
        ),
        context,
    )
    runtime.register_external_task(
        "debug-callback",
        {
            "task_id": "debug-failed",
            "kind": "workflow_debug",
            "run_id": "debug-failed",
            "status": "running",
            "output_dir": None,
            "attempt": 3,
            "max_attempts": 10,
            "keep_ui_lock": True,
            "submitted_at": "2026-08-27T00:00:00+00:00",
            "updated_at": "2026-08-27T00:00:00+00:00",
            "resume_pending": False,
        },
    )
    runtime.register_external_task(
        "debug-callback",
        {
            "task_id": "debug-succeeded",
            "kind": "workflow_debug",
            "run_id": "debug-succeeded",
            "status": "running",
            "output_dir": None,
            "attempt": 1,
            "max_attempts": 10,
            "keep_ui_lock": True,
            "submitted_at": "2026-08-27T00:00:00+00:00",
            "updated_at": "2026-08-27T00:00:00+00:00",
            "resume_pending": False,
        },
    )

    await runtime.deliver_external_task_status(
        "debug-callback", task_id="debug-failed", status="Failed"
    )
    await runtime.deliver_external_task_status(
        "debug-callback", task_id="debug-succeeded", status="Succeeded"
    )

    failed = adapter.external_task_notifications[0][1]
    succeeded = adapter.external_task_notifications[1][1]
    assert "analyze_task_failure" in failed.hidden_text
    assert failed.trigger_turn is True
    assert failed.reset_attempt_budget is False
    assert succeeded.reset_attempt_budget is True
    assert "do not submit workflow_debug again" in succeeded.hidden_text
    await runtime.close()


async def test_idle_conversation_is_evicted_and_reactivated(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, idle_eviction_seconds=3600)
    context = RequestContext(user_id="42")
    snapshot = await runtime.create_conversation(
        SessionSpec(
            conversation_id="conversation-evict",
            user_id="42",
            workspace_root=str(conversations),
        ),
        context,
    )

    active = runtime._active[snapshot.conversation_id]
    active.last_access -= 7200
    await runtime._evict_idle()

    assert snapshot.conversation_id not in runtime._active
    assert adapter.closed == [snapshot.conversation_id]

    await runtime.submit_command(
        snapshot.conversation_id,
        UserMessageCommand(
            command_id="command-retry",
            content=(TextContent(text="again"),),
        ),
        context,
    )
    assert snapshot.conversation_id in runtime._active
    assert runtime._active[snapshot.conversation_id].last_access > 0
    await runtime.close()


async def test_eviction_skips_running_or_subscribed_conversations(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, idle_eviction_seconds=3600)
    context = RequestContext(user_id="42")
    snapshot = await runtime.create_conversation(
        SessionSpec(
            conversation_id="conversation-guard",
            user_id="42",
            workspace_root=str(conversations),
        ),
        context,
    )
    active = runtime._active[snapshot.conversation_id]
    active.last_access -= 7200

    async for _ in runtime.stream_events(snapshot.conversation_id, 0, context):
        break

    await runtime._evict_idle()
    assert snapshot.conversation_id in runtime._active
    await runtime.close()


async def _wait_for_status(store: FileProductStore, expected: str) -> None:
    for _ in range(200):
        if store.load_snapshot().status == expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"conversation status never became {expected}")


async def _create(runtime: ConversationRuntime, root, conversation_id: str):
    return await runtime.create_conversation(
        SessionSpec(
            conversation_id=conversation_id,
            user_id="42",
            workspace_root=str(root / conversation_id),
        ),
        RequestContext(user_id="42"),
    )


def test_default_idle_release_window_is_five_minutes(tmp_path) -> None:
    runtime = ConversationRuntime(tmp_path, FakeAdapter())

    assert runtime._idle_eviction_seconds == 300
    assert runtime._release_grace_seconds == 300


async def test_finished_run_releases_runner_after_grace_period(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, release_grace_seconds=1)
    context = RequestContext(user_id="42")
    await _create(runtime, conversations, "conversation-grace")

    adapter.emit(
        "conversation-grace",
        "run.finished",
        {"status": "idle"},
        run_id="run-1",
        event_id="run-1:finished",
    )

    await asyncio.sleep(1.3)
    assert "conversation-grace" not in runtime._active
    assert adapter.closed == ["conversation-grace"]

    await runtime.submit_command(
        "conversation-grace",
        UserMessageCommand(command_id="command-1", content=(TextContent(text="hi"),)),
        context,
    )
    assert adapter.attached == ["conversation-grace"]
    assert "conversation-grace" in runtime._active
    await runtime.close()


async def test_promptless_conversation_is_released_after_grace(tmp_path) -> None:
    """A conversation created without a prompt has no run to end the session.

    The frontend can open a conversation before the user types anything, and
    that runner would otherwise stay resident for the whole idle window.
    """
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, release_grace_seconds=1)
    await _create(runtime, conversations, "conversation-empty")

    await asyncio.sleep(1.3)

    assert "conversation-empty" not in runtime._active
    assert adapter.closed == ["conversation-empty"]
    await runtime.close()


async def test_reads_do_not_reactivate_a_released_conversation(tmp_path) -> None:
    """Reads must not re-create a released session.

    Clients poll snapshots and re-open event streams while a conversation sits
    idle. Attaching on those reads re-created the harness session right after
    every release, so memory never dropped for a conversation anyone watched.
    """
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    context = RequestContext(user_id="42")
    await _create(runtime, conversations, "conversation-released")
    await runtime._release("conversation-released", reason="test")

    await runtime.get_snapshot("conversation-released", context)
    async for _ in runtime.stream_events("conversation-released", 0, context):
        break

    assert adapter.attached == []
    assert "conversation-released" not in runtime._active

    await runtime.submit_command(
        "conversation-released",
        UserMessageCommand(command_id="command-1", content=(TextContent(text="hi"),)),
        context,
    )
    assert adapter.attached == ["conversation-released"]
    await runtime.close()


async def test_released_conversation_keeps_streaming_to_watchers(tmp_path) -> None:
    """A watcher must not pin the runner, and must not lose events either.

    Events are persisted before they are published and the subscriber set is
    owned by the runtime, so the pump created by the next attach feeds the same
    queues a released conversation's watchers are already reading.
    """
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, release_grace_seconds=1)
    context = RequestContext(user_id="42")
    await _create(runtime, conversations, "conversation-stream")
    received: list[ProductEvent] = []

    async def consume() -> None:
        async for event in runtime.stream_events("conversation-stream", 0, context):
            received.append(event)

    consumer = asyncio.create_task(consume())
    adapter.emit(
        "conversation-stream",
        "run.finished",
        {"status": "idle"},
        run_id="run-1",
        event_id="run-1:finished",
    )

    await asyncio.sleep(1.3)
    assert "conversation-stream" not in runtime._active

    await runtime.submit_command(
        "conversation-stream",
        UserMessageCommand(command_id="command-1", content=(TextContent(text="hi"),)),
        context,
    )
    adapter.emit(
        "conversation-stream",
        "message.delta",
        {"message_id": "assistant-1", "text": "H"},
        run_id="command-1",
        event_id="assistant-1:delta:1",
    )
    for _ in range(50):
        if any(event.type == "message.delta" for event in received):
            break
        await asyncio.sleep(0.02)

    assert any(event.type == "message.delta" for event in received)
    consumer.cancel()
    await runtime.close()


async def test_grace_release_skips_command_already_in_flight(
    tmp_path, monkeypatch
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, release_grace_seconds=1)
    context = RequestContext(user_id="42")
    await _create(runtime, conversations, "conversation-busy")
    started = asyncio.Event()
    finish = asyncio.Event()

    async def blocked_send(handle, command, request_context):
        started.set()
        await finish.wait()
        return {"accepted": True}

    monkeypatch.setattr(adapter, "send", blocked_send)
    command = asyncio.create_task(
        runtime.submit_command(
            "conversation-busy",
            UserMessageCommand(
                command_id="command-busy", content=(TextContent(text="hi"),)
            ),
            context,
        )
    )
    await started.wait()
    adapter.emit(
        "conversation-busy",
        "run.finished",
        {"status": "idle"},
        run_id="run-2",
        event_id="run-2:finished",
    )

    await asyncio.sleep(1.3)
    assert "conversation-busy" in runtime._active
    assert adapter.closed == []

    finish.set()
    await command
    await runtime.close()


async def test_release_skips_conversation_with_running_external_task(
    tmp_path,
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, release_grace_seconds=1)
    await _create(runtime, conversations, "conversation-waiting")
    runtime.register_external_task(
        "conversation-waiting",
        {
            "task_id": "task-1",
            "kind": "data_cleaning",
            "run_id": "run-3",
            "status": "running",
            "output_dir": "/outputs/run-3",
            "submitted_at": "2026-09-01T00:00:00+00:00",
            "updated_at": "2026-09-01T00:00:00+00:00",
            "resume_pending": False,
        },
    )
    adapter.emit(
        "conversation-waiting",
        "run.finished",
        {"status": "idle"},
        run_id="run-3",
        event_id="run-3:finished",
    )

    await asyncio.sleep(1.3)
    assert "conversation-waiting" in runtime._active
    assert adapter.closed == []
    await runtime.close()


async def test_capacity_limit_releases_least_recent_conversation(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, max_active_conversations=2)
    for conversation_id in ("conversation-a", "conversation-b"):
        await _create(runtime, conversations, conversation_id)
        await asyncio.sleep(0.01)

    await _create(runtime, conversations, "conversation-c")

    assert set(runtime._active) == {"conversation-b", "conversation-c"}
    assert adapter.closed == ["conversation-a"]
    await runtime.close()


async def test_capacity_limit_rejects_new_conversation_when_all_busy(
    tmp_path,
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, max_active_conversations=1)
    await _create(runtime, conversations, "conversation-busy")
    store = FileProductStore(conversations / "conversation-busy")
    adapter.emit(
        "conversation-busy",
        "status.changed",
        {"status": "running"},
        event_id="status-running",
    )
    await _wait_for_status(store, "running")

    with pytest.raises(ProductRuntimeError) as error:
        await _create(runtime, conversations, "conversation-new")

    assert error.value.code == "capacity_exceeded"
    assert set(runtime._active) == {"conversation-busy"}
    await runtime.close()


async def test_subscriber_queue_sheds_deltas_but_keeps_state_events() -> None:
    runtime = ConversationRuntime("/tmp/unused", FakeAdapter())
    queue: asyncio.Queue[ProductEvent] = asyncio.Queue(3)  # type: ignore[assignment]
    runtime._subscribers["c1"] = {queue}

    def delta(index: int) -> ProductEvent:
        return ProductEvent(
            event_id=f"d{index}",
            conversation_id="c1",
            seq=index,
            type="message.delta",
            payload={},
        )

    for index in range(3):
        runtime._publish(delta(index))
    assert queue.full()

    runtime._publish(delta(99))
    assert queue.qsize() == 3

    state_event = ProductEvent(
        event_id="s1",
        conversation_id="c1",
        seq=100,
        type="status.changed",
        payload={},
    )
    runtime._publish(state_event)
    assert queue.qsize() == 3
    assert queue.get_nowait().seq == 1
    assert queue.get_nowait().seq == 2
    assert queue.get_nowait().seq == 100
