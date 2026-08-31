from __future__ import annotations

import asyncio

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
