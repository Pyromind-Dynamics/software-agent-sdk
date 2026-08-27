from __future__ import annotations

import asyncio

from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
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
    assert adapter.external_task_notifications[0][1]["status"] == "succeeded"
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
