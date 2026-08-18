from __future__ import annotations

import asyncio

from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
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
