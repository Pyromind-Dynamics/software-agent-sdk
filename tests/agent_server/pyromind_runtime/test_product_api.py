from __future__ import annotations

import asyncio

import httpx
import pyromind_runtime.product.router as product_router
import pytest
from pyromind_runtime.adapters.pi import SandboxGatewayError
from pyromind_runtime.contracts import (
    HarnessEvent,
    ProductEvent,
    TextContentBlock,
    ToolSpec,
    UserMessageCommand,
    WorkspaceRef,
)
from pyromind_runtime.projectors import SnapshotProjectionError
from pyromind_runtime.product import (
    CapabilityNotSupportedError,
    ConversationNotFoundError,
    HarnessRegistry,
    ProductConversationNotActiveError,
    ProductRuntimeService,
    ProductRuntimeSettings,
)
from pyromind_runtime.product.router import _resolve_cursor, _sse_stream
from pyromind_runtime.tool_host import (
    SessionToolContextStore,
    ToolRequestContext,
    ToolRequestContextNotAvailableError,
)

from openhands.agent_server.api import create_app
from openhands.agent_server.config import Config


async def test_runtime_persists_harness_events_and_replays_without_gap(
    product_runtime,
    fake_harness,
) -> None:
    snapshot = await product_runtime.create_conversation(user_id="user-1")
    conversation_id = snapshot.conversation_id
    events = product_runtime.stream_events(
        conversation_id,
        after_seq=1,
        user_id="user-1",
    )
    next_event = asyncio.ensure_future(anext(events))

    fake_harness.emit(
        conversation_id,
        HarnessEvent(
            session_id=conversation_id,
            type="run.started",
            run_id="run-1",
        ),
    )

    persisted = await asyncio.wait_for(next_event, timeout=1)
    assert persisted.seq == 2
    assert persisted.type == "run.started"
    assert (
        await product_runtime.get_snapshot(conversation_id, user_id="user-1")
    ).status == "running"

    await events.aclose()
    await product_runtime.close()


async def test_http_commands_and_snapshot_use_product_contracts(
    tmp_path,
    product_runtime,
    fake_harness,
) -> None:
    app = create_app(
        Config(
            conversations_path=tmp_path / "legacy-conversations",
            workspace_path=tmp_path / "workspace",
            enable_session_api_key_auth=False,
            enable_pyromind_jwt_auth=False,
        )
    )
    app.state.product_runtime = product_runtime
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        created = await client.post("/api/v2/pyromind/conversations", json={})
        assert created.status_code == 201
        conversation_id = created.json()["conversation_id"]

        command = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/commands",
            json={
                "command_id": "command-1",
                "type": "user_message",
                "content": [{"type": "text", "text": "hello"}],
            },
        )
        retry = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/commands",
            json={
                "command_id": "command-1",
                "type": "user_message",
                "content": [{"type": "text", "text": "hello"}],
            },
        )
        snapshot = await client.get(
            f"/api/v2/pyromind/conversations/{conversation_id}/snapshot"
        )
        conversations = await client.get("/api/v2/pyromind/conversations")

    assert command.status_code == 202
    assert command.json()["status"] == "completed"
    assert retry.json() == command.json()
    assert len(fake_harness.sent) == 1
    assert snapshot.status_code == 200
    assert snapshot.json()["through_seq"] == 1
    assert len(conversations.json()) == 1
    await product_runtime.close()


async def test_http_create_maps_sandbox_auth_failure_without_asgi_error(
    tmp_path,
    product_runtime,
    monkeypatch,
) -> None:
    app = create_app(
        Config(
            conversations_path=tmp_path / "legacy-conversations",
            workspace_path=tmp_path / "workspace",
            enable_session_api_key_auth=False,
            enable_pyromind_jwt_auth=False,
        )
    )
    app.state.product_runtime = product_runtime

    async def fail_create(**_kwargs: object) -> None:
        raise SandboxGatewayError(
            "permission_denied",
            "Sandbox authentication is unavailable",
        )

    monkeypatch.setattr(product_runtime, "create_conversation", fail_create)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v2/pyromind/conversations", json={})

    assert response.status_code == 401
    assert response.json() == {"detail": "Sandbox authentication is unavailable"}
    await product_runtime.close()


async def test_http_fork_uses_product_capability_and_returns_snapshot(
    tmp_path,
    product_runtime,
    fake_harness,
) -> None:
    fake_harness.capabilities = fake_harness.capabilities.model_copy(
        update={"fork": True}
    )
    app = create_app(
        Config(
            conversations_path=tmp_path / "legacy-conversations",
            workspace_path=tmp_path / "workspace",
            enable_session_api_key_auth=False,
            enable_pyromind_jwt_auth=False,
        )
    )
    app.state.product_runtime = product_runtime
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/v2/pyromind/conversations", json={})
        conversation_id = created.json()["conversation_id"]
        forked = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/forks",
            json={"after_seq": 1},
        )

    assert forked.status_code == 201
    assert forked.json()["conversation_id"] != conversation_id
    assert forked.json()["capabilities"]["fork"] is True
    assert fake_harness.forked == [(conversation_id, forked.json()["conversation_id"])]
    await product_runtime.close()


async def test_sse_replays_sequence_as_event_id(product_runtime) -> None:
    snapshot = await product_runtime.create_conversation(user_id="user-1")
    stream = _sse_stream(
        product_runtime,
        snapshot.conversation_id,
        after_seq=0,
        user_id="user-1",
    )

    first = await asyncio.wait_for(anext(stream), timeout=1)

    assert "id: 1\n" in first
    assert "event: conversation.created\n" in first
    assert '"seq":1' in first
    await stream.aclose()
    await product_runtime.close()


async def test_sse_sends_heartbeat_while_waiting_for_live_events(
    product_runtime,
    monkeypatch,
) -> None:
    snapshot = await product_runtime.create_conversation(user_id="user-1")
    monkeypatch.setattr(product_router, "_SSE_HEARTBEAT_SECONDS", 0.01)
    stream = _sse_stream(
        product_runtime,
        snapshot.conversation_id,
        after_seq=snapshot.through_seq,
        user_id="user-1",
    )

    heartbeat = await asyncio.wait_for(anext(stream), timeout=1)

    assert heartbeat == ": heartbeat\n\n"
    await stream.aclose()
    await product_runtime.close()


async def test_product_conversations_are_scoped_to_the_owner(product_runtime) -> None:
    user_one = await product_runtime.create_conversation(user_id="user-1")
    await product_runtime.create_conversation(user_id="user-2")

    visible = await product_runtime.list_conversations("user-1")

    assert [item.conversation_id for item in visible] == [user_one.conversation_id]
    with pytest.raises(ConversationNotFoundError):
        await product_runtime.get_snapshot(user_one.conversation_id, "user-2")
    await product_runtime.close()


async def test_list_conversations_skips_snapshot_projection_errors(
    product_runtime,
) -> None:
    visible = await product_runtime.create_conversation(user_id="user-1")
    invalid = await product_runtime.create_conversation(user_id="user-1")
    store = product_runtime._store(invalid.conversation_id)

    with pytest.raises(SnapshotProjectionError):
        store.append(
            ProductEvent(
                conversation_id=invalid.conversation_id,
                type="operation.completed",
                payload={"operation_id": "missing", "details": "undefined"},
            ),
            product_runtime.snapshot_projector.reduce,
        )

    snapshots = await product_runtime.list_conversations("user-1")

    assert [item.conversation_id for item in snapshots] == [visible.conversation_id]
    await product_runtime.close()


async def test_resource_event_projects_authoritative_workflow(
    tmp_path,
    product_runtime,
    fake_harness,
) -> None:
    workspace = tmp_path / "workflow-workspace"
    workflow_dir = workspace / "public_data" / "workflow_canvas"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "workflow.py").write_text("workflow dsl", encoding="utf-8")
    snapshot = await product_runtime.create_conversation(
        user_id="user-1",
        workspace=WorkspaceRef(workspace_id="workspace-1", root=str(workspace)),
    )

    fake_harness.emit(
        snapshot.conversation_id,
        HarnessEvent(
            session_id=snapshot.conversation_id,
            type="resource.updated",
            payload={
                "resource_type": "workflow",
                "resource_id": "workflow-1",
                "version": "version-1",
            },
        ),
    )

    projected = await product_runtime.get_snapshot(
        snapshot.conversation_id,
        "user-1",
    )
    for _ in range(20):
        if projected.workflow is not None:
            break
        await asyncio.sleep(0)
        projected = await product_runtime.get_snapshot(
            snapshot.conversation_id,
            "user-1",
        )
    assert projected.workflow is not None
    assert projected.workflow.dsl == "workflow dsl"
    assert projected.workflow.version == "version-1"
    await product_runtime.close()


def test_sse_cursor_prefers_after_seq_and_validates_last_event_id() -> None:
    assert _resolve_cursor(4, "2") == 4
    assert _resolve_cursor(None, "2") == 2


async def test_create_app_registers_pi_only_when_explicitly_enabled(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYROMIND_DEFAULT_HARNESS", "pi")
    monkeypatch.setenv("PYROMIND_PI_MODEL_API_KEY", "model-token")
    monkeypatch.setenv("PYROMIND_PI_MODEL_ID", "gpt-5.5")
    app = create_app(
        Config(
            conversations_path=tmp_path / "legacy-conversations",
            workspace_path=tmp_path / "workspace",
            enable_session_api_key_auth=False,
            enable_pyromind_jwt_auth=False,
        )
    )

    _, descriptor = await app.state.product_runtime.registry.resolve("pi")

    assert app.state.product_runtime.settings.default_harness_id == "pi"
    assert descriptor.harness_id == "pi"
    assert descriptor.capabilities.custom_tools is True
    assert descriptor.capabilities.native_workspace_tools == frozenset(
        {"read", "write", "edit", "bash"}
    )
    await app.state.product_runtime.close()


async def test_enable_pi_selects_it_when_default_is_not_explicit(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("PYROMIND_DEFAULT_HARNESS", raising=False)
    monkeypatch.setenv("PYROMIND_ENABLE_PI", "true")
    monkeypatch.setenv("PYROMIND_PI_MODEL_API_KEY", "model-token")
    app = create_app(
        Config(
            conversations_path=tmp_path / "legacy-conversations",
            workspace_path=tmp_path / "workspace",
            enable_session_api_key_auth=False,
            enable_pyromind_jwt_auth=False,
        )
    )

    assert app.state.product_runtime.settings.default_harness_id == "pi"
    await app.state.product_runtime.close()


async def test_runtime_selects_default_tools_per_harness_and_refreshes_context(
    tmp_path,
    fake_harness,
) -> None:
    tool = ToolSpec(
        name="test_tool",
        description="A neutral test tool.",
        input_schema={"type": "object"},
        timeout_seconds=1,
        risk_level="low",
    )
    context_store = SessionToolContextStore(allowed_header_names={"cookie"})
    registry = HarnessRegistry()
    registry.register("fake", fake_harness)
    runtime = ProductRuntimeService(
        ProductRuntimeSettings(
            storage_root=tmp_path / "product-conversations",
            default_harness_id="fake",
            default_workspace_root=str(tmp_path / "workspace"),
        ),
        registry,
        default_tools_by_harness={"fake": (tool,)},
        tool_context_store=context_store,
    )

    snapshot = await runtime.create_conversation(
        user_id="user-1",
        tool_context=ToolRequestContext(headers={"cookie": "first"}),
    )
    assert fake_harness.created_specs[0].tools == (tool,)
    assert context_store.get(snapshot.conversation_id).headers["cookie"] == "first"

    await runtime.submit_command(
        snapshot.conversation_id,
        UserMessageCommand(
            command_id="command-1",
            content=(TextContentBlock(text="hello"),),
        ),
        "user-1",
        tool_context=ToolRequestContext(headers={"cookie": "refreshed"}),
    )
    assert context_store.get(snapshot.conversation_id).headers["cookie"] == "refreshed"

    await runtime.close()
    with pytest.raises(ToolRequestContextNotAvailableError):
        context_store.get(snapshot.conversation_id)


async def test_inactive_resumable_conversation_uses_persisted_harness(
    tmp_path,
    fake_harness,
) -> None:
    fake_harness.capabilities = fake_harness.capabilities.model_copy(
        update={"resume": True}
    )
    registry = HarnessRegistry()
    registry.register("fake", fake_harness)
    settings = ProductRuntimeSettings(
        storage_root=tmp_path / "product-conversations",
        default_harness_id="fake",
        default_workspace_root=str(tmp_path / "workspace"),
    )
    first_runtime = ProductRuntimeService(settings, registry)
    snapshot = await first_runtime.create_conversation(user_id="user-1")
    await first_runtime.close()

    restarted_runtime = ProductRuntimeService(
        ProductRuntimeSettings(
            storage_root=settings.storage_root,
            default_harness_id="a-different-new-session-default",
            default_workspace_root=settings.default_workspace_root,
        ),
        registry,
    )
    receipt = await restarted_runtime.submit_command(
        snapshot.conversation_id,
        UserMessageCommand(
            command_id="command-after-restart",
            content=(TextContentBlock(text="resume"),),
        ),
        "user-1",
    )

    assert receipt.status == "completed"
    assert len(fake_harness.created_specs) == 2
    assert fake_harness.created_specs[-1].required_capabilities == frozenset({"resume"})
    assert fake_harness.sent[-1][0] == snapshot.conversation_id
    await restarted_runtime.close()


async def test_inactive_non_resumable_conversation_fails_without_recreation(
    tmp_path,
    fake_harness,
) -> None:
    registry = HarnessRegistry()
    registry.register("fake", fake_harness)
    settings = ProductRuntimeSettings(
        storage_root=tmp_path / "product-conversations",
        default_harness_id="fake",
        default_workspace_root=str(tmp_path / "workspace"),
    )
    first_runtime = ProductRuntimeService(settings, registry)
    snapshot = await first_runtime.create_conversation(user_id="user-1")
    await first_runtime.close()
    restarted_runtime = ProductRuntimeService(settings, registry)

    with pytest.raises(
        ProductConversationNotActiveError, match="does not support resume"
    ):
        await restarted_runtime.submit_command(
            snapshot.conversation_id,
            UserMessageCommand(
                command_id="must-not-run",
                content=(TextContentBlock(text="hello"),),
            ),
            "user-1",
        )

    assert len(fake_harness.created_specs) == 1
    assert not fake_harness.sent
    await restarted_runtime.close()


async def test_product_fork_keeps_neutral_history_and_source_harness(
    product_runtime,
    fake_harness,
) -> None:
    fake_harness.capabilities = fake_harness.capabilities.model_copy(
        update={"fork": True}
    )
    source = await product_runtime.create_conversation(user_id="user-1")
    fake_harness.emit(
        source.conversation_id,
        HarnessEvent(
            session_id=source.conversation_id,
            type="message.started",
            run_id="run-1",
            payload={
                "message_id": "message-1",
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            },
        ),
    )
    fake_harness.emit(
        source.conversation_id,
        HarnessEvent(
            session_id=source.conversation_id,
            type="message.completed",
            run_id="run-1",
            payload={
                "message_id": "message-1",
                "content": [{"type": "text", "text": "hello"}],
            },
        ),
    )
    for _ in range(20):
        current = await product_runtime.get_snapshot(source.conversation_id, "user-1")
        if current.through_seq == 3:
            break
        await asyncio.sleep(0)

    fork = await product_runtime.fork_conversation(
        source.conversation_id,
        "user-1",
        after_seq=3,
    )

    assert fork.conversation_id != source.conversation_id
    assert fork.through_seq == 3
    assert [message.message_id for message in fork.messages] == ["message-1"]
    assert fake_harness.forked == [(source.conversation_id, fork.conversation_id)]
    assert fake_harness.created_specs[-1].required_capabilities == frozenset({"fork"})
    await product_runtime.close()


async def test_product_fork_rejects_unsupported_or_historical_forks(
    product_runtime,
    fake_harness,
) -> None:
    source = await product_runtime.create_conversation(user_id="user-1")
    with pytest.raises(CapabilityNotSupportedError, match="does not support fork"):
        await product_runtime.fork_conversation(source.conversation_id, "user-1")

    fake_harness.capabilities = fake_harness.capabilities.model_copy(
        update={"fork": True}
    )
    capable_source = await product_runtime.create_conversation(user_id="user-1")
    with pytest.raises(CapabilityNotSupportedError, match="historical forks"):
        await product_runtime.fork_conversation(
            capable_source.conversation_id,
            "user-1",
            after_seq=0,
        )
    await product_runtime.close()
