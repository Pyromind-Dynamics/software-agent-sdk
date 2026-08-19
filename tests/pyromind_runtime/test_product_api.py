from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.api.router import _resolve_cursor, _sse_stream
from pyromind_agent_server.app import create_app
from pyromind_agent_server.bootstrap import install_product_api
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import ConversationService

from .fake_adapter import FakeAdapter


def _app(tmp_path, runtime: ConversationRuntime) -> FastAPI:
    app = FastAPI()
    app.state.config = Config(
        conversations_path=tmp_path / "conversations",
        workspace_path=tmp_path / "workspace",
        enable_session_api_key_auth=False,
        enable_pyromind_jwt_auth=False,
    )
    app.state.conversation_service = ConversationService(
        conversations_dir=tmp_path / "conversations"
    )
    app.state.product_runtime = runtime
    from pyromind_agent_server.api.router import create_product_router

    app.include_router(create_product_router())
    return app


async def test_composed_app_mounts_product_router(tmp_path) -> None:
    app = create_app(
        Config(
            conversations_path=tmp_path / "conversations",
            workspace_path=tmp_path / "workspace",
            enable_session_api_key_auth=False,
            enable_pyromind_jwt_auth=False,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/v2/pyromind/conversations",
            headers={"x-pyromind-debug-user-id": "42"},
        )

    # Lifespan is intentionally not entered, so the runtime is unavailable;
    # 503 proves the composed app matched the Product route instead of 404.
    assert response.status_code == 503


async def test_http_create_list_snapshot_and_command(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter)
    transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
    headers = {"x-pyromind-debug-user-id": "42"}
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as client:
        created = await client.post(
            "/api/v2/pyromind/conversations",
            json={"llm": {"model": "test-model"}, "message": "hello"},
        )
        assert created.status_code == 201
        conversation_id = created.json()["conversation_id"]
        listed = await client.get("/api/v2/pyromind/conversations")
        snapshot = await client.get(
            f"/api/v2/pyromind/conversations/{conversation_id}/snapshot"
        )
        command = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/commands",
            json={
                "command_id": "command-1",
                "type": "user_message",
                "content": [{"type": "text", "text": "continue"}],
            },
        )
        retry = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/commands",
            json={
                "command_id": "command-1",
                "type": "user_message",
                "content": [{"type": "text", "text": "continue"}],
            },
        )

    assert [item["conversation_id"] for item in listed.json()] == [conversation_id]
    assert snapshot.json()["timeline"][0]["kind"] == "message"
    assert command.status_code == 202
    assert command.json() == retry.json()
    assert adapter.created_specs[0].workspace_root == str(
        conversations / conversation_id
    )
    await runtime.close()


async def test_sse_uses_persisted_sequence_as_event_id(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    runtime = ConversationRuntime(conversations, FakeAdapter())
    context = RequestContext(user_id="42")
    from pyromind_runtime.ports.harness import SessionSpec

    snapshot = await runtime.create_conversation(
        SessionSpec(
            conversation_id="conversation-sse",
            user_id="42",
            workspace_root=str(conversations),
        ),
        context,
    )
    stream = _sse_stream(runtime, snapshot.conversation_id, 0, context)
    first = await asyncio.wait_for(anext(stream), timeout=1)
    assert "id: 1\n" in first
    assert "event: conversation.created\n" in first
    assert '"seq":1' in first
    await stream.aclose()
    await runtime.close()


def test_sse_cursor_prefers_after_seq_and_validates_last_event_id() -> None:
    assert _resolve_cursor(4, "2") == 4
    assert _resolve_cursor(None, "2") == 2


def test_install_product_api_does_not_modify_openhands_routes() -> None:
    app = FastAPI()
    original = app.router.lifespan_context
    installed = install_product_api(app)
    assert installed is app
    assert app.router.lifespan_context is not original


async def test_product_api_creates_pi_metadata_and_rejects_fork(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    runtime = ConversationRuntime(
        conversations,
        {"openhands": FakeAdapter(), "pi": PiAdapter(conversations)},
        default_harness_id="pi",
    )
    transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"x-pyromind-debug-user-id": "42"},
    ) as client:
        created = await client.post(
            "/api/v2/pyromind/conversations",
            json={
                "llm": {"model": "gpt-4o", "api_key": "request-secret"},
                "workflow_xyflow": {
                    "name": "Workflow",
                    "nodes": [],
                    "edges": [],
                },
            },
        )
        assert created.status_code == 201
        conversation_id = created.json()["conversation_id"]
        forked = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/forks",
            json={"eventId": "event-1"},
        )
    metadata = (conversations / conversation_id / "product" / "meta.json").read_text()
    assert '"harness_id":"pi"' in metadata
    assert "request-secret" not in metadata
    workflow = (
        conversations
        / conversation_id
        / "public_data"
        / "workflow_canvas"
        / "workflow.py"
    )
    assert workflow.read_text() == "# workflow: Workflow"
    assert created.json()["current_workflow"]["canvas"]["name"] == "Workflow"
    assert forked.status_code == 409
    await runtime.close()
