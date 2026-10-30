from __future__ import annotations

import asyncio
import json
from pathlib import Path

import harness_adapter.pi_adapter.adapter as pi_adapter_module
import httpx
from fastapi import FastAPI
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.api.router import _resolve_cursor, _sse_stream
from pyromind_agent_server.app import create_app
from pyromind_agent_server.bootstrap import install_product_api
from pyromind_agent_server.profiles import (
    ConversationProfile,
)
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec

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


def test_composed_app_mounts_echomind_project_routes(tmp_path) -> None:
    app = create_app(
        Config(
            conversations_path=tmp_path / "conversations",
            workspace_path=tmp_path / "workspace",
            enable_session_api_key_auth=False,
            enable_pyromind_jwt_auth=False,
        )
    )
    paths = set(app.openapi()["paths"])

    assert "/api/v2/echomind/conversations" in paths
    assert "/api/echomind/projects/{project_id}" in paths
    assert "/api/echomind/conversations" not in paths
    assert "/api/echomind/projects" not in paths
    assert not any("project-request" in path for path in paths)


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


async def test_product_profiles_isolate_conversations_with_the_same_profile_id(
    tmp_path,
) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    runtime = ConversationRuntime(conversations, FakeAdapter())
    first_profile = ConversationProfile(
        route_prefix="/api/first/conversations",
        profile_id="shared-profile",
        conversation_marker="first",
    )
    second_profile = ConversationProfile(
        route_prefix="/api/second/conversations",
        profile_id="shared-profile",
        conversation_marker="second",
    )
    app = _app(tmp_path, runtime)
    from pyromind_agent_server.api.router import create_product_router

    app.include_router(create_product_router(first_profile))
    app.include_router(create_product_router(second_profile))
    transport = httpx.ASGITransport(app=app)
    headers = {"x-pyromind-debug-user-id": "42"}
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as client:
        first = await client.post(
            "/api/first/conversations",
            json={"llm": {"model": "test-model"}},
        )
        second = await client.post(
            "/api/second/conversations",
            json={"llm": {"model": "test-model"}},
        )
        first_id = first.json()["conversation_id"]
        second_id = second.json()["conversation_id"]
        for conversation_id in (first_id, second_id):
            metadata_path = conversations / conversation_id / "product" / "meta.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["profile_id"] = None
            metadata_path.write_text(json.dumps(metadata))

        assert [
            item["conversation_id"]
            for item in (await client.get("/api/first/conversations")).json()
        ] == [first_id]
        assert [
            item["conversation_id"]
            for item in (await client.get("/api/second/conversations")).json()
        ] == [second_id]
        assert (await client.get("/api/v2/pyromind/conversations")).json() == []
        assert (
            await client.get(f"/api/first/conversations/{second_id}/snapshot")
        ).status_code == 404
        assert (
            await client.get(f"/api/second/conversations/{first_id}/snapshot")
        ).status_code == 404

    await runtime.close()


async def test_http_workflow_execution_prepares_confirmation_without_submitting(
    tmp_path,
) -> None:
    class Submission:
        def __init__(self) -> None:
            self.submitted = False

        async def prepare(self, workflow_path, context):
            return {
                "workflow": {"name": "sft", "nodes": []},
                "fingerprint": "abcdef1234567890",
                "summary": {"name": "sft", "node_count": 0},
            }

        async def submit(self, workflow, conversation_id, context):
            self.submitted = True
            return "task-1"

    conversations = tmp_path / "conversations"
    conversations.mkdir()
    submission = Submission()
    runtime = ConversationRuntime(
        conversations, FakeAdapter(), workflow_submission=submission
    )
    context = RequestContext(user_id="42")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="workflow-api",
            user_id="42",
            workspace_root=str(conversations / "workflow-api"),
        ),
        context,
    )
    transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v2/pyromind/conversations/workflow-api/workflow-executions",
            headers={"x-pyromind-debug-user-id": "42"},
            json={"workflow_path": "storage/project/workflow.py", "request_id": "r"},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "awaiting_confirmation"
    assert submission.submitted is False
    assert (
        len((await runtime.get_snapshot("workflow-api", context)).pending_permissions)
        == 1
    )
    await runtime.close()


async def test_create_conversation_reports_capacity_exceeded_as_429(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()
    runtime = ConversationRuntime(conversations, adapter, max_active_conversations=1)
    transport = httpx.ASGITransport(app=_app(tmp_path, runtime))
    headers = {"x-pyromind-debug-user-id": "42"}
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=headers
    ) as client:
        created = await client.post(
            "/api/v2/pyromind/conversations", json={"llm": {"model": "test-model"}}
        )
        assert created.status_code == 201
        runtime.register_external_task(
            created.json()["conversation_id"],
            {
                "task_id": "task-1",
                "kind": "data_cleaning",
                "run_id": "run-1",
                "status": "running",
                "output_dir": "/outputs/run-1",
                "submitted_at": "2026-09-01T00:00:00+00:00",
                "updated_at": "2026-09-01T00:00:00+00:00",
                "resume_pending": False,
            },
        )

        rejected = await client.post(
            "/api/v2/pyromind/conversations", json={"llm": {"model": "test-model"}}
        )

    assert rejected.status_code == 429
    assert rejected.json()["detail"]["code"] == "capacity_exceeded"
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
    assert first == ": connected\n\n"
    event_frame = await asyncio.wait_for(anext(stream), timeout=1)
    assert "id: 1\n" in event_frame
    assert "event: conversation.created\n" in event_frame
    assert '"seq":1' in event_frame
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


async def test_product_api_creates_pi_metadata_and_reports_missing_checkpoint(
    tmp_path, monkeypatch
) -> None:
    class FakeRunner:
        def __init__(self, **_kwargs) -> None:
            self.running = True

        async def start(self, _config) -> None:
            return None

        async def request(self, method, params):
            if method == "context.append":
                return {"checkpoint_entry_id": "workflow-checkpoint"}
            if method == "fork":
                target = Path(params["target_session_dir"]) / "branched.jsonl"
                target.write_text('{"type":"session"}\n', encoding="utf-8")
                return {"session_path": str(target)}
            raise AssertionError(f"unexpected runner request: {method}")

        async def close(self) -> None:
            self.running = False

    monkeypatch.setattr(pi_adapter_module, "PiRunnerProcess", FakeRunner)
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    runtime = ConversationRuntime(
        conversations,
        {
            "openhands": FakeAdapter(),
            "pi": PiAdapter(conversations, terminal_backend="os-sandbox"),
        },
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
                    "nodes": [
                        {"id": "n1", "data": {"nodeType": "CloneAndCacheDataset"}}
                    ],
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
        assert created.json()["current_workflow"] is None
        adapter = runtime.adapters["pi"]
        assert isinstance(adapter, PiAdapter)
        session = adapter._session(conversation_id)
        frame = {
            "protocolVersion": 2,
            "type": "pi.event",
            "sessionId": conversation_id,
            "runId": "edited-run",
            "eventId": "edit-final",
            "kind": "tool.completed",
            "payload": {
                "tool_call_id": "edit",
                "tool_name": "edit",
                "arguments": {"path": "public_data/workflow_canvas/workflow.py"},
                "content": [],
            },
        }
        await adapter._runner_event(session, frame)
        await adapter._runner_event(
            session,
            {
                **frame,
                "eventId": "finish-final",
                "kind": "run.finished",
                "payload": {
                    "outcome": {"status": "completed"},
                    "checkpoint_entry_id": "workflow-checkpoint",
                },
            },
        )
        async with asyncio.timeout(2):
            while (
                not FileProductStore(conversations / conversation_id)
                .load_snapshot()
                .current_workflow
            ):
                await asyncio.sleep(0.01)
        workflow_event = next(
            event
            for event in FileProductStore(conversations / conversation_id).replay()
            if event.type == "workflow.updated"
        )
        valid_fork = await client.post(
            f"/api/v2/pyromind/conversations/{conversation_id}/forks",
            json={"eventId": workflow_event.event_id, "title": "Pi fork"},
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
    workflow_text = workflow.read_text()
    assert workflow_text.startswith("# workflow: Workflow")
    assert 'CloneAndCacheDataset(id="n1")' in workflow_text
    canvas = workflow_event.payload["canvas"]
    assert isinstance(canvas, dict) and canvas["nodes"]
    assert forked.status_code == 404
    assert forked.json()["detail"]["code"] == "checkpoint_not_found"
    assert valid_fork.status_code == 201
    target_id = valid_fork.json()["conversation_id"]
    assert target_id != conversation_id
    assert (
        valid_fork.json()["current_workflow"]["version"]
        == workflow_event.payload["version"]
    )
    assert (conversations / target_id / "pi" / "session.jsonl").is_file()
    await runtime.close()
