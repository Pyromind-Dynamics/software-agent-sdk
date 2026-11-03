from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import FastAPI
from pyromind_agent_server.api.router import create_product_router
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities

from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import ConversationService

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
    app.include_router(create_product_router())
    return app


async def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


async def _create_with_pending_input(client, adapter, runtime):
    response = await client.post(
        "/api/v2/pyromind/conversations",
        headers={"x-pyromind-debug-user-id": "42"},
        json={
            "llm": {"model": "gpt-5", "api_key": "test-key"},
            "message": "hello",
            "user_input_enabled": True,
        },
    )
    assert response.status_code == 201
    conversation_id = response.json()["conversation_id"]
    adapter.emit(
        conversation_id,
        "input.requested",
        {
            "request_id": "input-1",
            "operation_id": "call-1",
            "questions": QUESTIONS,
        },
        run_id="run-1",
    )
    store = runtime._store(conversation_id)
    await _wait_for(lambda: store.load_snapshot().pending_inputs)
    return conversation_id, store


async def test_create_conversation_accepts_user_input_enabled(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = _input_adapter()
    runtime = ConversationRuntime(conversations, adapter)
    app = _app(tmp_path, runtime)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            conversation_id, _ = await _create_with_pending_input(
                client, adapter, runtime
            )
        spec = adapter.created_specs[0]
        assert spec.user_input_enabled is True
        snapshot = runtime._store(conversation_id).load_snapshot()
        assert snapshot.capabilities.user_input_reply is True
        assert snapshot.pending_inputs[0].request_id == "input-1"
    finally:
        await runtime.close()


async def test_input_response_command_answers_over_http(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = _input_adapter()
    runtime = ConversationRuntime(conversations, adapter)
    app = _app(tmp_path, runtime)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"x-pyromind-debug-user-id": "42"},
        ) as client:
            conversation_id, store = await _create_with_pending_input(
                client, adapter, runtime
            )
            response = await client.post(
                f"/api/v2/pyromind/conversations/{conversation_id}/commands",
                json={
                    "command_id": "answer-1",
                    "type": "input_response",
                    "request_id": "input-1",
                    "status": "answered",
                    "answers": [
                        {
                            "question_id": "duplicate_policy",
                            "option_id": "keep_first",
                        },
                        {"question_id": "output_name", "text": "cleaned.csv"},
                    ],
                },
            )
            assert response.status_code == 202
            assert response.json()["status"] == "completed"
            snapshot = store.load_snapshot()
            assert snapshot.pending_inputs == ()
            assert snapshot.status == "running"

            # Invalid answer: return 422 and keep the question pending.
            adapter.emit(
                conversation_id,
                "input.requested",
                {
                    "request_id": "input-2",
                    "operation_id": "call-2",
                    "questions": QUESTIONS,
                },
                run_id="run-1",
                event_id="input-2:requested",
            )
            await _wait_for(
                lambda: (
                    store.load_snapshot().pending_inputs
                    and store.load_snapshot().pending_inputs[0].request_id == "input-2"
                )
            )
            invalid = await client.post(
                f"/api/v2/pyromind/conversations/{conversation_id}/commands",
                json={
                    "command_id": "answer-2",
                    "type": "input_response",
                    "request_id": "input-2",
                    "status": "answered",
                    "answers": [
                        {"question_id": "duplicate_policy", "option_id": "nope"},
                    ],
                },
            )
            assert invalid.status_code == 422
            snapshot = store.load_snapshot()
            assert snapshot.pending_inputs[0].request_id == "input-2"

            # Reject ordinary messages during a wait with 409 input_pending.
            rejected = await client.post(
                f"/api/v2/pyromind/conversations/{conversation_id}/commands",
                json={
                    "command_id": "msg-1",
                    "type": "user_message",
                    "content": [{"type": "text", "text": "hi"}],
                },
            )
            assert rejected.status_code == 409
            assert rejected.json()["detail"]["code"] == "input_pending"

            # Answering a nonexistent request returns 409 input_not_pending.
            missing = await client.post(
                f"/api/v2/pyromind/conversations/{conversation_id}/commands",
                json={
                    "command_id": "answer-3",
                    "type": "input_response",
                    "request_id": "input-missing",
                    "status": "cancelled",
                },
            )
            assert missing.status_code == 409
            assert missing.json()["detail"]["code"] == "input_not_pending"

            # Different content with the same command_id returns 409 command_conflict.
            conflict = await client.post(
                f"/api/v2/pyromind/conversations/{conversation_id}/commands",
                json={
                    "command_id": "answer-1",
                    "type": "input_response",
                    "request_id": "input-1",
                    "status": "cancelled",
                },
            )
            assert conflict.status_code == 409
            assert conflict.json()["detail"]["code"] == "command_conflict"

            # Cancelling the request completes its receipt and suspends execution.
            cancel = await client.post(
                f"/api/v2/pyromind/conversations/{conversation_id}/commands",
                json={
                    "command_id": "cancel-2",
                    "type": "input_response",
                    "request_id": "input-2",
                    "status": "cancelled",
                },
            )
            assert cancel.status_code == 202
            snapshot = store.load_snapshot()
            assert snapshot.pending_inputs == ()
            assert snapshot.automatic_turns_suspended is True
            assert snapshot.status == "paused"

            # A user message explicitly resumes execution.
            resumed = await client.post(
                f"/api/v2/pyromind/conversations/{conversation_id}/commands",
                json={
                    "command_id": "msg-2",
                    "type": "user_message",
                    "content": [{"type": "text", "text": "继续"}],
                },
            )
            assert resumed.status_code == 202
            snapshot = store.load_snapshot()
            assert snapshot.automatic_turns_suspended is False
    finally:
        await runtime.close()


async def test_create_rejects_input_enabled_on_unsupported_harness(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter()  # user_input_reply 默认 False
    runtime = ConversationRuntime(conversations, adapter)
    app = _app(tmp_path, runtime)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v2/pyromind/conversations",
                headers={"x-pyromind-debug-user-id": "42"},
                json={
                    "llm": {"model": "gpt-5", "api_key": "test-key"},
                    "user_input_enabled": True,
                },
            )
            assert response.status_code == 409
            assert response.json()["detail"]["code"] == "capability_not_supported"
    finally:
        await runtime.close()
