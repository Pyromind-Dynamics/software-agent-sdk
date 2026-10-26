from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from harness_adapter.pi_adapter import PiAdapter
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
from pyromind_runtime.ports.harness import SessionSpec


async def _until(condition):
    async with asyncio.timeout(15):
        while not condition():
            await asyncio.sleep(0.01)


@pytest.mark.parametrize(
    "action",
    [
        "answered",
        "cancelled",
        "stop",
        "crash",
        "restart",
        "persistence_failure",
        "admission_race",
    ],
)
async def test_user_input_through_real_runtime_and_runner(
    tmp_path, monkeypatch, user_input_model, action
) -> None:
    base_url, requests = user_input_model
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setenv(
        "PYROMIND_PI_RUNTIME",
        str(repository / "harness-adapter/pi-runtime/dist/index.js"),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    root = tmp_path / "conversations"
    root.mkdir()
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(root, adapter, default_harness_id="pi")
    context = RequestContext(user_id="owner")
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="conversation",
                user_id="owner",
                workspace_root=str(root / "conversation"),
                user_input_enabled=True,
                model_configuration={
                    "model": "local-input-model",
                    "base_url": base_url,
                    "api_key": "local-test-key",
                },
            ),
            context,
        )
        store = runtime._store("conversation")
        session = adapter._sessions["conversation"]
        registration = asyncio.Event()
        release_registration = asyncio.Event()
        synced = []
        if action == "persistence_failure":
            append = FileProductStore.append

            def fail_input(self, event):
                if event.type == "input.requested":
                    raise OSError("injected input log failure")
                return append(self, event)

            monkeypatch.setattr(FileProductStore, "append", fail_input)
        if action == "admission_race":
            register = adapter._handle_input_request

            async def delay_registration(session, params):
                registration.set()
                await release_registration.wait()
                return await register(session, params)

            async def record_sync(*args):
                synced.append(True)

            monkeypatch.setattr(adapter, "_handle_input_request", delay_registration)
            monkeypatch.setattr(adapter, "_sync_xyflow", record_sync)
        await runtime.submit_command(
            "conversation",
            UserMessageCommand(
                command_id="start",
                content=(TextContent(text="Ask me about the output format."),),
            ),
            context,
        )
        if action == "persistence_failure":
            await _until(lambda: "conversation" not in runtime._active)
            assert "conversation" not in adapter._sessions
            assert not session.pending_user_inputs
            assert not store.load_snapshot().pending_inputs
            assert store.load_snapshot().status == "error"
            assert store.load_snapshot().automatic_turns_suspended
            assert len(requests) == 1
            return
        if action == "admission_race":
            async with asyncio.timeout(15):
                await registration.wait()
            assert not session.pending_user_inputs
            assert not store.load_snapshot().pending_inputs
            with pytest.raises(ProductRuntimeError) as error:
                await runtime.submit_command(
                    "conversation",
                    UserMessageCommand(
                        command_id="racing-message",
                        content=(TextContent(text="hi"),),
                        workflow_xyflow={"nodes": []},
                    ),
                    context,
                )
            assert error.value.code == "input_pending"
            assert synced == []
            release_registration.set()
        await _until(lambda: store.load_snapshot().pending_inputs)
        pending = store.load_snapshot().pending_inputs[0]
        assert pending.run_id == "start"
        assert len(requests) == 1
        assert any(
            tool["function"]["name"] == "ask_user" for tool in requests[0]["tools"]
        )
        if action in {"answered", "cancelled", "admission_race"}:
            receipt = await runtime.submit_command(
                "conversation",
                InputResponseCommand.model_validate(
                    {
                        "command_id": "response",
                        "request_id": pending.request_id,
                        "status": "answered" if action == "answered" else "cancelled",
                        **(
                            {"answers": [{"question_id": "format", "text": "csv"}]}
                            if action == "answered"
                            else {}
                        ),
                    }
                ),
                context,
            )
            assert receipt.status == "completed"
        elif action == "stop":
            await runtime.submit_command(
                "conversation", CancelCommand(command_id="stop"), context
            )
        elif action == "crash":
            assert session.runner is not None
            assert session.runner._process is not None
            session.runner._process.kill()
        else:
            await runtime.close()
            assert len(requests) == 1, (
                "closing an input wait must abort before returning a result"
            )
            adapter = PiAdapter(root, terminal_backend="os-sandbox")
            runtime = ConversationRuntime(root, adapter, default_harness_id="pi")
            recovered = await runtime.get_snapshot("conversation", context)
            assert not recovered.pending_inputs
            assert not adapter._sessions, "snapshot recovery must not start a runner"
        if action == "answered":
            await _until(lambda: store.load_snapshot().status == "idle")
            assert len(requests) == 2
            assert "csv" in json.dumps(
                [m for m in requests[1]["messages"] if m["role"] == "tool"]
            )
        else:
            await _until(
                lambda: (
                    not store.load_snapshot().pending_inputs
                    and store.load_snapshot().status in {"paused", "error"}
                )
            )
            assert store.load_snapshot().automatic_turns_suspended
            assert len(requests) == 1
            if action != "restart":
                await _until(
                    lambda: not session.pending_user_inputs and not session.running
                )
            await runtime.submit_command(
                "conversation",
                UserMessageCommand(
                    command_id="continue",
                    content=(TextContent(text="Continue."),),
                ),
                context,
            )
            await _until(
                lambda: len(requests) == 2 and store.load_snapshot().status == "idle"
            )
            assert not store.load_snapshot().automatic_turns_suspended
            if action in {"crash", "restart"}:
                results = [m for m in requests[1]["messages"] if m["role"] == "tool"]
                assert len(results) == 1, (
                    "recovery must preserve an existing native result"
                )
                assert "interrupted" in json.dumps(results) or (
                    action == "restart" and "aborted" in json.dumps(results)
                )
        assert (
            len([event for event in store.replay() if event.type == "input.resolved"])
            == 1
        )
    finally:
        await runtime.close()
