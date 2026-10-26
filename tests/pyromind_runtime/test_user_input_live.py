from __future__ import annotations

import asyncio
import json
import os
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
from pyromind_runtime.ports.harness import SessionSpec


async def _wait_until(condition):
    async with asyncio.timeout(120):
        while not condition():
            await asyncio.sleep(0.05)


@pytest.mark.parametrize("action", ["answered", "cancelled", "stop", "restart"])
async def test_user_input_with_live_openrouter(
    tmp_path, monkeypatch, live_user_input_model, action
) -> None:
    base_url, requests, results = live_user_input_model
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setenv(
        "PYROMIND_PI_RUNTIME",
        str(repository / "harness-adapter/pi-runtime/dist/index.js"),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "local-proxy-placeholder")
    root = tmp_path / "conversations"
    root.mkdir()
    model = os.getenv("OPENROUTER_TEST_MODEL", "deepseek/deepseek-v4.1-flash")
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(root, adapter, default_harness_id="pi")
    context = RequestContext(user_id="live-test")
    report = {"model": model, "action": action, "passed": False}
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="live-input",
                user_id=context.user_id,
                workspace_root=str(root / "live-input"),
                user_input_enabled=True,
                model_configuration={
                    "model": model,
                    "base_url": base_url,
                    "api_key": "local-proxy-placeholder",
                },
            ),
            context,
        )
        store = runtime._store("live-input")
        await runtime.submit_command(
            "live-input",
            UserMessageCommand(
                command_id="start",
                content=(
                    TextContent(
                        text=(
                            "This is a structured user-input integration test. "
                            "Call ask_user exactly once with three questions: "
                            "id=format: choose csv or json, using option ids csv/json "
                            "with labels CSV/JSON; include a description for csv. "
                            "id=filename: free-text output filename, no options. "
                            "id=note: free-text note, no options. "
                            "Do not call any other tool or write files. Wait for the "
                            "tool result, then reply with the format, filename "
                            "and note verbatim. Do not ask again."
                        )
                    ),
                ),
            ),
            context,
        )
        await _wait_until(
            lambda: (
                store.load_snapshot().pending_inputs
                or store.load_snapshot().status == "error"
            )
        )
        snapshot = store.load_snapshot()
        assert snapshot.pending_inputs, [(r["status"]) for r in results]
        pending = snapshot.pending_inputs[0]
        assert pending.run_id == "start"
        assert {q.id for q in pending.questions} == {"format", "filename", "note"}
        assert len(requests) == 1
        schema = next(
            t["function"]["parameters"]
            for t in requests[0]["tools"]
            if t["function"]["name"] == "ask_user"
        )
        report["tool_schema"] = schema
        await asyncio.sleep(1)
        assert len(requests) == 1, "the human wait must not call the model again"
        with pytest.raises(ProductRuntimeError) as error:
            await runtime.submit_command(
                "live-input",
                UserMessageCommand(
                    command_id="blocked", content=(TextContent(text="hello"),)
                ),
                context,
            )
        assert error.value.code == "input_pending"
        if action in {"answered", "cancelled"}:
            command = InputResponseCommand.model_validate(
                {
                    "command_id": "response",
                    "request_id": pending.request_id,
                    "status": action,
                    **(
                        {
                            "answers": [
                                {"question_id": "format", "option_id": "csv"},
                                {"question_id": "filename", "text": "live-7f32.csv"},
                                {"question_id": "note", "text": "中文自由输入验证"},
                            ]
                        }
                        if action == "answered"
                        else {}
                    ),
                }
            )
            receipt = await runtime.submit_command("live-input", command, context)
            assert receipt.status == "completed"
            assert (
                await runtime.submit_command("live-input", command, context) == receipt
            )
        elif action == "stop":
            await runtime.submit_command(
                "live-input", CancelCommand(command_id="stop"), context
            )
        else:
            await runtime.close()
            adapter = PiAdapter(root, terminal_backend="os-sandbox")
            runtime = ConversationRuntime(root, adapter, default_harness_id="pi")
            recovered = await runtime.get_snapshot("live-input", context)
            assert not recovered.pending_inputs
            assert not adapter._sessions
        if action != "answered":
            await _wait_until(lambda: not store.load_snapshot().pending_inputs)
            assert store.load_snapshot().automatic_turns_suspended
            if action != "restart":
                await _wait_until(lambda: not adapter._sessions["live-input"].running)
            await asyncio.sleep(1)
            assert len(requests) == 1
            await runtime.submit_command(
                "live-input",
                UserMessageCommand(
                    command_id="resume",
                    content=(
                        TextContent(
                            text="The question ended. Reply only LIVE_RESUMED. "
                            "Do not ask again or call any tool."
                        ),
                    ),
                ),
                context,
            )
        await _wait_until(
            lambda: (
                len(requests) >= 2 and store.load_snapshot().status in {"idle", "error"}
            )
        )
        snapshot = store.load_snapshot()
        assert snapshot.status == "idle"
        assert len(requests) == 2
        assert not snapshot.pending_inputs
        assert not snapshot.automatic_turns_suspended
        tool_results = [m for m in requests[1]["messages"] if m["role"] == "tool"]
        assert len(tool_results) == 1
        if action == "answered":
            assert "live-7f32.csv" in json.dumps(tool_results)
            assert "中文自由输入验证" in json.dumps(tool_results, ensure_ascii=False)
        completed = [e for e in store.replay() if e.type == "message.completed"]
        assistant_text = json.dumps(
            [e.payload for e in completed if e.payload.get("role") == "assistant"],
            ensure_ascii=False,
        )
        assert (
            "live-7f32.csv" if action == "answered" else "LIVE_RESUMED"
        ) in assistant_text
        resolutions = [e for e in store.replay() if e.type == "input.resolved"]
        assert len(resolutions) == 1
        if action == "answered":
            assert all(e.run_id == "start" for e in completed)
        report.update(
            passed=True,
            resolution=resolutions[0].payload["outcome"],
            final_status=snapshot.status,
            assistant_output=assistant_text,
        )
    finally:
        await runtime.close()
        report["request_count"] = len(requests)
        report["http_statuses"] = [r["status"] for r in results]
        metadata = []
        for result in results:
            for line in "".join(result["chunks"]).splitlines():
                if line.startswith("data: {"):
                    chunk = json.loads(line[6:])
                    metadata.append(
                        {
                            k: chunk[k]
                            for k in ("id", "model", "provider", "usage")
                            if k in chunk
                        }
                    )
        report["response_metadata"] = [
            item
            for index, item in enumerate(metadata)
            if index == 0 or item != metadata[index - 1]
        ]
        output = repository / ".agent_tmp" / "user-input-live"
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{action}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
