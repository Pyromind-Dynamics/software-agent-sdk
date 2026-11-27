from __future__ import annotations

import asyncio
import json
import os
import shlex
from importlib.metadata import version

import pytest
from aiohttp import ClientSession, ClientTimeout, web
from aiohttp.test_utils import TestServer

from harness_adapter.pi_adapter import PiAdapter
from harness_adapter.pi_adapter.adapter import _PiSession
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from pyromind_agent_server.echomind_consolidation import TrainReaderProbe
from pyromind_common.tools.utils import pyromind_api_client
from pyromind_common.tools.workflow import task_submission
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import SessionSpec

from .test_native_preparation import stream


@pytest.fixture
def pi_stop_session(tmp_path):
    adapter = PiAdapter(tmp_path, terminal_backend="os-sandbox")
    session = _PiSession(
        "stop-conversation",
        tmp_path,
        PiSessionFiles(tmp_path),
        {},
        RequestContext(user_id="42"),
    )
    return adapter, session


@pytest.fixture
def sdk_terminal_run(tmp_path, monkeypatch):
    async def run(app: web.Application, script: str) -> str:
        key = "atomic-sdk-access-key"
        cluster = "cn-east-1#pre2"
        model_requests = []
        platform_requests = []

        @web.middleware
        async def capture(request, handler):
            if request.path.startswith("/api/v1/"):
                platform_requests.append(
                    (
                        request.headers.get("authorization"),
                        request.headers.get("x-cluster"),
                        request.headers.get("cookie"),
                    )
                )
            return await handler(request)

        async def model(request):
            model_requests.append(await request.json())
            return (
                stream(
                    "terminal",
                    {
                        "command": '"$PYROMIND_SDK_PYTHON" -c ' + shlex.quote(script),
                        "timeout": 30,
                    },
                )
                if len(model_requests) == 1
                else stream()
            )

        app.middlewares.append(capture)
        app.router.add_post("/v1/chat/completions", model)
        monkeypatch.setenv("LLM_API_KEY", "atomic-local-model-key")
        monkeypatch.setattr(task_submission, "get_api_key", lambda **kwargs: key)
        root = tmp_path / "sdk-conversations"
        runtime = ConversationRuntime(
            root,
            PiAdapter(root, terminal_backend="os-sandbox"),
            default_harness_id="pi",
        )
        context = RequestContext(
            user_id="42", cookie="auth_token=atomic-request-token", x_cluster=cluster
        )
        try:
            async with TestServer(app) as server:
                monkeypatch.setattr(
                    pyromind_api_client,
                    "_api_url",
                    lambda env: str(server.make_url("/api/v1")),
                )
                await runtime.create_conversation(
                    SessionSpec(
                        conversation_id="sdk-operations",
                        user_id="42",
                        workspace_root=str(root / "sdk-operations"),
                        model_configuration={
                            "model": "atomic-local-model",
                            "base_url": str(server.make_url("/v1")),
                            "api_key": "atomic-local-model-key",
                        },
                        extra={"sandbox_default_image": "registry.example/sdk-test:v1"},
                    ),
                    context,
                )
                await runtime.submit_command(
                    "sdk-operations",
                    UserMessageCommand(
                        command_id="sdk-operations",
                        content=(TextContent(text="Run the platform operations."),),
                    ),
                    context,
                )
                async with asyncio.timeout(40):
                    while True:
                        snapshot = await runtime.get_snapshot("sdk-operations", context)
                        if len(model_requests) >= 2 and snapshot.status != "running":
                            break
                        await asyncio.sleep(0.02)
                assert snapshot.external_tasks == ()
                assert platform_requests
                assert all(
                    headers == (f"Bearer {key}", cluster, None)
                    for headers in platform_requests
                )
                removed = {
                    "sandbox_create",
                    "sandbox_delete",
                    "sandbox_read_file",
                    "sandbox_write_file",
                    "sandbox_delete_file",
                    "df_stop_task",
                    "inspect_workflow_task",
                }
                for body in model_requests:
                    names = {tool["function"]["name"] for tool in body["tools"]}
                    assert not names & removed
                    assert "sandbox_terminal" in names
                assert key not in json.dumps(model_requests)
                assert "atomic-request-token" not in json.dumps(model_requests)
                result = json.dumps(
                    [
                        message
                        for message in model_requests[-1]["messages"]
                        if message["role"] == "tool"
                    ],
                    ensure_ascii=False,
                )
                assert "Command exited with code" not in result, result
        finally:
            await runtime.close()
        for path in root.rglob("*"):
            if path.is_file():
                persisted = path.read_bytes()
                assert key.encode() not in persisted, str(path)
                assert b"atomic-request-token" not in persisted, str(path)
        return result

    return run


@pytest.fixture
async def user_input_model():
    """Local OpenAI stream for the real Python -> JSONL -> Pi input flow."""
    requests = []

    async def respond(reader, writer):
        headers = (await reader.readuntil(b"\r\n\r\n")).decode()
        length = next(
            int(line.split(":", 1)[1])
            for line in headers.split("\r\n")
            if line.lower().startswith("content-length:")
        )
        requests.append(json.loads(await reader.readexactly(length)))
        if len(requests) == 1:
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-input",
                        "type": "function",
                        "function": {
                            "name": "ask_user",
                            "arguments": json.dumps(
                                {
                                    "questions": [
                                        {
                                            "id": "format",
                                            "prompt": "Output format?",
                                            "options": [],
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ],
            }
            reason = "tool_calls"
        else:
            delta = {"role": "assistant", "content": "Continuing with your input."}
            reason = "stop"
        chunks = [
            {
                "id": "local",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local-input-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": "local",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local-input-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            },
        ]
        body = (
            "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            + "data: [DONE]\n\n"
        )
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Connection: close\r\n\r\n" + body.encode()
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async with await asyncio.start_server(respond, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/v1", requests


@pytest.fixture
async def live_user_input_model(monkeypatch):
    if os.getenv("PYROMIND_RUN_LIVE_USER_INPUT") != "1":
        pytest.skip("Set PYROMIND_RUN_LIVE_USER_INPUT=1 to enable paid model tests")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.fail("OPENROUTER_API_KEY is required for live model tests")
    monkeypatch.setenv("PYROMIND_PI_MODEL_MAX_RETRIES", "0")
    requests = []
    results = []
    async with ClientSession(timeout=ClientTimeout(total=120)) as client:

        async def forward(request):
            body = await request.json()
            requests.append(body)
            if len(requests) > 4:
                return web.json_response({"error": "live test call budget"}, status=429)
            async with client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            ) as upstream:
                result = {"status": upstream.status, "chunks": []}
                results.append(result)
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        "Content-Type": upstream.headers.get(
                            "Content-Type", "text/event-stream"
                        )
                    },
                )
                await response.prepare(request)
                try:
                    async for chunk in upstream.content.iter_any():
                        result["chunks"].append(chunk.decode(errors="replace"))
                        await response.write(chunk)
                    await response.write_eof()
                except (ConnectionResetError, RuntimeError):
                    pass
                return response

        app = web.Application()
        app.router.add_post("/v1/chat/completions", forward)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            port = runner.addresses[0][1]
            yield f"http://127.0.0.1:{port}/v1", requests, results
        finally:
            await runner.cleanup()


@pytest.fixture
def train_reader_probe() -> TrainReaderProbe:
    return TrainReaderProbe(
        method="datasets.Dataset.from_list",
        datasets_version=version("datasets"),
        pyarrow_version=version("pyarrow"),
    )
