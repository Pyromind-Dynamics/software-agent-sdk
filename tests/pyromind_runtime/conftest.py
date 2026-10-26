from __future__ import annotations

import asyncio
import json
import os

import pytest
from aiohttp import ClientSession, ClientTimeout, web


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
