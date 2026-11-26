"""Native SDK WebSocket compatibility against a controlled local platform."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.exec_stream import SandboxExecStreamError


async def test_sdk_exec_preserves_nonzero_output_and_rejects_incomplete_streams() -> (
    None
):
    requests = []
    key = "native-sdk-websocket-key"
    output = b"binary-stdout\xff\n" + b"x" * (256 * 1024)

    async def execute(request: web.Request) -> web.WebSocketResponse:
        assert request.query["token"] == key
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        body = await socket.receive_json()
        requests.append(body)
        if body["command"] == "nonzero":
            for start in range(0, len(output), 65536):
                await socket.send_bytes(b"\x01" + output[start : start + 65536])
            await socket.send_bytes(b"\x02command stderr\n")
            await socket.send_json({"type": "exit", "returncode": 17})
        elif body["command"] == "timeout":
            await socket.send_json({"type": "stdout", "data": "started"})
            await socket.receive()
        else:
            await socket.send_bytes(b"\x01partial output\n")
            await socket.close(code=4004, message=b"connection lost before exit")
        return socket

    app = web.Application()
    app.router.add_get("/api/v1/sandboxes/existing/exec-stream", execute)
    async with TestServer(app) as server:
        client = PyroMindAPIClient(
            api_key=key,
            base_url=str(server.make_url("/api/v1")),
            cluster="cn-east-1#pre2",
            max_retries=0,
        )
        try:
            async with asyncio.timeout(8):
                result = await asyncio.to_thread(
                    client.sandboxes.exec_command,
                    "existing",
                    "nonzero",
                    cwd="/tmp/with space",
                    timeout=3,
                )
                assert result.returncode == 17
                assert result.output == output.decode(errors="replace")
                assert result.stderr == "command stderr\n"
                for command, timeout, failure in (
                    ("timeout", 1, "timed out after 1s"),
                    ("disconnect", 3, "before any exit"),
                ):
                    with pytest.raises(SandboxExecStreamError, match=failure) as error:
                        await asyncio.to_thread(
                            lambda: list(
                                client.sandboxes.exec_command_stream(
                                    "existing", command, timeout=timeout
                                )
                            )
                        )
                    assert key not in str(error.value)
        finally:
            client.close()
    assert requests == [
        {"command": "nonzero", "cwd": "/tmp/with space", "tty": False, "timeout": 3},
        {"command": "timeout", "cwd": "", "tty": False, "timeout": 1},
        {"command": "disconnect", "cwd": "", "tty": False, "timeout": 3},
    ]
