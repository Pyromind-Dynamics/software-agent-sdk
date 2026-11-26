"""Operate sandbox files through a real terminal process and the installed SDK."""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestServer

from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.agent_profiles.pyromind import PYROMIND_TOOL_ALLOWLIST
from pyromind_common.tools.utils import pyromind_api_client
from pyromind_common.tools.workflow import task_submission
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import SessionSpec

from .test_native_preparation import stream


SDK_KEY = "platform-sdk-test-access-key"
CLUSTER = "cn-east-1#pre2"


async def test_main_terminal_uses_sdk_after_file_tools_are_withdrawn(
    tmp_path: Path, monkeypatch
) -> None:
    payloads = {
        "/tmp/example.txt": "SDK read succeeded: 已验证\n".encode(),
        "/tmp/large.txt": b"large-sdk-read\n" + b"x" * 90_000 + b"\nSDK-LARGE-END\n",
    }
    deletable_files = {
        "/tmp/deletable.txt": b"delete this file",
        "/tmp/directory/child.txt": b"delete this directory",
    }
    model_requests = []
    platform_requests = []
    delete_requests = []
    credential_requests = []
    actions = [
        (
            "terminal",
            {
                "command": '"$PYROMIND_SDK_PYTHON" -c '
                + shlex.quote(
                    "import os; from pyromind_sdk import PyroMindAPIClient; "
                    "client = PyroMindAPIClient(max_retries=0); "
                    "print(client.sandboxes.read_file('existing', "
                    "'/tmp/example.txt').decode()); "
                    "print(os.environ['PYROMIND_API_KEY']); client.close()"
                ),
                "timeout": 30,
            },
        ),
        (
            "terminal",
            {
                "command": '"$PYROMIND_SDK_PYTHON" -c '
                + shlex.quote(
                    "from pyromind_sdk import PyroMindAPIClient; "
                    "client = PyroMindAPIClient(max_retries=0); "
                    "print(client.sandboxes.read_file('existing', "
                    "'/tmp/large.txt').decode()); client.close()"
                ),
                "timeout": 30,
            },
        ),
        (
            "terminal",
            {
                "command": '"$PYROMIND_SDK_PYTHON" -c '
                + shlex.quote(
                    "from pyromind_sdk import PyroMindAPIClient\n"
                    "client = PyroMindAPIClient(max_retries=0)\n"
                    "client.sandboxes.delete_file('existing', "
                    "'/tmp/deletable.txt')\nprint('SDK-FILE-DELETED')\n"
                    "client.sandboxes.delete_file('existing', "
                    "'/tmp/directory', recursive=True)\n"
                    "print('SDK-DIRECTORY-DELETED')\n"
                    "client.sandboxes.delete_file('existing', "
                    "'/tmp/unavailable.txt')\n"
                    "print('SDK-DELETE-UNEXPECTED-SUCCESS')\nclient.close()"
                ),
                "timeout": 30,
            },
        ),
    ]
    read_again = actions[0]

    async def model(request: web.Request) -> web.Response:
        model_requests.append(await request.json())
        action = actions.pop(0) if actions else None
        return (
            stream(*action, call=f"sdk-call-{len(model_requests)}")
            if action
            else stream()
        )

    async def sandbox_file(request: web.Request) -> web.Response:
        platform_requests.append(
            (
                request.headers.get("authorization"),
                request.headers.get("x-cluster"),
                request.headers.get("cookie"),
                request.query["path"],
            )
        )
        return web.Response(body=payloads[request.query["path"]])

    async def delete_file(request: web.Request) -> web.Response:
        path = request.query["path"]
        recursive = request.query["recursive"]
        delete_requests.append(
            (
                request.headers.get("authorization"),
                request.headers.get("x-cluster"),
                request.headers.get("cookie"),
                path,
                recursive,
            )
        )
        if path == "/tmp/unavailable.txt":
            return web.json_response({"detail": "SDK deletion unavailable"}, status=503)
        targets = [
            name
            for name in deletable_files
            if name == path or (recursive == "true" and name.startswith(path + "/"))
        ]
        for name in targets:
            del deletable_files[name]
        return web.json_response({"success": True, "data": {"path": path}})

    def access_key(**kwargs) -> str:
        credential_requests.append(kwargs)
        return SDK_KEY

    app = web.Application()
    app.router.add_post("/v1/chat/completions", model)
    app.router.add_get("/api/v1/sandboxes/existing/files/read", sandbox_file)
    app.router.add_delete("/api/v1/sandboxes/existing/files/delete", delete_file)
    monkeypatch.setenv("LLM_API_KEY", "local-model-key")
    monkeypatch.setattr(task_submission, "get_api_key", access_key)
    root = tmp_path / "conversations"
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(root, adapter, default_harness_id="pi")
    context = RequestContext(
        user_id="42", cookie="auth_token=authenticated-context", x_cluster=CLUSTER
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
                    conversation_id="sdk-reader",
                    user_id="42",
                    workspace_root=str(root / "sdk-reader"),
                    model_configuration={
                        "model": "sdk-local-model",
                        "base_url": str(server.make_url("/v1")),
                        "api_key": "local-model-key",
                    },
                ),
                context,
            )
            await runtime.submit_command(
                "sdk-reader",
                UserMessageCommand(
                    command_id="read-files",
                    content=(TextContent(text="Read the existing sandbox files."),),
                ),
                context,
            )
            async with asyncio.timeout(40):
                while True:
                    snapshot = await runtime.get_snapshot("sdk-reader", context)
                    if len(model_requests) >= 4 and snapshot.status != "running":
                        break
                    await asyncio.sleep(0.02)
            assert platform_requests == [
                (f"Bearer {SDK_KEY}", CLUSTER, None, path) for path in payloads
            ], json.dumps(
                [
                    message["content"]
                    for body in model_requests
                    for message in body["messages"]
                    if message["role"] == "tool"
                ]
            )[:2000]
            assert credential_requests
            assert snapshot.external_tasks == ()
            assert not deletable_files
            assert delete_requests == [
                (f"Bearer {SDK_KEY}", CLUSTER, None, path, recursive)
                for path, recursive in (
                    ("/tmp/deletable.txt", "false"),
                    ("/tmp/directory", "true"),
                    ("/tmp/unavailable.txt", "false"),
                )
            ]
            assert all(
                item["auth_token"] == "authenticated-context"
                and item["env"] == "pre2"
                and item["origin_headers"]["x-cluster"] == CLUSTER
                for item in credential_requests
            )
            assert all(
                not {"sandbox_read_file", "sandbox_delete_file"}
                & {tool["function"]["name"] for tool in body["tools"]}
                for body in model_requests
            )
            first_result = json.dumps(
                [
                    message
                    for message in model_requests[1]["messages"]
                    if message["role"] == "tool"
                ],
                ensure_ascii=False,
            )
            assert "SDK read succeeded: 已验证" in first_result
            assert SDK_KEY not in json.dumps(model_requests)
            large_result = json.dumps(
                [
                    message
                    for message in model_requests[2]["messages"]
                    if message["role"] == "tool"
                ]
            )
            assert "SDK-LARGE-END" in large_result
            assert "Full output:" in large_result
            assert len(large_result) < len(payloads["/tmp/large.txt"])
            deletion_result = json.dumps(
                [
                    message
                    for message in model_requests[3]["messages"]
                    if message["role"] == "tool"
                ]
            )
            assert "SDK-FILE-DELETED" in deletion_result
            assert "SDK-DIRECTORY-DELETED" in deletion_result
            assert "503" in deletion_result
            assert "Command exited with code 1" in deletion_result
            assert "SDK-DELETE-UNEXPECTED-SUCCESS" not in deletion_result

            files = adapter._sessions["sdk-reader"].files
            legacy_config = files.load_session()
            legacy_config["tool_allowlist"] = list(PYROMIND_TOOL_ALLOWLIST)
            files.save_session(legacy_config)
            assert {"sandbox_read_file", "sandbox_delete_file"} <= set(
                files.load_session()["tool_allowlist"]
            )
            await runtime.close()

            adapter = PiAdapter(root, terminal_backend="os-sandbox")
            runtime = ConversationRuntime(root, adapter, default_harness_id="pi")
            actions.append(read_again)
            await runtime.submit_command(
                "sdk-reader",
                UserMessageCommand(
                    command_id="read-after-restart",
                    content=(TextContent(text="Read the small file again."),),
                ),
                context,
            )
            async with asyncio.timeout(40):
                while True:
                    snapshot = await runtime.get_snapshot("sdk-reader", context)
                    if len(model_requests) >= 6 and snapshot.status != "running":
                        break
                    await asyncio.sleep(0.02)
            assert platform_requests[-1] == (
                f"Bearer {SDK_KEY}",
                CLUSTER,
                None,
                "/tmp/example.txt",
            )
            assert len(platform_requests) == 3
            assert snapshot.external_tasks == ()
            assert len(delete_requests) == 3
            assert all(
                not {"sandbox_read_file", "sandbox_delete_file"}
                & {tool["function"]["name"] for tool in body["tools"]}
                for body in model_requests[4:]
            )
            restarted_results = json.dumps(
                [
                    message
                    for message in model_requests[-1]["messages"]
                    if message["role"] == "tool"
                ],
                ensure_ascii=False,
            )
            assert "SDK read succeeded: 已验证" in restarted_results
            assert SDK_KEY not in json.dumps(model_requests)
            assert "authenticated-context" not in json.dumps(model_requests)
    finally:
        await runtime.close()
    for path in root.rglob("*"):
        if path.is_file():
            persisted = path.read_bytes()
            assert SDK_KEY.encode() not in persisted, str(path.relative_to(root))
            assert b"authenticated-context" not in persisted, str(
                path.relative_to(root)
            )
