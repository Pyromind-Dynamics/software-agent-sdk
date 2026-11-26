"""Exercise migrated atomic operations through the real main-agent terminal."""

from __future__ import annotations

from aiohttp import web


async def test_sdk_lifecycle_chunked_writes_and_task_stop(sdk_terminal_run) -> None:
    created = []
    lifecycle = []
    status = "creating"
    polls = 0
    uploads = {}
    files = {}
    task_events = []

    def sandbox() -> web.Response:
        return web.json_response(
            {
                "success": True,
                "data": {
                    "sandbox_id": "sdk-sandbox",
                    "name": "sdk-e2e",
                    "sandbox_type": "custom",
                    "status": status,
                },
            }
        )

    async def create(request: web.Request) -> web.Response:
        created.append(await request.json())
        lifecycle.append("create")
        return sandbox()

    async def inspect(request: web.Request) -> web.Response:
        nonlocal polls, status
        polls += 1
        if status == "creating" and polls >= 2:
            status = "running"
        elif status == "pausing":
            status = "paused"
        lifecycle.append(f"get:{status}")
        return sandbox()

    async def pause(request: web.Request) -> web.Response:
        nonlocal status
        lifecycle.append("pause")
        status = "pausing"
        return sandbox()

    async def delete(request: web.Request) -> web.Response:
        lifecycle.append(f"delete:{status}")
        if status != "paused":
            return web.json_response(
                {"message": "can not delete while running"}, status=409
            )
        return web.json_response({"success": True})

    async def initialize_upload(request: web.Request) -> web.Response:
        upload_id = str(len(uploads))
        uploads[upload_id] = {
            "total_size": int(request.query["total_size"]),
            "total_chunks": int(request.query["total_chunks"]),
            "parts": [],
        }
        return web.json_response({"success": True, "data": {"upload_id": upload_id}})

    async def upload_part(request: web.Request) -> web.Response:
        upload = uploads[request.match_info["upload_id"]]
        assert int(request.query["chunk_index"]) == len(upload["parts"])
        upload["parts"].append(await request.read())
        return web.json_response({"success": True})

    async def complete_upload(request: web.Request) -> web.Response:
        upload = uploads[request.match_info["upload_id"]]
        assert int(request.query["total_chunks"]) == upload["total_chunks"]
        payload = b"".join(upload["parts"])
        assert len(payload) == upload["total_size"]
        files[request.query["path"]] = payload
        return web.json_response(
            {
                "success": True,
                "data": {"path": request.query["path"], "size": len(payload)},
            }
        )

    async def read_file(request: web.Request) -> web.Response:
        return web.Response(body=files[request.query["path"]])

    async def stop_task(request: web.Request) -> web.Response:
        task_events.append("stop")
        return web.json_response({"success": True, "data": {"task_id": "task-42"}})

    async def inspect_task(request: web.Request) -> web.Response:
        task_status = "running" if task_events == ["stop"] else "stopped"
        task_events.append(f"get:{task_status}")
        return web.json_response(
            {
                "success": True,
                "data": {
                    "task_id": "task-42",
                    "name": "SDK task",
                    "status": task_status,
                },
            }
        )

    app = web.Application(client_max_size=3 * 1024 * 1024)
    app.router.add_post("/api/v1/sandboxes", create)
    app.router.add_get("/api/v1/sandboxes/sdk-sandbox", inspect)
    app.router.add_post("/api/v1/sandboxes/sdk-sandbox/pause", pause)
    app.router.add_delete("/api/v1/sandboxes/sdk-sandbox", delete)
    upload_route = "/api/v1/sandboxes/sdk-sandbox/files/chunks"
    app.router.add_post(upload_route + "/init", initialize_upload)
    app.router.add_put(upload_route + "/{upload_id}/part", upload_part)
    app.router.add_post(upload_route + "/{upload_id}/complete", complete_upload)
    app.router.add_get("/api/v1/sandboxes/sdk-sandbox/files/read", read_file)
    app.router.add_post("/api/v1/studio/tasks/task-42/stop", stop_task)
    app.router.add_get("/api/v1/studio/tasks/task-42", inspect_task)

    result = await sdk_terminal_run(
        app,
        """import hashlib
import os
from pyromind_sdk import PyroMindAPIClient
from pyromind_sdk.client.base import PyroMindAPIError
from pyromind_sdk.client.models import ResourceConfig, SandboxRequest, SandboxType

client = PyroMindAPIClient(max_retries=0)
request = SandboxRequest(
    name="sdk-e2e",
    sandbox_type=SandboxType.CUSTOM,
    image=os.environ["PYROMIND_SANDBOX_DEFAULT_IMAGE"],
    resources=ResourceConfig(cpu="4", memory="8Gi"),
)
sandbox = client.sandboxes.create_and_wait(
    request, target_status="running", timeout=5, check_interval=1
)
assert sandbox.status == "running"
print("SDK-CREATE-READY")

for path, content in (
    ("/tmp/empty.txt", b""),
    ("/tmp/nested/large.bin", b"x" * (2 * 1024 * 1024) + b"\\0SDK-upload-end"),
):
    client.sandboxes.write_file(sandbox.id, path, content)
    actual = client.sandboxes.read_file(sandbox.id, path)
    assert hashlib.sha256(actual).digest() == hashlib.sha256(content).digest()
print("SDK-EMPTY-AND-MULTIPART-ROUNDTRIP")

try:
    client.sandboxes.delete(sandbox.id)
except PyroMindAPIError as error:
    assert error.status_code == 409
    client.sandboxes.pause(sandbox.id)
    assert client.sandboxes.wait_for_sandbox_status(
        sandbox.id, "paused", timeout=5, check_interval=1,
        intermediate_statuses=["running", "pausing"]
    )
    client.sandboxes.delete(sandbox.id)
else:
    raise AssertionError("Running sandbox deletion should require pause")
print("SDK-PAUSE-CONFIRMED-AND-DELETED")

accepted = client.studio.stop("task-42")
assert accepted.status == "running"
print("SDK-STOP-REQUEST-ACCEPTED")
confirmed = client.studio.get_task(accepted.task_id)
assert confirmed.status == "stopped"
print("SDK-STOP-TERMINAL-CONFIRMED")
client.close()
""",
    )

    assert len(created) == 1
    assert created[0]["image"] == "registry.example/sdk-test:v1"
    assert created[0]["resources"]["cpu"] == "4"
    assert created[0]["resources"]["memory"] == "8Gi"
    assert created[0]["sandbox_type"] == "custom"
    assert lifecycle == [
        "create",
        "get:creating",
        "get:running",
        "get:running",
        "delete:running",
        "pause",
        "get:paused",
        "delete:paused",
    ]
    assert uploads["0"] == {"total_size": 0, "total_chunks": 0, "parts": []}
    assert [len(part) for part in uploads["1"]["parts"]] == [2 * 1024 * 1024, 15]
    assert files["/tmp/empty.txt"] == b""
    assert files["/tmp/nested/large.bin"].endswith(b"\0SDK-upload-end")
    assert task_events == ["stop", "get:running", "get:stopped"]
    for marker in (
        "SDK-CREATE-READY",
        "SDK-EMPTY-AND-MULTIPART-ROUNDTRIP",
        "SDK-PAUSE-CONFIRMED-AND-DELETED",
        "SDK-STOP-REQUEST-ACCEPTED",
        "SDK-STOP-TERMINAL-CONFIRMED",
    ):
        assert marker in result
