from __future__ import annotations

import base64
import json
import shlex

import httpx
import pytest
from pyromind_runtime.adapters.pi import (
    PyromindHttpSandboxBackend,
    SandboxGatewayError,
)


async def test_pyromind_http_backend_uses_authenticated_exec_only_boundary() -> None:
    operations: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sandbox-token"
        assert request.headers["x-cluster"] == "cluster-a"
        if request.url.path == "/api/v1/sandboxes" and request.method == "POST":
            return httpx.Response(200, json={"data": {"id": "sandbox-1"}})
        if (
            request.url.path == "/api/v1/sandboxes/sandbox-1"
            and request.method == "GET"
        ):
            return httpx.Response(200, json={"data": {"status": "running"}})
        if (
            request.url.path == "/api/v1/sandboxes/sandbox-1"
            and request.method == "DELETE"
        ):
            return httpx.Response(204)
        if request.url.path == "/api/v1/sandboxes/sandbox-1/exec":
            payload = json.loads(request.content)
            assert isinstance(payload, dict)
            command = payload["command"]
            assert isinstance(command, str)
            if command.startswith("python3 -c"):
                encoded = shlex.split(command)[-1]
                operation = json.loads(base64.urlsafe_b64decode(encoded))
                assert isinstance(operation, dict)
                operations.append(operation)
                value = (
                    base64.b64encode(b"hello").decode()
                    if operation["op"] == "read"
                    else None
                )
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "output": json.dumps({"value": value}),
                            "returncode": 0,
                            "exception_info": "",
                        }
                    },
                )
            assert command.startswith("env -i LANG=C.UTF-8 /bin/sh -lc")
            return httpx.Response(
                200,
                json={
                    "data": {
                        "output": "done",
                        "returncode": 0,
                        "exception_info": "",
                    }
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = PyromindHttpSandboxBackend(
        base_url="https://sandbox.invalid/api/v1",
        api_key="sandbox-token",
        cluster="cluster-a",
        client=client,
    )

    sandbox_id = await backend.create_sandbox(
        {"sandbox_type": "swebench", "image": "python:3.13"}
    )
    await backend.wait_until_running(sandbox_id)
    await backend.write_file(sandbox_id, "/workspace/a.txt", b"hello")
    content = await backend.read_file(sandbox_id, "/workspace/a.txt")
    result = await backend.exec(
        sandbox_id,
        "pwd",
        cwd="/workspace",
        environment={"LANG": "C.UTF-8"},
        timeout_seconds=3.2,
    )
    await backend.delete_sandbox(sandbox_id)
    await client.aclose()

    assert content == b"hello"
    assert result.stdout == "done"
    assert result.exit_code == 0
    assert [operation["op"] for operation in operations] == ["write", "read"]


async def test_pyromind_http_backend_maps_api_errors_without_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="credential details must not escape")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = PyromindHttpSandboxBackend(
        base_url="https://sandbox.invalid",
        api_key="sandbox-token",
        cluster="cluster-a",
        client=client,
    )

    try:
        await backend.read_file("sandbox-1", "/workspace/a.txt")
    except RuntimeError as exc:
        assert str(exc) == (
            "Sandbox API POST /sandboxes/sandbox-1/exec returned HTTP 403"
        )
        assert "credential details" not in str(exc)
    else:
        raise AssertionError("sandbox error was not raised")
    finally:
        await client.aclose()


async def test_pyromind_http_backend_reports_safe_validation_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "type": "missing",
                        "loc": ["body", "resource_config"],
                        "msg": "Field required",
                        "input": {"secret": "must-not-escape"},
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = PyromindHttpSandboxBackend(
        base_url="https://sandbox.invalid",
        api_key="sandbox-token",
        cluster="cluster-a",
        client=client,
    )

    with pytest.raises(
        SandboxGatewayError,
        match=(
            r"POST /sandboxes returned HTTP 422: "
            r"body\.resource_config: Field required \(missing\)"
        ),
    ) as exc_info:
        await backend.create_sandbox({})
    assert exc_info.value.code == "invalid"
    assert "must-not-escape" not in str(exc_info.value)
    await client.aclose()


async def test_pyromind_http_backend_reports_string_validation_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"message": "validation failed", "detail": "image is invalid"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = PyromindHttpSandboxBackend(
        base_url="https://sandbox.invalid",
        api_key="sandbox-token",
        client=client,
    )

    with pytest.raises(
        SandboxGatewayError,
        match=(
            "POST /sandboxes returned HTTP 422: "
            "validation failed; image is invalid"
        ),
    ):
        await backend.create_sandbox({})
    await client.aclose()


async def test_pyromind_http_backend_reports_nested_validation_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": {
                    "error": {"code": "INVALID_IMAGE", "message": "not allowed"}
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = PyromindHttpSandboxBackend(
        base_url="https://sandbox.invalid",
        api_key="sandbox-token",
        client=client,
    )

    with pytest.raises(
        SandboxGatewayError,
        match="POST /sandboxes returned HTTP 422: INVALID_IMAGE: not allowed",
    ):
        await backend.create_sandbox({})
    await client.aclose()


async def test_pyromind_http_backend_reports_short_plaintext_validation_detail(
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            text="image reference is required",
            headers={"content-type": "text/plain; charset=utf-8"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = PyromindHttpSandboxBackend(
        base_url="https://sandbox.invalid",
        api_key="sandbox-token",
        client=client,
    )

    with pytest.raises(
        SandboxGatewayError,
        match="POST /sandboxes returned HTTP 422: image reference is required",
    ):
        await backend.create_sandbox({})
    await client.aclose()


async def test_pyromind_http_backend_exchanges_portal_cookie_for_access_key() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/account/find_access_key":
            assert request.method == "POST"
            assert request.headers["auth_token"] == "portal-token"
            assert request.headers["cookie"] == (
                "locale=zh; auth_token=portal-token; theme=dark"
            )
            assert request.headers["x-cluster"] == "us-west-1#pre"
            return httpx.Response(
                200,
                json={"success": True, "data": {"accessKey": "exchanged-key"}},
            )
        if request.url.path == "/api/v1/sandboxes":
            assert request.headers["authorization"] == "Bearer exchanged-key"
            assert "cookie" not in request.headers
            return httpx.Response(200, json={"data": {"id": "sandbox-1"}})
        if request.url.path == "/api/v1/sandboxes/sandbox-1":
            assert request.headers["authorization"] == "Bearer exchanged-key"
            return httpx.Response(200, json={"data": {"status": "running"}})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = PyromindHttpSandboxBackend(
        base_url="https://pre-api-portal.pyromind.ai/api/v1",
        cluster="us-west-2",
        headers={
            "cookie": "locale=zh; auth_token=portal-token; theme=dark",
            "x-cluster": "us-west-1#pre",
        },
        client=client,
    )

    sandbox_id = await backend.create_sandbox(
        {"sandbox_type": "swebench", "image": "python:3.13"}
    )
    await backend.wait_until_running(sandbox_id, timeout_seconds=0.1)
    await client.aclose()

    assert requests == [
        "/account/find_access_key",
        "/api/v1/sandboxes",
        "/api/v1/sandboxes/sandbox-1",
    ]


async def test_pyromind_http_backend_keeps_explicit_api_key_authoritative() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/sandboxes"
        assert request.headers["authorization"] == "Bearer sandbox-token"
        return httpx.Response(200, json={"data": {"id": "sandbox-1"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = PyromindHttpSandboxBackend(
        base_url="https://sandbox.invalid/api/v1",
        api_key="sandbox-token",
        headers={"authorization": "Bearer inbound-request-token"},
        client=client,
    )

    assert await backend.create_sandbox({}) == "sandbox-1"
    await client.aclose()


async def test_pyromind_http_backend_reports_missing_auth_without_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = PyromindHttpSandboxBackend(
        base_url="https://sandbox.invalid/api/v1",
        client=client,
    )

    with pytest.raises(
        SandboxGatewayError,
        match="sign in to Pyromind or configure PYROMIND_API_KEY",
    ):
        await backend.create_sandbox({})
    await client.aclose()
