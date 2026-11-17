"""Tests for the shared workflow submission helper."""

from __future__ import annotations

import ssl
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from openhands.tools.workflow import task_submission
from openhands.tools.workflow.task_submission import (
    _retry_transient,
    submit_workflow_task,
)


@pytest.mark.parametrize(
    ("headers", "route", "host"),
    [
        ({"X-Cluster": "us-west-2#pre"}, "us-west-2#pre", "pre-api-portal.pyromind.ai"),
        ({}, "us-west-1", "api-portal.pyromind.ai"),
    ],
)
def test_workflow_sdk_client_uses_request_route(monkeypatch, headers, route, host):
    monkeypatch.setenv("APP_ENV", "pre2")
    monkeypatch.setenv("PYROMIND_CLUSTER", "another-cluster")
    exchange = MagicMock(return_value="access-key")
    monkeypatch.setattr(task_submission, "get_api_key", exchange)

    client = task_submission.create_workflow_api_client(
        env=None, cluster=None, auth_token="session-token", headers=headers
    )

    assert exchange.call_args.kwargs["origin_headers"] == {"x-cluster": route}
    assert client.sandboxes.session.headers["X-Cluster"] == route
    assert client.sandboxes.base_url == f"https://{host}/api/v1"


class _FakeSSLError(OSError):
    """Mimics requests' SSLError (an OSError subclass) for SSL EOF."""


def test_retry_transient_recovers_after_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openhands.tools.workflow.task_submission.time.sleep", MagicMock()
    )
    calls = []

    def operation() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise _FakeSSLError(
                "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in "
                "violation of protocol (_ssl.c:1028)"
            )
        return "ok"

    assert _retry_transient(operation) == "ok"
    assert len(calls) == 3


def test_retry_transient_reraises_after_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openhands.tools.workflow.task_submission.time.sleep", MagicMock()
    )
    operation = MagicMock(side_effect=ConnectionResetError("reset"))
    with pytest.raises(ConnectionResetError):
        _retry_transient(operation)
    assert operation.call_count == 3


def test_retry_transient_passes_through_non_network_errors() -> None:
    operation = MagicMock(side_effect=ValueError("bad payload"))
    with pytest.raises(ValueError, match="bad payload"):
        _retry_transient(operation)
    assert operation.call_count == 1


def test_retry_transient_covers_httpx_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openhands.tools.workflow.task_submission.time.sleep", MagicMock()
    )
    calls = []

    def operation() -> int:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("tls handshake failed")
        if len(calls) == 2:
            raise httpx.RemoteProtocolError("peer closed connection")
        return 42

    assert _retry_transient(operation) == 42
    assert len(calls) == 3


def test_submit_workflow_task_retries_ssl_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openhands.tools.workflow.task_submission.time.sleep", MagicMock()
    )
    client = MagicMock()
    response = SimpleNamespace(task_id="task-1", status="Pending")
    client.studio.create.side_effect = [
        _FakeSSLError(str(ssl.SSLError("EOF occurred in violation of protocol"))),
        response,
    ]

    result = submit_workflow_task(
        client=client,
        workflow={"id": "w"},
        name="agent-edp",
        conversation_id="conv-1",
    )

    assert result.task_id == "task-1"
    assert client.studio.create.call_count == 2


@pytest.mark.parametrize(
    ("cluster", "use_handler", "uses_mirror"),
    [
        ("cn-east-1", False, True),
        ("cn-east-1#pre", True, True),
        ("us-west-2", False, False),
    ],
)
def test_submit_workflow_task_applies_shanghai_pip_index(
    cluster, use_handler, uses_mirror
):
    original = (
        """sh -c 'printf "%s\\n" "$PIP_INDEX_URL"'\n"""
        """pip() { printf "%s\\n" "$@"; }\n"""
        "pip install -i https://private.example/simple package\n"
        "exit 7"
    )
    node_types = [
        "CustomCommandCPUNode",
        "CustomCommandNode",
        "CustomCommandGPUNode",
        "ModelTrainSFTNode",
    ]
    workflow = {
        "nodes": [
            {"data": {"nodeType": node_type, "config": {"command": original}}}
            for node_type in node_types
        ]
    }
    result = SimpleNamespace(task_id="task-1", status="Pending")
    client = MagicMock()
    client.studio.cluster = cluster
    client.studio.create.return_value = result
    handler = MagicMock(return_value=result) if use_handler else None

    submit_workflow_task(
        client=client,
        workflow=workflow,
        name="test",
        conversation_id="conversation-1",
        submission_handler=handler,
        source_code="source workflow",
    )
    if handler:
        submitted, name, source_code = handler.call_args.args
        assert name == "test"
        assert source_code == "source workflow"
        client.studio.create.assert_not_called()
    else:
        submitted = client.studio.create.call_args.args[0].workflow

    mirror = "https://mirrors.aliyun.com/pypi/simple/"
    expected = f"export PIP_INDEX_URL={mirror}\n{original}" if uses_mirror else original
    assert [node["data"]["config"]["command"] for node in workflow["nodes"]] == [
        original
    ] * len(node_types)
    commands = [node["data"]["config"]["command"] for node in submitted["nodes"]]
    assert commands == [expected, expected, expected, original]

    if uses_mirror:
        completed = subprocess.run(
            ["bash", "-c", commands[0]], capture_output=True, text=True, check=False
        )
        assert completed.returncode == 7
        assert completed.stdout.splitlines() == [
            mirror,
            "install",
            "-i",
            "https://private.example/simple",
            "package",
        ]
        submit_workflow_task(
            client=client,
            workflow=submitted,
            name="test",
            conversation_id="conversation-1",
        )
        assert client.studio.create.call_args.args[0].workflow == submitted


@pytest.mark.asyncio
async def test_concurrent_submission_scopes_keep_the_origin_and_task_kind():
    import asyncio

    from pyromind_runtime.domain.task_correlation import (
        TaskOrigin,
        parse_task_correlation,
    )

    from openhands.tools.workflow.task_submission import workflow_submission_scope

    requests = {}
    client = MagicMock()

    def create(request):
        requests[request.name] = request
        return SimpleNamespace(task_id=request.name, status="Pending")

    client.studio.create.side_effect = create

    async def submit(identity):
        origin = TaskOrigin(type="subagent", subagent_id=identity)
        with workflow_submission_scope(origin) as scope:
            await asyncio.to_thread(
                submit_workflow_task,
                client=client,
                workflow={},
                name=identity,
                conversation_id="c",
                task_kind="data_preparation",
            )
            assert scope.submissions[identity].origin == origin
            return parse_task_correlation(requests[identity].out_id)

    first, second = await asyncio.gather(submit("worker-1"), submit("worker-2"))
    assert first is not None and second is not None
    assert first.subagent_id == "worker-1" and second.subagent_id == "worker-2"
    assert first.task_kind == second.task_kind == "data_preparation"
    submit_workflow_task(client=client, workflow={}, name="main", conversation_id="c")
    main = parse_task_correlation(requests["main"].out_id)
    assert main is not None and main.type == "main" and main.subagent_id is None
