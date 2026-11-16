"""Tests for transient-network retry in the shared workflow submission helper."""

from __future__ import annotations

import ssl
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from pyromind_runtime.domain.task_correlation import TaskOrigin

from openhands.tools.workflow import task_submission
from openhands.tools.workflow.task_submission import (
    _retry_transient,
    submit_workflow_task,
    workflow_submission_scope,
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


@pytest.mark.parametrize("test_mode", [False, True])
def test_resource_rejection_prevents_platform_create_and_scope_registration(test_mode):
    client = MagicMock()
    workflow = {"id": "actual-frozen", "nodes": [], "edges": []}
    checked = []

    def reject(definition):
        checked.append(definition)
        raise ValueError("WORKFLOW_RESOURCES_INSUFFICIENT")

    with workflow_submission_scope(TaskOrigin(), resource_validator=reject) as scope:
        with pytest.raises(ValueError, match="WORKFLOW_RESOURCES_INSUFFICIENT"):
            submit_workflow_task(
                client=client,
                workflow=workflow,
                name="admission",
                conversation_id="c",
                test_mode=test_mode,
            )
        assert scope.submissions == {} and scope.direct_tasks == set()
    assert checked == [workflow]
    client.studio.create.assert_not_called()


def test_each_direct_task_checks_resources_before_create():
    client = MagicMock()
    client.studio.create.return_value = SimpleNamespace(
        task_id="task", status="Pending"
    )
    checked = []

    def admit(workflow):
        checked.append(workflow["id"])
        if len(checked) == 2:
            raise ValueError("RESOURCE_CHECK_FAILED")

    with workflow_submission_scope(TaskOrigin(), resource_validator=admit):
        submit_workflow_task(
            client=client, workflow={"id": "first"}, name="first", conversation_id="c"
        )
        with pytest.raises(ValueError, match="RESOURCE_CHECK_FAILED"):
            submit_workflow_task(
                client=client,
                workflow={"id": "second"},
                name="second",
                conversation_id="c",
            )
    assert checked == ["first", "second"]
    assert client.studio.create.call_count == 1


@pytest.mark.asyncio
async def test_concurrent_submission_scopes_keep_the_origin_and_task_kind():
    import asyncio

    from pyromind_runtime.domain.task_correlation import (
        TaskOrigin,
        parse_task_correlation,
    )

    from openhands.tools.workflow.task_submission import workflow_submission_scope

    requests = {}
    admitted = []
    client = MagicMock()

    def create(request):
        requests[request.name] = request
        return SimpleNamespace(task_id=request.name, status="Pending")

    client.studio.create.side_effect = create

    async def submit(identity):
        origin = TaskOrigin(type="subagent", subagent_id=identity)
        with workflow_submission_scope(
            origin,
            resource_validator=lambda workflow: admitted.append(
                (identity, workflow["owner"])
            ),
        ) as scope:
            await asyncio.to_thread(
                submit_workflow_task,
                client=client,
                workflow={"owner": identity},
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
    assert sorted(admitted) == [("worker-1", "worker-1"), ("worker-2", "worker-2")]
    submit_workflow_task(client=client, workflow={}, name="main", conversation_id="c")
    assert len(admitted) == 2
    main = parse_task_correlation(requests["main"].out_id)
    assert main is not None and main.type == "main" and main.subagent_id is None
