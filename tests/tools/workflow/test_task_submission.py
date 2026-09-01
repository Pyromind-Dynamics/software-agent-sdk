"""Tests for transient-network retry in the shared workflow submission helper."""

from __future__ import annotations

import ssl
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from openhands.tools.workflow.task_submission import (
    _retry_transient,
    submit_workflow_task,
)


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
