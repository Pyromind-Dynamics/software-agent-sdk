"""Tests for the df_stop_task tool."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest
from pyromind_sdk import PyroMindAPIClient, PyroMindAPIError
from pyromind_sdk.client.models import TrainingTaskResponse

from pyromind_common.sdk.conversation.secret_registry import SecretRegistry
from pyromind_common.tools.data_preparation import stop_task
from pyromind_common.tools.data_preparation.platform_submit import (
    DataPreparationTaskAssociation,
    DataPreparationTaskStore,
)
from pyromind_common.tools.data_preparation.stop_task import (
    DfStopTaskAction,
    DfStopTaskExecutor,
)


def _make_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "stopped",
    error: Exception | None = None,
    **params: Any,
) -> tuple[DfStopTaskExecutor, Mock, Mock]:
    client = PyroMindAPIClient(
        api_key="sdk-key", base_url="https://sdk-route.test/api/v1"
    )
    stop = Mock(
        return_value=TrainingTaskResponse(
            task_id="344", name="pipeline", status=status
        ),
        side_effect=error,
    )
    monkeypatch.setattr(client.studio, "stop", stop)
    monkeypatch.setattr(client, "close", Mock(wraps=client.close))
    factory = Mock(return_value=client)
    monkeypatch.setattr(stop_task, "create_workflow_api_client", factory)
    executor = DfStopTaskExecutor(
        task_store_dir=str(tmp_path / "tasks"),
        headers=params.pop("headers", {"cookie": "auth_token=request-token"}),
        **params,
    )
    return executor, factory, stop


def _save_association(tmp_path: Path, **overrides: Any) -> None:
    fields: dict[str, Any] = {
        "task_id": "task-9",
        "conversation_id": "conv-9",
        "run_id": "run-9",
        "output_dir": "/out/run-9",
        "input_path": "/data/in.jsonl",
        "script_path": "/scripts/p.py",
    }
    fields.update(overrides)
    store = DataPreparationTaskStore(tmp_path / "tasks")
    store.save(DataPreparationTaskAssociation(**fields))


@pytest.mark.parametrize(
    ("status", "terminal_confirmed"),
    [("stopped", True), ("running", False), ("cancelling", False)],
)
def test_stop_by_task_id_success(tmp_path, monkeypatch, status, terminal_confirmed):
    executor, factory, stop = _make_executor(tmp_path, monkeypatch, status=status)

    obs = executor(DfStopTaskAction(task_id="344"), conversation=None)

    assert obs.stopped is True
    assert obs.status == "Stopped"
    assert obs.task_id == "344"
    assert "request accepted" in obs.text
    assert obs.platform_status == status
    assert obs.terminal_confirmed is terminal_confirmed
    stop.assert_called_once_with("344")
    factory.assert_called_once_with(
        env="prod",
        cluster=None,
        auth_token="request-token",
        headers={
            "accept": "*/*",
            "content-type": "application/json",
            "cookie": "auth_token=request-token",
        },
        timeout=30,
    )
    factory.return_value.close.assert_called_once_with()


def test_stop_without_identifier(tmp_path: Path, monkeypatch) -> None:
    executor, factory, stop = _make_executor(tmp_path, monkeypatch)

    obs = executor(DfStopTaskAction(), conversation=None)

    assert obs.stopped is False
    assert obs.is_error
    assert obs.task_id is None
    factory.assert_not_called()
    stop.assert_not_called()


def test_stop_resolve_by_run_id(tmp_path: Path, monkeypatch) -> None:
    _save_association(tmp_path, task_id="task-run", run_id="run-abc")
    executor, _, stop = _make_executor(tmp_path, monkeypatch)

    obs = executor(DfStopTaskAction(run_id="run-abc"), conversation=None)

    assert obs.stopped is True
    assert obs.task_id == "task-run"
    stop.assert_called_once_with("task-run")


def test_stop_resolve_by_output_dir(tmp_path: Path, monkeypatch) -> None:
    _save_association(tmp_path, task_id="task-out", output_dir="/out/run-xyz")
    executor, _, stop = _make_executor(tmp_path, monkeypatch)

    obs = executor(DfStopTaskAction(output_dir="/out/run-xyz/"), conversation=None)

    assert obs.stopped is True
    assert obs.task_id == "task-out"
    stop.assert_called_once_with("task-out")


def test_stop_unresolved_lookup(tmp_path: Path, monkeypatch) -> None:
    executor, factory, stop = _make_executor(tmp_path, monkeypatch)

    obs = executor(DfStopTaskAction(run_id="missing"), conversation=None)

    assert obs.stopped is False
    assert obs.is_error
    factory.assert_not_called()
    stop.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        OSError("connection lost"),
        httpx.ConnectError("connection lost"),
        PyroMindAPIError("task not running", status_code=409),
        TypeError("invalid platform response"),
        ValueError("invalid platform response"),
    ],
)
def test_stop_api_error(tmp_path, monkeypatch, error):
    executor, factory, stop = _make_executor(tmp_path, monkeypatch, error=error)

    obs = executor(DfStopTaskAction(task_id="344"), conversation=None)

    assert obs.stopped is False
    assert obs.is_error
    assert obs.status == "Failed"
    assert str(error) in obs.text
    assert not obs.terminal_confirmed
    stop.assert_called_once_with("344")
    factory.return_value.close.assert_called_once_with()


@pytest.mark.parametrize(
    "identifier",
    [{"task_id": "task-9"}, {"run_id": "run-9"}, {"output_dir": "/out/run-9"}],
)
def test_stop_rejects_another_conversations_task(tmp_path, monkeypatch, identifier):
    _save_association(tmp_path, conversation_id="other-conversation")
    executor, factory, stop = _make_executor(tmp_path, monkeypatch)
    conversation = SimpleNamespace(id="conv-9")

    obs = executor(DfStopTaskAction.model_validate(identifier), cast(Any, conversation))

    assert obs.is_error
    assert not obs.stopped
    factory.assert_not_called()
    stop.assert_not_called()


def test_stop_uses_conversation_credentials_and_cluster(tmp_path, monkeypatch):
    _save_association(tmp_path)
    registry = SecretRegistry()
    registry.update_secrets({"auth_token": "conversation-token"})
    conversation = SimpleNamespace(
        id="conv-9", state=SimpleNamespace(secret_registry=registry, agent_state={})
    )
    executor, factory, stop = _make_executor(
        tmp_path,
        monkeypatch,
        env="pre2",
        cluster="cn-east-1",
        headers={"x-cluster": "cn-east-1#pre2"},
    )

    obs = executor(DfStopTaskAction(task_id="task-9"), cast(Any, conversation))

    assert obs.stopped
    stop.assert_called_once_with("task-9")
    assert factory.call_args.kwargs["auth_token"] == "conversation-token"
    assert factory.call_args.kwargs["env"] == "pre2"
    assert factory.call_args.kwargs["cluster"] == "cn-east-1"
    assert factory.return_value.studio.base_url == ("https://sdk-route.test/api/v1")


@pytest.mark.parametrize(
    ("task_id", "payload", "accepted"),
    [
        ("344", {"success": True}, True),
        ("task-9", {"success": True}, True),
        ("344", {"success": False, "message": "task not running"}, False),
    ],
)
def test_stop_preserves_custom_endpoint(
    tmp_path, monkeypatch, task_id, payload, accepted
):
    post = Mock(return_value=httpx.Response(200, json=payload))
    monkeypatch.setattr(stop_task.httpx, "post", post)
    endpoint = "https://portal.test/custom/stop_task"
    with pytest.warns(DeprecationWarning, match="custom stop_url"):
        executor, factory, stop = _make_executor(
            tmp_path, monkeypatch, stop_url=endpoint
        )

    result = executor(DfStopTaskAction(task_id=task_id))

    assert result.stopped is accepted
    assert result.is_error is not accepted
    assert result.platform_status is None
    assert not result.terminal_confirmed
    post.assert_called_once_with(
        endpoint,
        headers={
            "accept": "*/*",
            "content-type": "application/json",
            "cookie": "auth_token=request-token",
            "x-cluster": "us-west-1",
        },
        json={"task_id": int(task_id) if task_id.isdigit() else task_id},
        timeout=30.0,
    )
    factory.assert_not_called()
    stop.assert_not_called()
