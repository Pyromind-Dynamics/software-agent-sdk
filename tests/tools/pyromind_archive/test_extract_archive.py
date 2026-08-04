from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import httpx
from pyromind_sdk.client.models import TrainingTaskCreateResponse

from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.tools.pyromind_archive import (
    ExtractArchiveAction,
    ExtractArchiveExecutor,
    ExtractArchiveTool,
)


_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000789")


def _mock_storage_check_success(monkeypatch):
    """Monkeypatch httpx.post so _check_storage_file_exists returns None."""

    def _mock_post(*args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"success": True}
        return resp

    monkeypatch.setattr(httpx, "post", _mock_post)


def _secret_registry() -> SecretRegistry:
    registry = SecretRegistry()
    registry.update_secrets({"auth_token": "session-token"})
    return registry


def _fake_conversation():
    state = type(
        "FakeState",
        (),
        {
            "secret_registry": _secret_registry(),
            "agent_state": {},
        },
    )()
    return type(
        "FakeConversation",
        (),
        {"id": _CONVERSATION_ID, "state": state},
    )()


def test_extract_archive_submits_zip_workflow(monkeypatch):
    _mock_storage_check_success(monkeypatch)
    mock_client = MagicMock()
    mock_client.studio.create.return_value = TrainingTaskCreateResponse(
        task_id="task-123",
        name="agent-extract-abc12345",
        status="Pending",
    )
    client_factory = MagicMock(return_value=mock_client)
    monkeypatch.setattr(
        "openhands.tools.pyromind_archive.definition.create_workflow_api_client",
        client_factory,
    )

    conversation = _fake_conversation()
    observation = ExtractArchiveExecutor(
        env="pre",
        cluster="us-west-1",
        headers={"x-cluster": "us-west-1#pre", "request-app": "openhands"},
        timeout=5,
    )(
        ExtractArchiveAction(
            archive_path="datasets/data.zip",
            format="zip",
        ),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.task_id == "task-123"
    assert observation.run_id is not None
    assert observation.output_dir == (
        f"/.pyromind-agent/{_CONVERSATION_ID}/extracted/{observation.run_id}"
    )

    client_factory.assert_called_once_with(
        env="pre",
        cluster="us-west-1",
        auth_token="session-token",
        headers={"x-cluster": "us-west-1#pre", "request-app": "openhands"},
        timeout=5,
    )

    request = mock_client.studio.create.call_args.args[0]
    workflow = request.workflow
    assert workflow["id"] == observation.run_id
    assert workflow["edges"] == []
    assert len(workflow["nodes"]) == 1
    node = workflow["nodes"][0]
    assert node["data"]["nodeType"] == "CustomCommandNode"
    assert node["data"]["config"]["cpu"] == 1
    command = node["data"]["config"]["command"]
    pod_output_dir = f"/target-workspace{observation.output_dir}"
    assert f"mkdir -p {pod_output_dir}" in command
    assert (
        f"python3 -m zipfile -e /target-workspace/datasets/data.zip {pod_output_dir}"
    ) in command


def test_extract_archive_auto_detects_tar_gz(monkeypatch):
    _mock_storage_check_success(monkeypatch)
    mock_client = MagicMock()
    mock_client.studio.create.return_value = TrainingTaskCreateResponse(
        task_id="task-456", name="agent-extract", status="Pending"
    )
    monkeypatch.setattr(
        "openhands.tools.pyromind_archive.definition.create_workflow_api_client",
        MagicMock(return_value=mock_client),
    )

    conversation = _fake_conversation()
    observation = ExtractArchiveExecutor(
        env="pre",
        cluster="us-west-1",
        headers={"x-cluster": "us-west-1#pre"},
        timeout=5,
    )(
        ExtractArchiveAction(archive_path="datasets/data.tar.gz"),
        cast(Any, conversation),
    )

    assert not observation.is_error
    request = mock_client.studio.create.call_args.args[0]
    command = request.workflow["nodes"][0]["data"]["config"]["command"]
    assert "import tarfile,sys" in command
    assert "tarfile.open" in command
    assert "extractall" in command


def test_extract_archive_requires_conversation():
    observation = ExtractArchiveExecutor()(
        ExtractArchiveAction(archive_path="datasets/data.zip"),
        conversation=None,
    )
    assert observation.is_error
    assert "requires an active conversation" in observation.text


def test_extract_archive_unknown_format():
    observation = ExtractArchiveExecutor()(
        ExtractArchiveAction(archive_path="datasets/data.xyz"),
        conversation=cast(Any, _fake_conversation()),
    )
    assert observation.is_error
    assert "Cannot detect archive format" in observation.text


def test_extract_archive_tool_create_derives_execution_target():
    tool = ExtractArchiveTool.create(
        headers={"x-cluster": "us-west-1#pre"},
        secret_headers={"cookie": "PYROMIND_STORAGE_AUTH_COOKIE"},
        endpoint_url="https://legacy.test/std2/studio_api/api/prompt",
    )[0]

    executor = tool.executor
    assert isinstance(executor, ExtractArchiveExecutor)
    assert executor._env == "pre"
    assert executor._cluster == "us-west-1"


def test_extract_archive_tool_create_ignores_runtime_dir():
    tool = ExtractArchiveTool.create(
        headers={"x-cluster": "us-west-1"},
        runtime_dir="/some/path",
    )[0]

    executor = tool.executor
    assert isinstance(executor, ExtractArchiveExecutor)
    # runtime_dir should be silently ignored
    assert not hasattr(executor, "_runtime_dir")


def test_extract_archive_custom_output_dir(monkeypatch):
    _mock_storage_check_success(monkeypatch)
    mock_client = MagicMock()
    mock_client.studio.create.return_value = TrainingTaskCreateResponse(
        task_id="task-789", name="agent-extract", status="Pending"
    )
    monkeypatch.setattr(
        "openhands.tools.pyromind_archive.definition.create_workflow_api_client",
        MagicMock(return_value=mock_client),
    )

    conversation = _fake_conversation()
    observation = ExtractArchiveExecutor(
        env="pre",
        cluster="us-west-1",
        headers={"x-cluster": "us-west-1#pre"},
        timeout=5,
    )(
        ExtractArchiveAction(
            archive_path="datasets/data.zip",
            format="zip",
            output_dir="my/custom/output",
        ),
        cast(Any, conversation),
    )

    assert not observation.is_error
    assert observation.output_dir == "/my/custom/output"
    request = mock_client.studio.create.call_args.args[0]
    command = request.workflow["nodes"][0]["data"]["config"]["command"]
    assert "/target-workspace/my/custom/output" in command


def test_extract_archive_strips_workspace_prefix(monkeypatch):
    """Paths starting with /workspace/ should be stripped to storage paths."""
    _mock_storage_check_success(monkeypatch)
    mock_client = MagicMock()
    mock_client.studio.create.return_value = TrainingTaskCreateResponse(
        task_id="task-ws", name="agent-extract", status="Pending"
    )
    monkeypatch.setattr(
        "openhands.tools.pyromind_archive.definition.create_workflow_api_client",
        MagicMock(return_value=mock_client),
    )

    conversation = _fake_conversation()
    observation = ExtractArchiveExecutor(
        env="pre",
        cluster="us-west-1",
        headers={"x-cluster": "us-west-1#pre"},
        timeout=5,
    )(
        ExtractArchiveAction(
            archive_path="/workspace/datasets/michaelauli/data/wikipedia-biography-dataset.zip",
            format="zip",
        ),
        cast(Any, conversation),
    )

    assert not observation.is_error
    request = mock_client.studio.create.call_args.args[0]
    command = request.workflow["nodes"][0]["data"]["config"]["command"]
    # The /workspace prefix should be stripped, so the pod path should be
    # /target-workspace/datasets/... not /target-workspace/workspace/datasets/...
    assert (
        "/target-workspace/datasets/michaelauli/data/wikipedia-biography-dataset.zip"
        in command
    )


def test_extract_archive_file_not_found(monkeypatch):
    """_check_storage_file_exists should return error when file not found."""

    def _mock_post_fail(*args, **kwargs):
        resp = MagicMock()
        resp.status_code = 404
        return resp

    monkeypatch.setattr(httpx, "post", _mock_post_fail)

    conversation = _fake_conversation()
    observation = ExtractArchiveExecutor(
        env="pre",
        cluster="us-west-1",
        headers={"x-cluster": "us-west-1#pre"},
        timeout=5,
    )(
        ExtractArchiveAction(
            archive_path="datasets/data.zip",
            format="zip",
        ),
        cast(Any, conversation),
    )

    assert observation.is_error
    assert "not found in storage" in observation.text
    assert observation.status == "Failed"


def test_extract_archive_workspace_path_not_found(monkeypatch):
    """Path with /workspace/ prefix that doesn't exist in storage should error."""

    def _mock_post_fail(*args, **kwargs):
        resp = MagicMock()
        resp.status_code = 404
        return resp

    monkeypatch.setattr(httpx, "post", _mock_post_fail)

    conversation = _fake_conversation()
    observation = ExtractArchiveExecutor(
        env="pre",
        cluster="us-west-1",
        headers={"x-cluster": "us-west-1#pre"},
        timeout=5,
    )(
        ExtractArchiveAction(
            archive_path="/workspace/datasets/nonexistent/file.zip",
            format="zip",
        ),
        cast(Any, conversation),
    )

    assert observation.is_error
    # The error message should reference the stripped path (without /workspace)
    assert "not found in storage" in observation.text
    assert observation.status == "Failed"
