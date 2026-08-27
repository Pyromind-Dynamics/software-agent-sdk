"""Tests for the Pyromind sandbox tools."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pyromind_sdk.client.models import SandboxResponse, SandboxType

from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.conversation.secret_registry import SecretRegistry
from openhands.sdk.tool import Tool
from openhands.sdk.tool.registry import resolve_tool
from openhands.tools.sandbox import (
    SandboxCreateAction,
    SandboxCreateExecutor,
    SandboxDeleteAction,
    SandboxDeleteExecutor,
    SandboxMountInput,
    SandboxPortInput,
    SandboxReadFileAction,
    SandboxReadFileExecutor,
    SandboxTerminalAction,
    SandboxTerminalExecutor,
)


_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000111")
_SANDBOX_ID = "sb-1"


def _executor_kwargs(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "env": "pre",
        "cluster": "us-west-1",
        "current_user": object(),
        "headers": {"x-cluster": "us-west-1#pre", "request-app": "openhands"},
    }
    defaults.update(overrides)
    return defaults


def _fake_conversation() -> BaseConversation:
    registry = SecretRegistry()
    registry.update_secrets({"auth_token": "jwt-token"})
    return cast(
        BaseConversation,
        SimpleNamespace(
            id=_CONVERSATION_ID,
            workspace=SimpleNamespace(working_dir="/tmp"),
            state=SimpleNamespace(secret_registry=registry, agent_state={}),
        ),
    )


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: MagicMock) -> None:
    monkeypatch.setattr(
        "openhands.tools.workflow.task_submission.get_api_key",
        lambda **kwargs: "access-key-1",
    )
    monkeypatch.setattr(
        "openhands.tools.workflow.task_submission.get_pyromind_api_client",
        lambda **kwargs: client,
    )


class _FakeTerminal:
    def __init__(self, url: str, chunks: list[bytes]):
        self.url = url
        self.chunks = list(chunks)
        self.sent: list[bytes] = []

    def __enter__(self) -> "_FakeTerminal":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def send(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, timeout: float) -> bytes:
        if not self.chunks:
            raise TimeoutError
        return self.chunks.pop(0)


def _fake_terminal(
    monkeypatch: pytest.MonkeyPatch, chunks: list[bytes]
) -> list[_FakeTerminal]:
    terminals: list[_FakeTerminal] = []

    def fake_connect(url: str, **kwargs: object) -> _FakeTerminal:
        terminal = _FakeTerminal(url, chunks)
        terminals.append(terminal)
        return terminal

    monkeypatch.setattr("openhands.tools.sandbox.definition.connect", fake_connect)
    return terminals


def _sandbox(**overrides: Any) -> SandboxResponse:
    values: dict[str, Any] = {
        "id": _SANDBOX_ID,
        "name": "demo",
        "type": SandboxType.CUSTOM,
        "status": "running",
    }
    values.update(overrides)
    return SandboxResponse(**values)


def test_sandbox_tools_create_and_resolve() -> None:
    """Every sandbox tool registers under its snake_case name and resolves."""
    params = _executor_kwargs()
    names = {
        "sandbox_create",
        "sandbox_delete",
        "sandbox_read_file",
        "sandbox_terminal",
    }
    for name in names:
        resolved = resolve_tool(Tool(name=name, params=params), cast(Any, None))
        assert len(resolved) == 1
        assert resolved[0].name == name

    create_tool = resolve_tool(
        Tool(name="sandbox_create", params=params), cast(Any, None)
    )[0]
    assert create_tool.annotations is not None
    assert {
        "name",
        "image",
        "volume_mounts",
        "port_mappings",
        "cpu",
        "memory",
        "wait_timeout",
    } <= set(create_tool.action_type.model_fields)


def test_sandbox_create_waits_for_running(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.sandboxes.create.return_value = _sandbox(status="creating")
    client.sandboxes.wait_for_sandbox_status.return_value = True
    client.sandboxes.get_sandbox.return_value = _sandbox()
    _patch_client(monkeypatch, client)

    observation = SandboxCreateExecutor(**_executor_kwargs())(
        SandboxCreateAction(
            name="demo",
            cpu=8,
            image="python:3.12-slim",
            wait_timeout=30,
        ),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert observation.sandbox_id == _SANDBOX_ID
    assert observation.status == "running"

    request = client.sandboxes.create.call_args[0][0]
    assert request.name == "demo"
    assert request.sandbox_type == SandboxType.CUSTOM
    assert request.image == "python:3.12-slim"
    assert request.resources.cpu == "8"
    assert request.resources.memory == "16Gi"
    assert request.configuration is None
    assert request.system_image_path is None
    client.sandboxes.wait_for_sandbox_status.assert_called_once_with(
        _SANDBOX_ID, target_status="running", timeout=30
    )
    client.sandboxes.get_sandbox.assert_called_once_with(_SANDBOX_ID)


def test_sandbox_create_default_image_by_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.create.return_value = _sandbox(status="creating")
    _patch_client(monkeypatch, client)

    observation = SandboxCreateExecutor(**_executor_kwargs(cluster="us-west-2"))(
        SandboxCreateAction(wait_timeout=0), _fake_conversation()
    )

    assert not observation.is_error
    assert (
        client.sandboxes.create.call_args[0][0].image
        == "pyrominddynamics/jupyter-lab-with-ssh:v0.9-aws"
    )


def test_sandbox_create_other_cluster_default_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.create.return_value = _sandbox(status="creating")
    _patch_client(monkeypatch, client)

    observation = SandboxCreateExecutor(**_executor_kwargs(cluster="us-west-1"))(
        SandboxCreateAction(wait_timeout=0), _fake_conversation()
    )

    assert not observation.is_error
    assert (
        client.sandboxes.create.call_args[0][0].image
        == "pyrominddynamics/jupyter-lab-with-ssh:v0.9"
    )


def test_sandbox_create_custom_mounts_and_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.create.return_value = _sandbox(status="creating")
    _patch_client(monkeypatch, client)

    observation = SandboxCreateExecutor(**_executor_kwargs())(
        SandboxCreateAction(
            volume_mounts=[
                SandboxMountInput(host_path="/workspace", mount_path="/data")
            ],
            port_mappings=[SandboxPortInput(container_port=8080)],
            wait_timeout=0,
        ),
        _fake_conversation(),
    )

    assert not observation.is_error
    request = client.sandboxes.create.call_args[0][0]
    assert request.volume_mounts[0].host_path == "/workspace"
    assert request.volume_mounts[0].mount_path == "/data"
    assert request.volume_mounts[0].read_only is False
    assert request.port_mappings[0].container_port == 8080
    assert request.port_mappings[0].protocol == "TCP"


def test_sandbox_create_no_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.sandboxes.create.return_value = _sandbox(status="creating")
    _patch_client(monkeypatch, client)

    observation = SandboxCreateExecutor(**_executor_kwargs())(
        SandboxCreateAction(wait_timeout=0), _fake_conversation()
    )

    assert not observation.is_error
    assert observation.status == "creating"
    client.sandboxes.wait_for_sandbox_status.assert_not_called()
    client.sandboxes.get_sandbox.assert_not_called()


def test_sandbox_create_not_running_within_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.create.return_value = _sandbox(status="creating")
    client.sandboxes.wait_for_sandbox_status.return_value = False
    client.sandboxes.get_sandbox.return_value = _sandbox(status="pending")
    _patch_client(monkeypatch, client)

    observation = SandboxCreateExecutor(**_executor_kwargs())(
        SandboxCreateAction(wait_timeout=5), _fake_conversation()
    )

    assert not observation.is_error
    assert observation.status == "pending"
    assert "did not reach 'running'" in observation.text


def test_sandbox_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    _patch_client(monkeypatch, client)

    observation = SandboxDeleteExecutor(**_executor_kwargs())(
        SandboxDeleteAction(sandbox_id=_SANDBOX_ID), _fake_conversation()
    )

    assert not observation.is_error
    assert f"Sandbox {_SANDBOX_ID} deleted" in observation.text
    client.sandboxes.delete.assert_called_once_with(_SANDBOX_ID)
    client.sandboxes.pause.assert_not_called()


def test_sandbox_delete_pauses_running_sandbox_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.delete.side_effect = [
        RuntimeError(
            "INTERNAL_SERVER_ERROR: InstanceService.delete_instance-"
            "instance`s status is Running, can not delete!"
        ),
        None,
    ]
    _patch_client(monkeypatch, client)

    observation = SandboxDeleteExecutor(**_executor_kwargs())(
        SandboxDeleteAction(sandbox_id=_SANDBOX_ID), _fake_conversation()
    )

    assert not observation.is_error
    assert f"Sandbox {_SANDBOX_ID} deleted" in observation.text
    client.sandboxes.pause.assert_called_once_with(_SANDBOX_ID)
    assert client.sandboxes.delete.call_count == 2


def test_sandbox_delete_other_errors_do_not_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.delete.side_effect = RuntimeError("NOT_FOUND: no such sandbox")
    _patch_client(monkeypatch, client)

    observation = SandboxDeleteExecutor(**_executor_kwargs())(
        SandboxDeleteAction(sandbox_id=_SANDBOX_ID), _fake_conversation()
    )

    assert observation.is_error
    assert "no such sandbox" in observation.text
    client.sandboxes.pause.assert_not_called()


def test_sandbox_read_file_text(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.sandboxes.read_file.return_value = b"hello world\n"
    _patch_client(monkeypatch, client)

    observation = SandboxReadFileExecutor(**_executor_kwargs())(
        SandboxReadFileAction(sandbox_id=_SANDBOX_ID, path="/data/README.md"),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert observation.encoding == "utf-8"
    assert observation.file_content == "hello world\n"
    assert observation.truncated is False
    client.sandboxes.read_file.assert_called_once_with(_SANDBOX_ID, "/data/README.md")


def test_sandbox_read_file_binary_is_base64_encoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.read_file.return_value = b"\x89PNG\r\n\x1a\n"
    _patch_client(monkeypatch, client)

    observation = SandboxReadFileExecutor(**_executor_kwargs())(
        SandboxReadFileAction(sandbox_id=_SANDBOX_ID, path="/data/logo.png"),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert observation.encoding == "base64"
    assert observation.file_content is not None
    assert observation.file_content.startswith("iVBORw0KGgo=")


def test_sandbox_read_file_truncates_large_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.read_file.return_value = b"x" * 50000
    _patch_client(monkeypatch, client)

    observation = SandboxReadFileExecutor(**_executor_kwargs())(
        SandboxReadFileAction(sandbox_id=_SANDBOX_ID, path="/data/large.log"),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert observation.truncated is True
    assert "truncated" in observation.text
    assert observation.file_content is not None
    assert len(observation.file_content) < 50000


def test_sandbox_read_file_surfaces_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.read_file.side_effect = RuntimeError("boom")
    _patch_client(monkeypatch, client)

    observation = SandboxReadFileExecutor(**_executor_kwargs())(
        SandboxReadFileAction(sandbox_id=_SANDBOX_ID, path="/data/x.txt"),
        _fake_conversation(),
    )

    assert observation.is_error
    assert "boom" in observation.text


def test_sandbox_executor_requires_conversation() -> None:
    observation = SandboxCreateExecutor(**_executor_kwargs())(
        SandboxCreateAction(), None
    )

    assert observation.is_error
    assert "active conversation" in observation.text


def test_sandbox_terminal_command(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.sandboxes.base_url = "https://pre-api-portal.pyromind.ai/api/v1"
    client.sandboxes.api_key = "access-key-1"
    _patch_client(monkeypatch, client)
    terminal = _fake_terminal(
        monkeypatch,
        [
            b"root@pod:/# ",
            b"hi\r\n",
            b"__PYROMIND_TERMINAL_EXIT__:0\r\n",
        ],
    )

    observation = SandboxTerminalExecutor(**_executor_kwargs())(
        SandboxTerminalAction(
            sandbox_id=_SANDBOX_ID,
            command="echo hi",
            cwd="/data",
            timeout_seconds=30,
        ),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert observation.returncode == 0
    assert observation.timed_out is False
    assert observation.output is not None
    assert "hi" in observation.output
    sent = terminal[0].sent
    assert b"cd /data && echo hi" in sent[0]
    assert sent[1].startswith(b"echo __PYROMIND_TERMINAL_EXIT__:$?")
    assert terminal[0].url.startswith("wss://pre-api.pyromind.ai")
    assert "token=access-key-1" in terminal[0].url


def test_sandbox_terminal_falls_back_to_client_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.base_url = "https://api-portal.pyromind.ai/api/v1"
    client.sandboxes.api_key = "access-key-1"
    client.sandboxes.cluster = None
    _patch_client(monkeypatch, client)
    terminal = _fake_terminal(monkeypatch, [b"__PYROMIND_TERMINAL_EXIT__:0\r\n"])

    observation = SandboxTerminalExecutor(
        **_executor_kwargs(cluster="unknown-cluster")
    )(
        SandboxTerminalAction(sandbox_id=_SANDBOX_ID, command="ls"),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert terminal[0].url.startswith("wss://api-portal.pyromind.ai")


def test_sandbox_terminal_surfaces_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.base_url = "https://pre-api.pyromind.ai/api/v1"
    client.sandboxes.api_key = "access-key-1"
    _patch_client(monkeypatch, client)
    _fake_terminal(monkeypatch, [b"__PYROMIND_TERMINAL_EXIT__:127\r\n"])

    observation = SandboxTerminalExecutor(**_executor_kwargs())(
        SandboxTerminalAction(sandbox_id=_SANDBOX_ID, command="nope"),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert observation.returncode == 127


def test_sandbox_terminal_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.sandboxes.base_url = "https://pre-api.pyromind.ai/api/v1"
    client.sandboxes.api_key = "access-key-1"
    _patch_client(monkeypatch, client)
    _fake_terminal(monkeypatch, [b"$ "])  # never emits the exit marker

    observation = SandboxTerminalExecutor(**_executor_kwargs())(
        SandboxTerminalAction(sandbox_id=_SANDBOX_ID, command="sleep 9"),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert observation.timed_out is True
    assert observation.returncode is None


def test_sandbox_terminal_surfaces_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.base_url = "https://pre-api.pyromind.ai/api/v1"
    client.sandboxes.api_key = "access-key-1"
    _patch_client(monkeypatch, client)

    def broken_connect(url: str, **kwargs: object) -> object:
        raise RuntimeError("ws refused")

    monkeypatch.setattr("openhands.tools.sandbox.definition.connect", broken_connect)

    observation = SandboxTerminalExecutor(**_executor_kwargs())(
        SandboxTerminalAction(sandbox_id=_SANDBOX_ID, command="ls"),
        _fake_conversation(),
    )

    assert observation.is_error
    assert "ws refused" in observation.text
