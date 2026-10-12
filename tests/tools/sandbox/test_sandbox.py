"""Tests for the Pyromind sandbox tools."""

import base64
import re
import subprocess
import tempfile
from pathlib import Path
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
    SandboxWriteFileAction,
    SandboxWriteFileExecutor,
    write_sandbox_file,
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
    def __init__(
        self,
        url: str,
        chunks: list[bytes],
        workspace: str | None = None,
    ):
        self.url = url
        self.sent: list[bytes] = []
        self.workspace = workspace
        self._scripted = list(chunks)
        self._pending: list[tuple[bytes | None, bytes]] = []

    def _execute(self, script: str) -> None:
        """Run one sent line the way the sandbox shell would."""
        if self.workspace is None:
            self.workspace = tempfile.mkdtemp(prefix="fake-sandbox-")
        completed = subprocess.run(
            ["sh", "-c", script],
            cwd=self.workspace,
            capture_output=True,
            timeout=60,
        )
        if completed.stdout:
            self._pending.append((None, completed.stdout))
        if completed.stderr:
            self._pending.append((None, completed.stderr))

    def _exit_marker(self) -> bytes:
        for data in self.sent:
            if data.startswith(b"echo __PYROMIND_TERMINAL_EXIT__"):
                return data.split(b":$?", 1)[0].removeprefix(b"echo ")
        return b""

    def __enter__(self) -> "_FakeTerminal":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def send(self, data: bytes) -> None:
        self.sent.append(data)
        if self._scripted:
            # Scripted terminals replay the chunks the test provides.
            return
        text = data.decode("utf-8", "replace").rstrip("\r\n")
        # The PTY echoes the whole input as it reaches the line discipline,
        # before the shell runs any of it. Acknowledgements and the status
        # line are printed by the shell, so the echo alone never satisfies a
        # wait.
        self._pending.append((None, f"{text}\r\n".encode()))
        self._execute(text)

    def recv(self, timeout: float) -> bytes:
        if self._scripted:
            chunk = self._scripted.pop(0)
            chunk = chunk.replace(b"<EXIT_MARKER>", self._exit_marker())
            if b"<ECHO>" in chunk:
                echoed = self.sent[1].strip() if len(self.sent) > 1 else b""
                chunk = chunk.replace(b"<ECHO>", echoed)
            return chunk
        if not self._pending:
            raise TimeoutError
        _, chunk = self._pending.pop(0)
        return chunk


def _fake_terminal(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[bytes],
    workspace: str | None = None,
) -> list[_FakeTerminal]:
    terminals: list[_FakeTerminal] = []

    def fake_connect(url: str, **kwargs: object) -> _FakeTerminal:
        terminal = _FakeTerminal(url, chunks, workspace)
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


def test_sandbox_create_uses_deployment_default_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.sandboxes.create.return_value = _sandbox(status="running")
    _patch_client(monkeypatch, client)

    SandboxCreateExecutor(
        **_executor_kwargs(cluster="us-west-1"),
        default_image="registry.test/embodied-runtime:1.29.5",
    )(SandboxCreateAction(wait_timeout=0), _fake_conversation())

    assert (
        client.sandboxes.create.call_args[0][0].image
        == "registry.test/embodied-runtime:1.29.5"
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


def _shell_argument(text: str) -> str:
    """The first shell word of ``text``, keeping quoted characters."""
    value = ""
    quote = ""
    for index, char in enumerate(text):
        if quote == "'" and char == "'":
            quote = ""
            continue
        if quote == '"' and char == '"':
            quote = ""
            continue
        if quote and char == "\\" and index + 1 < len(text):
            value += text[index + 1]
            continue
        if not quote and char in {"'", '"'}:
            quote = char
            continue
        if not quote and char == " ":
            break
        value += char
    return value


def _staged_payload(terminal: _FakeTerminal) -> bytes:
    """Rebuild the bytes one write staged, in staging order."""
    chunks = [
        _shell_argument(match.group(1).decode())
        for data in terminal.sent
        for match in re.finditer(rb"printf '%s' (.*?) >> ", data)
    ]
    return base64.b64decode("".join(chunks))


def _ack_needles(terminal: _FakeTerminal) -> list[bytes]:
    """The marker each staged burst is acknowledged by."""
    needles: list[bytes] = []
    for data in terminal.sent:
        match = re.search(rb"'__PM_ACK__' '([^']+)' ok", data)
        if match is None:
            continue
        needle = b"__PM_ACK__" + match.group(1) + b":ok"
        # The shell assembles the marker from separate words, so the command
        # line the PTY echoes back never carries it.
        assert needle not in data
        needles.append(needle)
    return needles


def test_sandbox_write_file_creates_empty_files_through_the_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The file API is reached through the portal domain, so even empty files
    # travel the cluster-direct terminal socket.
    client = MagicMock()
    _patch_client(monkeypatch, client)
    terminals = _fake_terminal(monkeypatch, [], str(tmp_path))

    observation = SandboxWriteFileExecutor(**_executor_kwargs())(
        SandboxWriteFileAction(
            sandbox_id=_SANDBOX_ID, path="pkg/__init__.py", content=""
        ),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert observation.size == 0
    client.sandboxes.write_file.assert_not_called()
    assert len(terminals) == 1
    assert terminals[0].url.startswith("wss://pre-api.pyromind.ai")
    assert any(b"mkdir -p pkg" in s for s in terminals[0].sent)
    assert any(b"mkdir -p .pyromind-agent-staging" in s for s in terminals[0].sent)
    assert any(
        b"python3 -c" in s and b" pkg/__init__.py 0" in s for s in terminals[0].sent
    )
    assert (tmp_path / "pkg" / "__init__.py").stat().st_size == 0


def test_sandbox_write_file_stages_content_over_the_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = MagicMock()
    _patch_client(monkeypatch, client)
    terminals = _fake_terminal(monkeypatch, [], str(tmp_path))

    observation = SandboxWriteFileExecutor(**_executor_kwargs())(
        SandboxWriteFileAction(sandbox_id=_SANDBOX_ID, path="notes.md", content="hi\n"),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert observation.size == 3
    client.sandboxes.write_file.assert_not_called()
    assert _staged_payload(terminals[0]) == b"hi\n"
    assert (tmp_path / "notes.md").read_bytes() == b"hi\n"


def test_sandbox_write_file_batches_a_large_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = MagicMock()
    _patch_client(monkeypatch, client)
    terminals = _fake_terminal(monkeypatch, [], str(tmp_path))
    content = "x" * 9000

    observation = SandboxWriteFileExecutor(**_executor_kwargs())(
        SandboxWriteFileAction(sandbox_id=_SANDBOX_ID, path="big.txt", content=content),
        _fake_conversation(),
    )

    assert not observation.is_error
    # One acknowledgement per staged burst; the payload is not one huge line.
    assert len(_ack_needles(terminals[0])) > 1
    assert _staged_payload(terminals[0]) == content.encode()
    assert (tmp_path / "big.txt").read_text(encoding="utf-8") == content


def test_sandbox_write_file_surfaces_a_staging_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = MagicMock()
    _patch_client(monkeypatch, client)
    _fake_terminal(monkeypatch, [], str(tmp_path))

    def broken_connect(url: str, **kwargs: object) -> object:
        raise RuntimeError("ws refused")

    monkeypatch.setattr("openhands.tools.sandbox.definition.connect", broken_connect)

    observation = SandboxWriteFileExecutor(**_executor_kwargs())(
        SandboxWriteFileAction(sandbox_id=_SANDBOX_ID, path="notes.md", content="hi\n"),
        _fake_conversation(),
    )

    assert observation.is_error
    assert "ws refused" in observation.text


class _RealShellTerminal:
    """A real shell behind the terminal interface, for transport checks."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process

    def __enter__(self) -> "_RealShellTerminal":
        return self

    def __exit__(self, *args: object) -> bool:
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(b"exit\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        return False

    def send(self, data: bytes) -> None:
        if data.startswith(b"exit"):
            return
        assert self.process.stdin is not None
        self.process.stdin.write(data.replace(b"\r\n", b"\n"))
        self.process.stdin.flush()

    def recv(self, timeout: float) -> bytes:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise TimeoutError
        return line


def test_sandbox_write_file_protocol_round_trips_through_a_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The staging protocol decodes to the exact bytes a real shell received."""
    process = subprocess.Popen(
        ["sh"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=tmp_path,
    )
    monkeypatch.setattr(
        "openhands.tools.sandbox.definition.connect",
        lambda url, **kwargs: _RealShellTerminal(process),
    )
    payload = ("hello 中文\n" * 200).encode()

    write_sandbox_file(
        cast(Any, SimpleNamespace(api_key="access-key-1")),
        ws_base_url="https://pre-api.pyromind.ai/api/v1",
        sandbox_id=_SANDBOX_ID,
        path="out.bin",
        source=payload,
    )

    assert (tmp_path / "out.bin").read_bytes() == payload


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
            b"<EXIT_MARKER>:0\r\n",
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
    assert sent[1].startswith(b"echo __PYROMIND_TERMINAL_EXIT__")
    assert sent[1].endswith(b":$?\r\n")
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
    terminal = _fake_terminal(monkeypatch, [b"<EXIT_MARKER>:0\r\n"])

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
    _fake_terminal(monkeypatch, [b"<EXIT_MARKER>:127\r\n"])

    observation = SandboxTerminalExecutor(**_executor_kwargs())(
        SandboxTerminalAction(sandbox_id=_SANDBOX_ID, command="nope"),
        _fake_conversation(),
    )

    assert not observation.is_error
    assert observation.returncode == 127


def test_sandbox_terminal_ignores_the_echoed_exit_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The TTY echoes the marker line before the command produces its status."""
    client = MagicMock()
    client.sandboxes.base_url = "https://pre-api.pyromind.ai/api/v1"
    client.sandboxes.api_key = "access-key-1"
    _patch_client(monkeypatch, client)
    _fake_terminal(
        monkeypatch,
        [
            b"root@pod:/# <ECHO>\r\n",
            b"hi\r\n",
            b"<EXIT_MARKER>:0\r\n",
        ],
    )

    observation = SandboxTerminalExecutor(**_executor_kwargs())(
        SandboxTerminalAction(sandbox_id=_SANDBOX_ID, command="echo hi"),
        _fake_conversation(),
    )

    assert observation.returncode == 0
    assert observation.output is not None
    assert "hi" in observation.output


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
