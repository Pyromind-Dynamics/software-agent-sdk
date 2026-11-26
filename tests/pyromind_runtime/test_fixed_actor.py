import base64
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from harness_adapter.pi_adapter.actor_workspace import ActorTransport
from harness_adapter.pi_adapter.business_tool_host import ToolExecutionContext
from harness_adapter.pi_adapter.fixed_actor import (
    _FORK_SCRIPT,
    FixedActorExecutionManager,
)
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.protocol import FixedActorUnavailable
from pyromind_common.sdk.workspace.models import CommandResult
from pyromind_runtime.domain.context import RequestContext


def test_fixed_actor_requires_a_complete_trusted_route(monkeypatch):
    url, target = "PYROMIND_SANDBOX_ACTOR_URL", "PYROMIND_SANDBOX_ACTOR_TARGET"
    monkeypatch.delenv(url, raising=False)
    monkeypatch.delenv(target, raising=False)
    assert FixedActorExecutionManager.from_env() is None
    monkeypatch.setenv(target, "dev/actor")
    with pytest.raises(ValueError, match="together"):
        FixedActorExecutionManager.from_env()
    monkeypatch.setenv(url, "https://dev:password@router.test")
    with pytest.raises(ValueError, match="router HTTP"):
        FixedActorExecutionManager.from_env()
    monkeypatch.setenv(url, "http://router.test/")
    assert isinstance(FixedActorExecutionManager.from_env(), FixedActorExecutionManager)


def test_actor_commands_drain_output_and_files_limit_reads(monkeypatch):
    stdout = b"x" * 65536 + "好".encode()
    transport = ActorTransport("http://router.test/actor-proxy", "dev/a")
    seen = []

    def handle_request(_transport, req: httpx.Request) -> httpx.Response:
        assert req.headers["ate-target-actor"] == "dev/a"
        assert req.url.path.startswith("/actor-proxy/")
        if req.url.path.endswith("/files"):
            return httpx.Response(200, content=stdout)
        seen.append(req)
        reply = {
            "run_id": "run",
            "status": "running",
            "stdout": "",
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "stdout_offset": 0,
            "stderr_offset": 0,
        }
        if req.method == "POST":
            body = json.loads(req.content)
            assert body["command"] == ["/bin/bash", "-c", "printf hello"]
            assert body["cwd"] == "/workspace/test"
        else:
            offset = int(req.url.params["stdout_offset"])
            data = stdout[offset : offset + 65536]
            reply.update(
                status="succeeded",
                exit_code=0,
                stdout=base64.b64encode(data).decode(),
                stdout_offset=offset + len(data),
            )
        return httpx.Response(200, json=reply)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    result = transport.execute("printf hello", "/workspace/test", 20)
    assert result.stdout == stdout.decode()
    assert result.exit_code == 0
    assert [req.method for req in seen] == ["POST", "GET", "GET"]
    assert transport.read_file("/workspace/data", max_bytes=len(stdout)) == stdout
    with pytest.raises(ValueError, match="size limit"):
        transport.read_file("/workspace/data", max_bytes=len(stdout) - 1)


async def test_fixed_actor_stages_once_and_preserves_its_external_lifetime(
    tmp_path: Path, monkeypatch
):
    context = ToolExecutionContext(
        conversation_id="conv-a",
        workspace_root=tmp_path,
        request_context=RequestContext(user_id="u"),
        model_configuration={},
        extra={},
    )
    files = PiSessionFiles(tmp_path)
    files.save_sandbox({"sandbox_id": "old-platform-instance"})
    files.save_pending_sandbox_staging(
        source_conversation_id="conv-source", workflow_dsl="echo test"
    )
    transport = ActorTransport("http://router.test", "dev/a")
    manager = FixedActorExecutionManager(transport)
    commands = []
    writes = {}
    interpreter_exists = True

    def execute(_transport, command, cwd, timeout):
        commands.append((command, cwd))
        return CommandResult(
            command=command,
            stdout="",
            stderr="",
            exit_code=int(command.startswith("test -x ") and not interpreter_exists),
            timeout_occurred=False,
        )

    monkeypatch.setattr(ActorTransport, "execute", execute)
    monkeypatch.setattr(ActorTransport, "ready", lambda self: None)
    monkeypatch.setattr(
        ActorTransport,
        "write_file",
        lambda self, path, content: writes.update({path: content}),
    )
    with (
        patch(
            "harness_adapter.pi_adapter.fixed_actor.prepare_actor_resources",
            side_effect=RuntimeError("mount unavailable"),
        ),
        pytest.raises(FixedActorUnavailable, match="mount unavailable"),
    ):
        await manager.ensure(context, files)
    assert files.load_pending_sandbox_staging()
    assert manager.recorded_workspace("conv-a", files) is None

    endpoint = await manager.ensure(context, files)
    count = len(commands)
    assert await manager.ensure(context, files) == endpoint
    assert len(commands) == count
    assert any("conv-source" in script and "cp -a" in script for script, _ in commands)
    path = "/workspace/.pyromind-agent/conv-a"
    assert writes[f"{path}/public_data/workflow_canvas/workflow.py"] == b"echo test"
    assert endpoint["workspace_path"] == path
    assert files.load_pending_sandbox_staging() is None
    with pytest.raises(FixedActorUnavailable, match="cannot be replaced"):
        await manager.ensure(context, files, replace=True)
    with patch(
        "harness_adapter.pi_adapter.fixed_actor.prepare_actor_sdk",
        return_value="/tmp/test-sdk/bin/python",
    ) as prepare:
        interpreter = await manager.sdk_python(context, files)
        assert await manager.sdk_python(context, files) == interpreter
        assert prepare.call_count == 1
        interpreter_exists = False
        assert await manager.sdk_python(context, files) == interpreter
        assert prepare.call_count == 2
    count = len(commands)
    await manager.pause(context, files)
    assert manager.recorded_workspace("conv-a", files)
    await manager.delete(context, files)
    restored = FixedActorExecutionManager(transport)
    assert restored.recorded_workspace("conv-a", files)
    for route in (
        ActorTransport("http://router.test", "dev/b"),
        ActorTransport("http://other-router.test", "dev/a"),
    ):
        other = FixedActorExecutionManager(route)
        assert other.recorded_workspace("conv-a", files) is None
        with pytest.raises(FixedActorUnavailable, match="already bound"):
            await other.ensure(context, files)
    assert restored.recorded_workspace("conv-a", files)
    assert files.load_sandbox() == {"sandbox_id": "old-platform-instance"}
    assert len(commands) == count


def test_retrying_an_actor_fork_reuses_only_the_recorded_target(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    target.mkdir()
    data = target / "data.txt"
    data.write_text("existing user data")
    sibling = target.with_suffix(".fork")
    sibling.mkdir()
    receipt = f"{source} -> {target}"
    argv = ["/bin/bash", "-c", _FORK_SCRIPT, "actor-fork", source, target, receipt]
    assert subprocess.run(argv, check=False).returncode == 43
    (target / ".pyromind-actor-fork").write_text(receipt)
    assert subprocess.run(argv, check=False).returncode == 0
    assert data.read_text() == "existing user data"
    assert sibling.is_dir()
