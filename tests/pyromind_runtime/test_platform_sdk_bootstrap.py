"""Exercise SDK preparation before the first sandbox terminal command."""

from __future__ import annotations

import asyncio
import builtins
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from harness_adapter.pi_adapter import sandbox_runtime
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.sandbox_runtime import (
    SandboxExecutionManager,
    _SandboxState,
)
from harness_adapter.pi_adapter.sdk_environment import (
    SANDBOX_SDK_PYTHON,
    PlatformSDKSetupError,
    platform_sdk_bootstrap_source,
)

from .test_pi_sandbox_runtime import _context, _FakeSandboxClient


@pytest.fixture
def sdk_preparation(tmp_path, monkeypatch):
    client = _FakeSandboxClient()
    state = _SandboxState(
        client=client,
        endpoint={
            "workspace_path": "/workspace/.pyromind-agent/conv-1",
            "sandbox_id": "sbx-1",
            "ws_base_url": "wss://sandbox.example",
            "cluster": client.cluster,
        },
        credential_fingerprint="test",
    )
    manager = SandboxExecutionManager(client_factory=lambda context: client)
    monkeypatch.setattr(manager, "_ensure_state", AsyncMock(return_value=state))
    upload = Mock()
    monkeypatch.setattr(sandbox_runtime, "write_sandbox_file", upload)
    run = AsyncMock(return_value=("SDK environment ready", 0, False))
    monkeypatch.setattr(manager, "_run_script", run)
    return (
        manager,
        client,
        state,
        _context(tmp_path),
        PiSessionFiles(tmp_path),
        run,
        upload,
    )


async def test_parallel_terminals_share_sdk_preparation_and_reuse_it(sdk_preparation):
    manager, _, _, context, files, run, upload = sdk_preparation
    outputs = await asyncio.gather(
        *(manager.sdk_python(context, files) for _ in range(3))
    )
    assert outputs == [SANDBOX_SDK_PYTHON] * 3
    run.assert_awaited_once()
    upload.assert_called_once()
    assert await manager.sdk_python(context, files) == SANDBOX_SDK_PYTHON
    run.assert_awaited_once()
    upload.assert_called_once()


@pytest.mark.parametrize(
    ("cluster", "override", "index"),
    [
        ("cn-east-1#pre2", None, "https://mirrors.aliyun.com/pypi/simple/"),
        (
            "cn-east-1#pre2",
            "https://mirror.example/simple",
            "https://mirror.example/simple",
        ),
        ("us-west-2", "https://mirror.example/simple", "https://mirror.example/simple"),
        ("us-west-2", None, "https://global.example/simple"),
    ],
)
async def test_sdk_installation_uses_the_cluster_download_source(
    sdk_preparation, monkeypatch, cluster, override, index
):
    manager, client, state, context, files, run, _ = sdk_preparation
    client.cluster = cluster
    state.endpoint["cluster"] = cluster
    monkeypatch.delenv("PYROMIND_SANDBOX_DATAFLOW_PIP_INDEX_URL", raising=False)
    monkeypatch.setenv("PIP_INDEX_URL", "https://global.example/simple")
    if override is not None:
        monkeypatch.setenv("PYROMIND_SANDBOX_DATAFLOW_PIP_INDEX_URL", override)
    await manager.sdk_python(context, files)
    command = run.await_args.args[3]
    assert index in command
    assert "PIP_INDEX_URL" in command
    assert client.api_key not in command


@pytest.fixture
def bootstrap_execution(tmp_path):
    directory = tmp_path / "sdk"
    versions = {
        "/image/python3.9": (3, 9),
        "/image/python3.11": (3, 11),
    }
    calls = []
    installed = []
    environments = []
    install_failure = []

    def run(command, **options):
        calls.append((command, options))
        if command[1] == "-c":
            if "sys.version_info" in command[2]:
                status = 0 if versions.get(command[0], (0, 0)) >= (3, 10) else 1
            else:
                status = 0 if installed else 1
            return subprocess.CompletedProcess(command, status)
        if command[1:3] == ["-m", "venv"]:
            target = Path(command[-1]) / "bin" / "python"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
            versions[str(target)] = versions[command[0]]
        elif command[1:3] == ["-m", "pip"]:
            environments.append(options["env"])
            options["stdout"].write("pip test installation output\n")
            time.sleep(0.05)
            if install_failure:
                raise subprocess.TimeoutExpired(command, options["timeout"])
            installed.append(True)
        else:
            raise AssertionError(f"Unexpected subprocess {command}")
        return subprocess.CompletedProcess(command, 0)

    class FastHeartbeat:
        def __init__(self):
            self.event = threading.Event()

        def wait(self, timeout):
            return self.event.wait(timeout / 1000)

        def set(self):
            self.event.set()

    imports = {
        "subprocess": SimpleNamespace(
            run=run,
            DEVNULL=subprocess.DEVNULL,
            STDOUT=subprocess.STDOUT,
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
        "sys": SimpleNamespace(executable="/image/python3.9"),
        "shutil": SimpleNamespace(
            which=lambda name: "/image/python3.11" if name == "python3.11" else None
        ),
        "threading": SimpleNamespace(Event=FastHeartbeat, Thread=threading.Thread),
    }
    original_import = builtins.__import__

    def redirect_import(name, *args, **kwargs):
        return (
            imports[name] if name in imports else original_import(name, *args, **kwargs)
        )

    def execute(**options):
        source = platform_sdk_bootstrap_source(sdk_directory=str(directory), **options)
        exec(
            compile(source, "<platform-sdk-bootstrap>", "exec"),
            {"__builtins__": {**vars(builtins), "__import__": redirect_import}},
        )

    return SimpleNamespace(
        execute=execute,
        versions=versions,
        calls=calls,
        environments=environments,
        install_failure=install_failure,
        directory=directory,
    )


def test_bootstrap_selects_compatible_python_and_reuses_the_ready_venv(
    bootstrap_execution, capsys
):
    state = bootstrap_execution
    state.execute()
    venv_calls = [args for args, _ in state.calls if args[1:3] == ["-m", "venv"]]
    assert len(venv_calls) == 1
    assert venv_calls[0][0] == "/image/python3.11"
    state.execute()
    pip_calls = [args for args, _ in state.calls if args[1:3] == ["-m", "pip"]]
    assert len(pip_calls) == 1
    assert capsys.readouterr().out.count("PYROMIND_SDK_SETUP: ready") == 2


@pytest.mark.parametrize("configured", [None, "/image/python3.9"])
def test_bootstrap_rejects_incompatible_python_before_installing(
    bootstrap_execution, capsys, configured
):
    state = bootstrap_execution
    if configured is None:
        state.versions["/image/python3.11"] = (3, 9)
    with pytest.raises(RuntimeError, match="Python >=3.10"):
        state.execute(system_python=configured)
    assert not any(args[1] == "-m" for args, _ in state.calls)
    assert "PYROMIND_SDK_SETUP_ERROR:python" in capsys.readouterr().out


@pytest.mark.parametrize("fails", [False, True])
def test_silent_sdk_installation_emits_heartbeat_and_keeps_its_log(
    bootstrap_execution, capsys, fails
):
    state = bootstrap_execution
    if fails:
        state.install_failure.append(True)
        with pytest.raises(subprocess.TimeoutExpired):
            state.execute(pip_index_url="https://mirror.example/simple")
    else:
        state.execute(pip_index_url="https://mirror.example/simple")
    output = capsys.readouterr().out
    assert output.count("PYROMIND_SDK_SETUP: install") >= 2
    assert state.environments[0]["PIP_INDEX_URL"] == "https://mirror.example/simple"
    pip_args = next(args for args, _ in state.calls if args[1:3] == ["-m", "pip"])
    assert (
        pip_args[pip_args.index("--index-url") + 1] == "https://mirror.example/simple"
    )
    assert pip_args[pip_args.index("--timeout") + 1] == "20"
    assert pip_args[pip_args.index("--retries") + 1] == "2"
    pip_options = next(
        options for args, options in state.calls if args[1:3] == ["-m", "pip"]
    )
    assert 0 < pip_options["timeout"] <= 240
    log = Path(str(state.directory) + ".bootstrap.log")
    assert log.read_text() == "pip test installation output\n"
    assert log.stat().st_mode & 0o777 == 0o600
    if fails:
        assert "PYROMIND_SDK_SETUP_ERROR:install:timeout" in output
        assert "PYROMIND_SDK_SETUP: ready" not in output


async def test_sdk_timeout_reports_its_stage_without_exposing_raw_output(
    sdk_preparation,
):
    manager, _, _, context, files, run, _ = sdk_preparation
    run.return_value = (
        "private-token-do-not-expose\nPYROMIND_SDK_SETUP_ERROR:install:timeout",
        1,
        False,
    )
    with pytest.raises(PlatformSDKSetupError) as caught:
        await manager.sdk_python(context, files)
    assert caught.value.stage == "install"
    assert caught.value.timed_out
    assert "bootstrap.log" in str(caught.value)
    assert "private-token-do-not-expose" not in str(caught.value)
