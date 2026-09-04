"""Tests for the frozen sandbox_runner.py skill script (edp paradigm)."""

import importlib.util
import logging
import shlex
import sys
import types
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / ".agents" / "skills" / "data-processing" / "scripts" / "edp"
RUNNER_PATH = SCRIPTS_DIR / "sandbox_runner.py"
PROFILE_PATH = (
    SCRIPTS_DIR
    / "pod_runtime"
    / "openhands"
    / "sdk"
    / "profiles"
    / "processing_profile.py"
)

_SHIM_PACKAGES = (
    "openhands",
    "openhands.sdk",
    "openhands.sdk.profiles",
    "openhands.tools",
)
_TRANSIENT_MODULES = (
    *_SHIM_PACKAGES,
    "openhands.tools.sandbox",
    "openhands.sdk.profiles.processing_profile",
    "sandbox_runner_under_test",
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: module-level dataclasses resolve their own module
    # through sys.modules while the class body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def runner() -> Any:
    """Import sandbox_runner with a scoped pod-runtime-style openhands shim.

    The frozen runner resolves processing_profile + the sandbox shim through
    the openhands namespace only on the platform (PYTHONPATH=pod_runtime);
    the same names are stubbed for the import and restored afterwards so the
    real openhands packages used by the rest of the suite stay untouched.
    """
    saved = {name: sys.modules.get(name) for name in _TRANSIENT_MODULES}
    try:
        for name in _SHIM_PACKAGES:
            shim = types.ModuleType(name)
            shim.__path__ = []
            sys.modules[name] = shim
        sys.modules["openhands.sdk.profiles.processing_profile"] = _load_module(
            "openhands.sdk.profiles.processing_profile", PROFILE_PATH
        )
        sandbox_shim = types.ModuleType("openhands.tools.sandbox")
        setattr(sandbox_shim, "create_sandbox_api_client", lambda **kwargs: None)
        sys.modules["openhands.tools.sandbox"] = sandbox_shim
        return _load_module("sandbox_runner_under_test", RUNNER_PATH)
    finally:
        for name in _TRANSIENT_MODULES:
            previous = saved.get(name)
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class ExecResult:
    """Scripted exec_command response (avoid dataclass: importlib import mode)."""

    def __init__(self, returncode: int, output: str) -> None:
        self.returncode = returncode
        self.output = output


class FakeClient:
    def __init__(self, script: list[Any] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[tuple[str, int]] = []
        self.deleted: list[str] = []
        self.paused: list[str] = []

    def exec_command(self, sandbox_id: str, command: str, timeout: int = 30) -> Any:
        self.calls.append((command, timeout))
        item = self.script.pop(0) if self.script else ExecResult(0, "")
        if isinstance(item, Exception):
            raise item
        return item

    def write_file(self, sandbox_id: str, path: str, content: bytes) -> None:
        self.calls.append((f"write {path}", 0))

    def delete(self, sandbox_id: str) -> None:
        self.deleted.append(sandbox_id)

    def pause(self, sandbox_id: str) -> None:
        self.paused.append(sandbox_id)


def _install_step(runner: Any) -> Any:
    return runner.ProcessingStep(
        name="install_pi",
        params={
            "env": {
                "LLM_BASE_URL": "http://gateway",
                "LLM_AUTH_TOKEN": "tok",
                "LLM_MODEL": "m",
            }
        },
    )


def _install_record() -> dict[str, Any]:
    return {"task_id": "t1", "workdir": "/home/user"}


def test_exec_marker_skips_when_present(runner: Any) -> None:
    client: Any = FakeClient([ExecResult(0, "yes")])
    step = runner.ProcessingStep(
        name="exec",
        params={"command": "make build", "marker": "/workspace/.built"},
    )
    code, output = runner._exec(step, {}, client, "sb-1")
    assert (code, output) == (0, "")
    assert len(client.calls) == 1
    assert ".built" in client.calls[0][0]


def test_exec_marker_written_on_success_only(runner: Any) -> None:
    step = runner.ProcessingStep(
        name="exec",
        params={"command": "make build", "marker": "/workspace/.built"},
    )
    client: Any = FakeClient(
        [ExecResult(0, "no"), ExecResult(0, "built"), ExecResult(0, "")]
    )
    code, output = runner._exec(step, {}, client, "sb-1")
    assert (code, output) == (0, "built")
    assert client.calls[1][0] == "make build"
    assert client.calls[2][0].startswith("touch /workspace/.built")

    failing: Any = FakeClient([ExecResult(0, "no"), ExecResult(1, "boom")])
    code, output = runner._exec(step, {}, failing, "sb-1")
    assert (code, output) == (1, "boom")
    assert len(failing.calls) == 2


def test_install_pi_skips_when_done_marker_exists(runner: Any) -> None:
    client: Any = FakeClient([ExecResult(0, "yes")])
    runner._install_pi(_install_step(runner), _install_record(), client, "sb-1")
    assert len(client.calls) == 1
    assert ".pi_install.done" in client.calls[0][0]


def test_install_pi_runs_detached_with_exit_file_poll(runner: Any) -> None:
    client: Any = FakeClient(
        [ExecResult(0, "no"), ExecResult(0, "launched"), ExecResult(0, "0")]
    )
    runner._install_pi(_install_step(runner), _install_record(), client, "sb-1")
    launch = client.calls[1][0]
    assert "mkdir /opt/.pi_install.lock" in launch
    assert "nohup" in launch
    assert ".pi_install_exit.txt" in launch
    assert client.calls[2][0].startswith("test -f /workspace/.pi_install_exit.txt")


def test_install_pi_lock_conflict_fails_fast(runner: Any) -> None:
    client: Any = FakeClient(
        [ExecResult(0, "no"), ExecResult(0, "launched"), ExecResult(0, "9")]
    )
    with pytest.raises(runner.StepError, match="already running"):
        runner._install_pi(_install_step(runner), _install_record(), client, "sb-1")


def test_install_pi_failure_tails_install_log(runner: Any) -> None:
    client: Any = FakeClient(
        [
            ExecResult(0, "no"),
            ExecResult(0, "launched"),
            ExecResult(0, "1"),
            ExecResult(0, "npm ERR! missing package"),
        ]
    )
    with pytest.raises(runner.StepError, match="npm ERR! missing package"):
        runner._install_pi(_install_step(runner), _install_record(), client, "sb-1")


def test_poll_exit_file_heartbeats_then_kills_on_timeout(
    runner: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(runner, "HEARTBEAT_SECONDS", 0.0)
    client: Any = FakeClient()
    with caplog.at_level(logging.INFO, logger="sandbox_runner"):
        result = runner._poll_exit_file(
            client,
            "sb-1",
            exit_file="/workspace/.pi_exit.txt",
            timeout=0.05,
            poll_interval=0.01,
            label="pi agent",
            progress_file="/workspace/.pi_trace.jsonl",
            kill_pattern="node_modules/.bin/pi",
        )
    assert result is None
    assert "still running" in caplog.text
    assert any("tail -c" in call[0] for call in client.calls)
    assert any("pkill -9 -f node_modules/.bin/pi" in call[0] for call in client.calls)


def test_run_pi_launch_guards_duplicate_start(runner: Any) -> None:
    client: Any = FakeClient([ExecResult(0, "launched"), ExecResult(0, "0")])
    step = runner.ProcessingStep(
        name="run_pi",
        params={"workdir": "/workspace", "env": {"LLM_MODEL": "m"}},
    )
    record = {"task_id": "t1", "prompt": "solve", "workdir": "/workspace"}
    runner._run_pi(step, record, client, "sb-1")
    launch = client.calls[2][0]
    assert "mkdir /workspace/.pi_run.lock" in launch
    assert "echo 9 > /workspace/.pi_exit.txt" in launch


def test_delete_sandbox_kills_processes_then_pause_retries(runner: Any) -> None:
    client: Any = FakeClient()

    def flaky_delete(sandbox_id: str) -> None:
        client.deleted.append(sandbox_id)
        if len(client.deleted) == 1:
            raise RuntimeError("status is Running, can not delete")

    client.delete = flaky_delete
    runner._delete_sandbox(client, "sb-1")
    assert client.paused == ["sb-1"]
    assert len(client.deleted) == 2
    joined = " ".join(command for command, _ in client.calls)
    assert "pkill -9 -f node_modules/.bin/pi" in joined
    assert f"pkill -9 -f {shlex.quote('npm install')}" in joined
    assert "pkill -9 -f .pi_run.sh" in joined
