"""Sandbox reuse policy tests for the frozen EDP runner.

The runner executes on platform nodes against the pod_runtime shim (the real
openhands distributions need Python >= 3.12, nodes run 3.10). Register the
shim modules under their dotted names, then load the frozen runner by path.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


_REPO = Path(__file__).resolve().parents[3]
_EDP = _REPO / ".agents" / "skills" / "data-processing" / "scripts" / "edp"
_POD_RUNTIME = _EDP / "pod_runtime"
_RUNNER = _EDP / "sandbox_runner.py"

_SHIM_MODULES = {
    "openhands.sdk.profiles.processing_profile": (
        "openhands/sdk/profiles/processing_profile.py"
    ),
    "openhands.tools.sandbox": "openhands/tools/sandbox/__init__.py",
}


def _load_shim(dotted: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(dotted, _POD_RUNTIME / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


def _load_runner() -> ModuleType:
    for dotted, relative in _SHIM_MODULES.items():
        if dotted not in sys.modules:
            _load_shim(dotted, relative)
    spec = importlib.util.spec_from_file_location("edp_sandbox_runner", _RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ExecResult:
    def __init__(self, returncode: int, output: str = "") -> None:
        self.returncode = returncode
        self.output = output


class _SandboxInfo:
    def __init__(self, sandbox_id: str) -> None:
        self.id = sandbox_id


class FakeClient:
    """exec_command pops queued return codes; an empty queue succeeds."""

    def __init__(self, exit_codes: list[int] | None = None) -> None:
        self.created = 0
        self.deleted: list[str] = []
        self.exit_codes = deque(exit_codes or [])

    def create(self, request: Any) -> _SandboxInfo:
        self.created += 1
        return _SandboxInfo(f"sb-{self.created:03d}")

    def wait_for_sandbox_status(
        self, sandbox_id: str, *, target_status: str, timeout: int
    ) -> bool:
        return True

    def exec_command(self, sandbox_id: str, command: str, timeout: int) -> _ExecResult:
        if self.exit_codes:
            return _ExecResult(self.exit_codes.popleft())
        return _ExecResult(0)

    def write_file(self, sandbox_id: str, path: str, content: bytes) -> None:
        return None

    def read_file(self, sandbox_id: str, path: str) -> bytes:
        raise FileNotFoundError(path)

    def pause(self, sandbox_id: str) -> None:
        return None

    def delete(self, sandbox_id: str) -> None:
        self.deleted.append(sandbox_id)


@pytest.fixture
def runner() -> ModuleType:
    return _load_runner()


def _profile(runner: ModuleType, reuse: str) -> Any:
    return runner.ProcessingProfile.model_validate(
        {
            "schema_version": 1,
            "name": "reuse-test",
            "steps": [
                {
                    "name": "create_sandbox",
                    "params": {"image": "img:1", "wait_timeout": 1},
                },
                {"name": "probe", "params": {"command": "true", "timeout": 5}},
                {"name": "exec", "params": {"command": "run {task_id}", "timeout": 60}},
            ],
            "verdict": {"kind": "exit_code", "success_codes": [0]},
            "execution": {"sandbox_reuse": reuse},
        }
    )


def _records(count: int) -> list[dict[str, Any]]:
    return [{"task_id": f"e{i}", "image": "img:1"} for i in range(count)]


def test_per_shard_reuses_one_sandbox(runner: ModuleType, tmp_path: Path) -> None:
    profile = _profile(runner, "per_shard")
    client = FakeClient()

    summary = runner.run_batch(profile, _records(3), tmp_path, client)

    assert summary.usable == 3
    assert client.created == 1
    assert len(client.deleted) == 1  # finalizer released the shared sandbox


def test_per_record_keeps_isolation(runner: ModuleType, tmp_path: Path) -> None:
    profile = _profile(runner, "per_record")
    client = FakeClient()

    runner.run_batch(profile, _records(3), tmp_path, client)

    assert client.created == 3
    assert len(client.deleted) == 3


def test_exec_failure_keeps_shared_sandbox_alive(
    runner: ModuleType, tmp_path: Path
) -> None:
    profile = _profile(runner, "per_shard")
    client = FakeClient(exit_codes=[0, 1])  # record 1: probe ok, exec fails

    summary = runner.run_batch(profile, _records(3), tmp_path, client)

    assert summary.error == 1
    assert summary.usable == 2
    assert client.created == 1  # exec failure must not recreate


def test_probe_failure_recreates_for_next_record(
    runner: ModuleType, tmp_path: Path
) -> None:
    profile = _profile(runner, "per_shard")
    client = FakeClient(exit_codes=[1, 0])  # record 1: probe fails, wedge cleared

    summary = runner.run_batch(profile, _records(2), tmp_path, client)

    assert client.created == 2
    assert summary.error == 1
    assert summary.usable == 1
    assert len(client.deleted) == 1


def test_default_policy_is_per_record(runner: ModuleType) -> None:
    profile = runner.ProcessingProfile.model_validate(
        {
            "schema_version": 1,
            "name": "default",
            "steps": [
                {"name": "create_sandbox", "params": {"image": "img:1"}},
                {"name": "exec", "params": {"command": "true"}},
            ],
            "verdict": {"kind": "exit_code", "success_codes": [0]},
        }
    )

    assert profile.execution.sandbox_reuse == "per_record"
