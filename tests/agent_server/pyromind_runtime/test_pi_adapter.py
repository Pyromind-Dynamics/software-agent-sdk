from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pyromind_runtime.adapters.pi import (
    PiAdapter,
    PiRunnerLaunch,
    StaticPiRunnerLauncher,
    safe_runner_environment,
)
from pyromind_runtime.contracts import (
    HarnessEvent,
    ModelProfile,
    SessionSpec,
    TextContentBlock,
    UserMessageCommand,
)
from pyromind_runtime.contracts.sandbox import SandboxRef, WorkspaceRef


_FAKE_RUNNER = Path(__file__).with_name("fake_pi_runner.py")


def _spec(tmp_path: Path) -> SessionSpec:
    return SessionSpec(
        product_session_id="session-1",
        user_id="user-1",
        workspace=WorkspaceRef(workspace_id="workspace-1", root=str(tmp_path)),
        sandbox=SandboxRef(sandbox_id="sandbox-1", backend="fake"),
        model_profile=ModelProfile(profile_id="test"),
    )


def _launcher() -> StaticPiRunnerLauncher:
    return StaticPiRunnerLauncher(
        (sys.executable, str(_FAKE_RUNNER)),
        environment=safe_runner_environment(
            {
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "OPENAI_API_KEY": "must-not-leak",
            }
        ),
    )


async def _types(
    events: AsyncIterator[HarnessEvent],
    count: int,
) -> list[HarnessEvent]:
    return [await asyncio.wait_for(anext(events), timeout=2) for _ in range(count)]


async def test_pi_adapter_streams_neutral_events_and_capabilities(tmp_path) -> None:
    adapter = PiAdapter(_launcher())
    descriptor = await adapter.describe()
    assert descriptor.capabilities.model_dump() == {
        "resume": False,
        "steer": False,
        "cancel": True,
        "permission_reply": False,
        "partial_message": True,
        "custom_tools": True,
        "fork": False,
        "native_workspace_tools": frozenset({"read", "write", "edit", "bash"}),
    }
    handle = await adapter.create_session(_spec(tmp_path))
    events = adapter.subscribe(handle.session_id)

    await adapter.send(
        handle.session_id,
        UserMessageCommand(
            command_id="run-1",
            content=(TextContentBlock(text="hello"),),
        ),
    )
    output = await _types(events, 8)
    assert [event.type for event in output] == [
        "run.started",
        "message.started",
        "message.completed",
        "message.started",
        "message.delta",
        "message.completed",
        "usage.updated",
        "run.completed",
    ]
    assert {event.run_id for event in output} == {"run-1"}
    assert all(not event.provider_metadata for event in output)

    await adapter.close(handle.session_id)
    with pytest.raises(StopAsyncIteration):
        await anext(events)


async def test_pi_adapter_restarts_after_abnormal_exit(tmp_path) -> None:
    cleanup_count = 0

    class Launcher:
        async def prepare(self, spec: SessionSpec) -> PiRunnerLaunch:
            nonlocal cleanup_count

            async def cleanup() -> None:
                nonlocal cleanup_count
                cleanup_count += 1

            return PiRunnerLaunch(
                command=(sys.executable, str(_FAKE_RUNNER)),
                environment=safe_runner_environment({"PATH": "/usr/bin:/bin"}),
                cleanup=cleanup,
            )

    adapter = PiAdapter(Launcher())
    handle = await adapter.create_session(_spec(tmp_path))
    events = adapter.subscribe(handle.session_id)
    await adapter.send(
        handle.session_id,
        UserMessageCommand(
            command_id="run-crash",
            content=(TextContentBlock(text="crash"),),
        ),
    )
    failed = (await _types(events, 1))[0]
    assert failed.type == "run.failed"
    assert failed.payload["error_code"] == "runner_exited"

    await adapter.send(
        handle.session_id,
        UserMessageCommand(
            command_id="run-after-crash",
            content=(TextContentBlock(text="hello"),),
        ),
    )
    restarted = await _types(events, 8)
    assert restarted[0].type == "run.started"
    assert restarted[-1].type == "run.completed"
    assert cleanup_count == 1

    await adapter.close(handle.session_id)
    assert cleanup_count == 2


async def test_pi_adapter_cancel_maps_to_cancelled_completion(tmp_path) -> None:
    adapter = PiAdapter(_launcher())
    handle = await adapter.create_session(_spec(tmp_path))
    events = adapter.subscribe(handle.session_id)
    await adapter.send(
        handle.session_id,
        UserMessageCommand(
            command_id="run-wait",
            content=(TextContentBlock(text="wait"),),
        ),
    )
    started = await _types(events, 4)
    assert [event.type for event in started] == [
        "run.started",
        "message.started",
        "message.completed",
        "message.started",
    ]

    await adapter.cancel(handle.session_id)
    completed = (await _types(events, 1))[0]
    assert completed.type == "run.completed"
    assert completed.payload == {"outcome": "cancelled"}
    await adapter.close(handle.session_id)


def test_safe_runner_environment_does_not_inherit_credentials() -> None:
    assert safe_runner_environment(
        {"PATH": "/bin", "OPENAI_API_KEY": "secret", "CUSTOM": "value"}
    ) == {"PATH": "/bin"}
