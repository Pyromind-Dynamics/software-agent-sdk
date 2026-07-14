"""End-to-end test of the debug_workflow tool against the mock platform.

Covers the loop the debug-workflow skill drives: submit -> block -> wake on
callback -> observation -- including the "fail twice, then pass" scenario
and the tool-level guardrails (missing workflow.py, timeout, max attempts).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.sdk.tool.registry import list_registered_tools
from openhands.tools.pyromind_debug.broker import (
    DebugResultBroker,
    get_debug_result_broker,
)
from openhands.tools.pyromind_debug.definition import (
    DebugWorkflowAction,
    DebugWorkflowTool,
)
from openhands.tools.pyromind_debug.impl import DebugWorkflowExecutor
from openhands.tools.pyromind_debug.mock_platform import MockDebugPlatform


def test_legacy_debug_workflow_tool_remains_registered():
    assert DebugWorkflowTool.name in list_registered_tools()


def _fake_conversation(working_dir) -> LocalConversation:
    return cast(
        LocalConversation,
        SimpleNamespace(workspace=SimpleNamespace(working_dir=str(working_dir))),
    )


def _write_workflow(tmp_path) -> None:
    (tmp_path / "workflow.py").write_text("# workflow: demo\n", encoding="utf-8")


class _StubPlatform:
    """Deterministic platform stub: resolves synchronously, no timers."""

    def __init__(self, broker: DebugResultBroker, fail_attempts: int = 2) -> None:
        self._broker = broker
        self._fail_attempts = fail_attempts
        self.submitted_attempts: list[int] = []

    def submit(self, task_id: str, workflow_source: str, attempt: int) -> None:
        del workflow_source
        self.submitted_attempts.append(attempt)
        if attempt <= self._fail_attempts:
            self._broker.resolve(
                task_id, status="failed", error_log=f"boom on attempt {attempt}"
            )
        else:
            self._broker.resolve(task_id, status="passed")


def test_fails_twice_then_passes(tmp_path):
    """The core debug-loop scenario: 2 failures, then a passing 3rd attempt."""
    _write_workflow(tmp_path)
    conversation = _fake_conversation(tmp_path)
    # The executor always resolves against the process-wide singleton broker
    # (see DebugWorkflowExecutor.__call__), so the stub must resolve against
    # that same instance.
    platform = _StubPlatform(get_debug_result_broker(), fail_attempts=2)
    executor = DebugWorkflowExecutor(
        max_attempts=10, timeout_seconds=5, platform=platform
    )

    results = [
        executor(DebugWorkflowAction(), conversation=conversation) for _ in range(3)
    ]

    assert [r.status for r in results] == ["failed", "failed", "passed"]
    assert [r.attempt for r in results] == [1, 2, 3]
    assert all(r.max_attempts == 10 for r in results)
    assert results[0].error_log == "boom on attempt 1"
    assert results[1].error_log == "boom on attempt 2"
    assert results[2].error_log is None
    assert results[0].is_error is False, "a normal failed run is not a tool error"
    assert results[2].is_error is False
    assert platform.submitted_attempts == [1, 2, 3]


def test_mock_platform_end_to_end(tmp_path, webhook_base_url):
    """Same scenario, but through the real MockDebugPlatform + its timer
    thread + a genuine HTTP callback into the live webhook route (see
    ``webhook_base_url`` in conftest.py)."""
    _write_workflow(tmp_path)
    conversation = _fake_conversation(tmp_path)
    platform = MockDebugPlatform(
        delay_seconds=0.05, fail_attempts=2, callback_base_url=webhook_base_url
    )
    executor = DebugWorkflowExecutor(
        max_attempts=10, timeout_seconds=5, platform=platform
    )

    statuses = []
    for _ in range(3):
        obs = executor(DebugWorkflowAction(), conversation=conversation)
        statuses.append(obs.status)

    assert statuses == ["failed", "failed", "passed"]


def test_missing_workflow_file_returns_error(tmp_path):
    conversation = _fake_conversation(tmp_path)
    executor = DebugWorkflowExecutor(platform=MockDebugPlatform(delay_seconds=0.01))

    obs = executor(DebugWorkflowAction(), conversation=conversation)

    assert obs.status == "error"
    assert obs.is_error is True
    assert obs.attempt == 0


def test_no_conversation_returns_error(tmp_path):
    executor = DebugWorkflowExecutor(platform=MockDebugPlatform(delay_seconds=0.01))

    obs = executor(DebugWorkflowAction(), conversation=None)

    assert obs.status == "error"
    assert obs.is_error is True


def test_timeout_when_platform_never_resolves(tmp_path):
    _write_workflow(tmp_path)
    conversation = _fake_conversation(tmp_path)

    class _SilentPlatform:
        def submit(self, task_id: str, workflow_source: str, attempt: int) -> None:
            del task_id, workflow_source, attempt  # never resolves the broker

    executor = DebugWorkflowExecutor(
        max_attempts=10, timeout_seconds=0.05, platform=_SilentPlatform()
    )

    obs = executor(DebugWorkflowAction(), conversation=conversation)

    assert obs.status == "timeout"
    assert obs.is_error is True
    assert obs.attempt == 1


def test_max_attempts_enforced(tmp_path, webhook_base_url):
    _write_workflow(tmp_path)
    conversation = _fake_conversation(tmp_path)
    platform = MockDebugPlatform(
        delay_seconds=0.01, fail_attempts=0, callback_base_url=webhook_base_url
    )
    executor = DebugWorkflowExecutor(
        max_attempts=2, timeout_seconds=5, platform=platform
    )

    first = executor(DebugWorkflowAction(), conversation=conversation)
    second = executor(DebugWorkflowAction(), conversation=conversation)
    third = executor(DebugWorkflowAction(), conversation=conversation)

    assert first.status == "passed"
    assert second.status == "passed"
    assert third.status == "error"
    assert third.is_error is True
    assert third.attempt == 2  # no new attempt was consumed
    assert "maximum" in third.text.lower()


@pytest.mark.parametrize("fail_attempts", [0, 1])
def test_passes_immediately_when_no_failures_configured(
    tmp_path, webhook_base_url, fail_attempts
):
    _write_workflow(tmp_path)
    conversation = _fake_conversation(tmp_path)
    platform = MockDebugPlatform(
        delay_seconds=0.01,
        fail_attempts=fail_attempts,
        callback_base_url=webhook_base_url,
    )
    executor = DebugWorkflowExecutor(platform=platform)

    for _ in range(fail_attempts):
        obs = executor(DebugWorkflowAction(), conversation=conversation)
        assert obs.status == "failed"

    obs = executor(DebugWorkflowAction(), conversation=conversation)
    assert obs.status == "passed"


def test_tool_create_runs_fail_fail_pass_loop(tmp_path, monkeypatch, webhook_base_url):
    """DebugWorkflowTool.create() -> executor -> MockDebugPlatform -> webhook.

    Exercises the legacy debug_workflow implementation directly (registry
    registration is disabled; production uses run_workflow(test_mode=True)).
    """
    import openhands.tools.pyromind_debug.mock_platform as mock_platform_module

    monkeypatch.setattr(mock_platform_module, "DEFAULT_DELAY_SECONDS", 0.05)
    monkeypatch.setattr(mock_platform_module, "DEFAULT_FAIL_ATTEMPTS", 2)
    monkeypatch.setattr(
        mock_platform_module, "DEFAULT_CALLBACK_BASE_URL", webhook_base_url
    )
    _write_workflow(tmp_path)
    conversation = _fake_conversation(tmp_path)

    tools = DebugWorkflowTool.create(callback_base_url=webhook_base_url)
    assert len(tools) == 1
    tool = tools[0]
    assert tool.executor is not None

    statuses = []
    for _ in range(3):
        action = DebugWorkflowAction()
        obs = tool.executor(action, conversation=conversation)
        statuses.append(obs.status)

    assert statuses == ["failed", "failed", "passed"]
