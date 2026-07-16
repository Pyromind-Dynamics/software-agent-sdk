"""Real end-to-end test of the debug webhook over an actual HTTP hop.

Everything in ``tests/tools/pyromind_debug/test_debug_workflow.py`` drives
:class:`DebugWorkflowExecutor` and :class:`DebugResultBroker` directly, in
the same process, with no network involved. That leaves one important thing
unverified: that ``POST /api/pyromind/debug/callback`` is actually reachable
and wired correctly (route, auth-exemption, request schema) when hit for
real -- which is exactly what an external debug platform would do.

This test starts a real ``uvicorn`` server (the actual FastAPI app built by
``create_app``) on a free port and points a real ``MockDebugPlatform`` at
it, so the "fail twice, then pass" loop is driven entirely through genuine
HTTP requests into the live webhook route.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
import uvicorn

from openhands.agent_server.api import create_app
from openhands.agent_server.config import Config
from openhands.sdk.conversation.impl.local_conversation import LocalConversation
from openhands.tools.pyromind_debug.definition import DebugWorkflowAction
from openhands.tools.pyromind_debug.impl import DebugWorkflowExecutor
from openhands.tools.pyromind_debug.mock_platform import MockDebugPlatform
from openhands.workspace.docker.workspace import find_available_tcp_port


@contextmanager
def _live_webhook_server() -> Generator[str]:
    """Start the real agent-server app on a free loopback port."""
    config = Config(session_api_keys=[])  # irrelevant to the webhook route,
    # but keeps the rest of the app's routes usable with no auth ceremony.
    app = create_app(config)

    port = find_available_tcp_port()
    server_config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(server_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    ready = False
    for _ in range(50):
        try:
            with httpx.Client() as client:
                response = client.get(f"{base_url}/health", timeout=2.0)
                if response.status_code == 200:
                    ready = True
                    break
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.1)
    if not ready:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Live agent-server failed to start within timeout")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _fake_conversation(working_dir) -> LocalConversation:
    return cast(
        LocalConversation,
        SimpleNamespace(workspace=SimpleNamespace(working_dir=str(working_dir))),
    )


def test_debug_callback_webhook_is_reachable_without_auth():
    """The webhook must not 401 -- it's called by an external platform, not
    a logged-in user, so it must sit outside check_session_api_key."""
    with _live_webhook_server() as base_url:
        with httpx.Client() as client:
            response = client.post(
                f"{base_url}/api/pyromind/debug/callback",
                json={"task_id": "does-not-exist", "status": "passed"},
                timeout=5.0,
            )
    # 404 (unknown task) is expected and fine; 401/403 would mean the
    # auth-exemption regressed.
    assert response.status_code == 404, response.text


def test_mock_platform_fail_fail_pass_over_real_http(tmp_path):
    """The core debug-loop scenario, driven end-to-end through a real HTTP
    callback into the live webhook route rather than an in-process shortcut.
    """
    (tmp_path / "public_data" / "workflow_canvas").mkdir(parents=True, exist_ok=True)
    (tmp_path / "public_data" / "workflow_canvas" / "workflow.py").write_text(
        "# workflow: demo\n", encoding="utf-8"
    )
    conversation = _fake_conversation(tmp_path)

    with _live_webhook_server() as base_url:
        platform = MockDebugPlatform(
            delay_seconds=0.05, fail_attempts=2, callback_base_url=base_url
        )
        executor = DebugWorkflowExecutor(
            max_attempts=10, timeout_seconds=5, platform=platform
        )

        statuses = []
        error_logs = []
        for _ in range(3):
            obs = executor(DebugWorkflowAction(), conversation=conversation)
            statuses.append(obs.status)
            error_logs.append(obs.error_log)

    assert statuses == ["failed", "failed", "passed"]
    assert error_logs[0] is not None
    assert error_logs[1] is not None
    assert error_logs[2] is None


@pytest.mark.parametrize("wrong_task_status", ["passed", "failed"])
def test_unknown_task_callback_returns_404_over_real_http(wrong_task_status):
    with _live_webhook_server() as base_url:
        with httpx.Client() as client:
            response = client.post(
                f"{base_url}/api/pyromind/debug/callback",
                json={"task_id": "unregistered-task-id", "status": wrong_task_status},
                timeout=5.0,
            )
    assert response.status_code == 404
