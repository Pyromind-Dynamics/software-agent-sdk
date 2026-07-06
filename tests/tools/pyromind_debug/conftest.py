"""Shared fixture: a real, minimal HTTP server exposing just the debug
webhook route, for tests that exercise ``MockDebugPlatform`` for real
(rather than through a synchronous stub). Keeps these tool-level tests
independent of a full agent-server app (conversation service, config, etc.)
while still proving the HTTP hop actually works -- see
``tests/agent_server/test_pyromind_debug_webhook_live.py`` for the fuller
version against the real app.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from openhands.agent_server.pyromind_router import pyromind_debug_webhook_router
from openhands.workspace.docker.workspace import find_available_tcp_port


@pytest.fixture
def webhook_base_url() -> Generator[str]:
    app = FastAPI()
    app.include_router(pyromind_debug_webhook_router)

    port = find_available_tcp_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    ready = False
    for _ in range(50):
        try:
            with httpx.Client() as client:
                # Any response (even 404/422) means the server is up.
                client.post(f"{base_url}/api/pyromind/debug/callback", timeout=1.0)
                ready = True
                break
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        time.sleep(0.1)
    if not ready:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("Local debug-webhook server failed to start in time")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)
