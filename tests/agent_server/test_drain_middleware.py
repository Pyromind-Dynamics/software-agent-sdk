"""Tests for the draining middleware that 503s HTTP requests on shutdown."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

import openhands.agent_server.drain as drain_mod
from openhands.agent_server.drain import RESTART_REASON
from openhands.agent_server.middleware import DrainingMiddleware


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    app.add_middleware(DrainingMiddleware)
    return app


def test_http_request_passes_through_when_not_draining():
    with TestClient(_make_app()) as client:
        response = client.get("/ping")
        assert response.status_code == 200


def test_http_request_returns_503_with_restart_reason_while_draining():
    app = _make_app()
    drain_mod.mark_draining(app)
    try:
        with TestClient(app) as client:
            response = client.get("/ping")
            assert response.status_code == 503
            assert response.json() == {"detail": RESTART_REASON}
    finally:
        drain_mod.reset_draining(app)


def test_draining_state_is_isolated_per_app():
    app1, app2 = _make_app(), _make_app()
    drain_mod.mark_draining(app1)
    with TestClient(app1) as client1, TestClient(app2) as client2:
        assert client1.get("/ping").status_code == 503
        assert client2.get("/ping").status_code == 200
