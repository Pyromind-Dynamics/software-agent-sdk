"""
Unit tests for dependency-based authentication functionality.
Tests the check_session_api_key dependency with multiple session API keys support.
"""

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from openhands.agent_server import dependencies as dependencies_module
from openhands.agent_server.config import Config
from openhands.agent_server.dependencies import (
    check_session_api_key,
    get_current_user_id,
)
from openhands.agent_server.pyromind_auth import CurrentLoginUser


def _make_app(session_api_keys: list[str]) -> FastAPI:
    app = FastAPI()
    app.state.config = Config(
        session_api_keys=session_api_keys,
        enable_pyromind_jwt_auth=False,
    )

    @app.get("/test", dependencies=[Depends(check_session_api_key)])
    async def test_endpoint():
        return {"message": "success"}

    return app


def _make_user_app(session_api_keys: list[str]) -> FastAPI:
    app = FastAPI()
    app.state.config = Config(
        session_api_keys=session_api_keys,
        enable_pyromind_jwt_auth=False,
    )

    @app.get("/test", dependencies=[Depends(check_session_api_key)])
    async def test_endpoint(request: Request):
        current_user = request.state.current_user
        assert isinstance(current_user, CurrentLoginUser)
        return {
            "auth_method": request.state.auth_method,
            "user_id": get_current_user_id(request),
            "username": current_user.username,
            "cookie": current_user.cookie,
            "x_cluster": current_user.x_cluster,
        }

    return app


def test_check_session_api_key_valid():
    client = TestClient(_make_app(["test-key"]), raise_server_exceptions=False)
    assert (
        client.get("/test", headers={"X-Session-API-Key": "test-key"}).status_code
        == 200
    )


def test_check_session_api_key_invalid():
    client = TestClient(_make_app(["test-key"]), raise_server_exceptions=False)
    assert (
        client.get("/test", headers={"X-Session-API-Key": "wrong-key"}).status_code
        == 401
    )


def test_check_session_api_key_missing():
    client = TestClient(_make_app(["test-key"]), raise_server_exceptions=False)
    assert client.get("/test").status_code == 401


def test_check_session_api_key_no_keys_configured():
    """When no keys are configured the endpoint is open."""
    client = TestClient(_make_app([]), raise_server_exceptions=False)
    assert client.get("/test").status_code == 200
    assert (
        client.get("/test", headers={"X-Session-API-Key": "any-key"}).status_code == 200
    )


def test_check_session_api_key_reflects_config_update():
    """Updating app.state.config is reflected immediately; no route re-registration needed."""  # noqa: E501
    app = _make_app(["old-key"])
    client = TestClient(app, raise_server_exceptions=False)

    assert (
        client.get("/test", headers={"X-Session-API-Key": "old-key"}).status_code == 200
    )

    app.state.config = Config(session_api_keys=["new-key"])

    assert (
        client.get("/test", headers={"X-Session-API-Key": "new-key"}).status_code == 200
    )
    assert (
        client.get("/test", headers={"X-Session-API-Key": "old-key"}).status_code == 401
    )


def test_check_session_api_key_accepts_dev_pyromind_headers(monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    client = TestClient(_make_user_app(["test-key"]), raise_server_exceptions=False)

    response = client.get(
        "/test",
        headers={
            "X-Pyromind-Debug-User-Id": "42",
            "X-Pyromind-Debug-User-Name": "debug-user-42",
            "Cookie": "auth_token=session-token; other=value",
            "X-Cluster": "us-west-1#pre",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "auth_method": "pyromind_dev",
        "user_id": "42",
        "username": "debug-user-42",
        "cookie": "auth_token=session-token; other=value",
        "x_cluster": "us-west-1#pre",
    }


def test_check_session_api_key_accepts_dev_pyromind_query_params(monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    client = TestClient(_make_user_app(["test-key"]), raise_server_exceptions=False)

    response = client.get(
        "/test",
        params={
            "pyromind-debug-user-id": "42",
            "pyromind-debug-user-name": "debug-user-42",
        },
        headers={
            "Cookie": "auth_token=session-token; other=value",
            "X-Cluster": "us-west-1#pre",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "auth_method": "pyromind_dev",
        "user_id": "42",
        "username": "debug-user-42",
        "cookie": "auth_token=session-token; other=value",
        "x_cluster": "us-west-1#pre",
    }


def test_check_session_api_key_prefers_dev_pyromind_headers_over_query(monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    client = TestClient(_make_user_app(["test-key"]), raise_server_exceptions=False)

    response = client.get(
        "/test",
        params={
            "pyromind-debug-user-id": "99",
            "pyromind-debug-user-name": "query-user",
        },
        headers={
            "X-Pyromind-Debug-User-Id": "42",
            "X-Pyromind-Debug-User-Name": "header-user",
        },
    )

    assert response.status_code == 200
    assert response.json()["user_id"] == "42"
    assert response.json()["username"] == "header-user"


def test_check_session_api_key_rejects_dev_headers_outside_dev(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    client = TestClient(_make_app(["test-key"]), raise_server_exceptions=False)

    response = client.get(
        "/test",
        headers={"X-Pyromind-Debug-User-Id": "42"},
    )

    assert response.status_code == 401


def test_check_session_api_key_stores_context_for_pyromind_jwt(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")

    def fake_verify_pyromind_jwt_token(config, token):
        if token != "portal-token":
            return None
        return CurrentLoginUser(
            username="portal-user",
            email="portal-user@example.test",
            user_id=42,
        )

    monkeypatch.setattr(
        dependencies_module,
        "verify_pyromind_jwt_token",
        fake_verify_pyromind_jwt_token,
    )
    client = TestClient(_make_user_app(["test-key"]), raise_server_exceptions=False)

    response = client.get(
        "/test",
        headers={
            "Cookie": "auth_token=portal-token; other=value",
            "X-Cluster": "us-west-1#pre",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "auth_method": "pyromind_jwt",
        "user_id": "42",
        "username": "portal-user",
        "cookie": "auth_token=portal-token; other=value",
        "x_cluster": "us-west-1#pre",
    }
