from fastapi import HTTPException
from pyromind_agent_server.api.echomind_projects import (
    _require_sandbox_credentials,
)
from starlette.requests import Request


def _request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [
                (name.lower().encode(), value.encode())
                for name, value in headers.items()
            ],
        }
    )


def test_dataset_start_requires_sandbox_credentials(monkeypatch) -> None:
    monkeypatch.setenv("PYROMIND_HARNESS_BACKEND", "pi")
    monkeypatch.setenv("PYROMIND_PI_TERMINAL_BACKEND", "sandbox")

    try:
        _require_sandbox_credentials(_request({"x-cluster": "us-west-1#pre"}))
    except HTTPException as exc:
        assert exc.status_code == 409
        assert "auth_token" in str(exc.detail)
    else:
        raise AssertionError("missing sandbox credentials must fail before dispatch")


def test_dataset_start_accepts_auth_token_cookie(monkeypatch) -> None:
    monkeypatch.setenv("PYROMIND_HARNESS_BACKEND", "pi")
    monkeypatch.setenv("PYROMIND_PI_TERMINAL_BACKEND", "sandbox")

    _require_sandbox_credentials(
        _request(
            {
                "cookie": "auth_token=test-token",
                "x-cluster": "us-west-1#pre",
            }
        )
    )
