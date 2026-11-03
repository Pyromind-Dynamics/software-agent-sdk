import time

import jwt
import pytest

from openhands.tools.utils import pyromind_api_client
from openhands.tools.utils.pyromind_api_client import (
    _build_access_key_request_headers,
    _build_sensitive_access_key_request,
    _build_sensitive_data_request_key,
    _parse_cookie_pairs,
    _serialize_cookie_pairs,
)


TEST_SECRET = "test-secret-key-with-at-least-32-bytes"


def test_find_secret_key_prefers_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYROMIND_SENSITIVE_DATA_REQUEST_SECRET", TEST_SECRET)
    monkeypatch.setenv(
        "PYROMIND_AGENT_ACCESS_KEY_REQUEST_SECRET",
        "legacy-secret-that-must-not-win",
    )
    pyromind_api_client._find_secret_key.cache_clear()

    try:
        assert pyromind_api_client._find_secret_key() == TEST_SECRET
    finally:
        pyromind_api_client._find_secret_key.cache_clear()


def test_find_secret_key_supports_legacy_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYROMIND_SENSITIVE_DATA_REQUEST_SECRET", raising=False)
    monkeypatch.setenv("PYROMIND_AGENT_ACCESS_KEY_REQUEST_SECRET", TEST_SECRET)
    pyromind_api_client._find_secret_key.cache_clear()

    try:
        assert pyromind_api_client._find_secret_key() == TEST_SECRET
    finally:
        pyromind_api_client._find_secret_key.cache_clear()


def test_parse_cookie_pairs() -> None:
    assert _parse_cookie_pairs("session=sess-1; auth_token=old") == {
        "session": "sess-1",
        "auth_token": "old",
    }


def test_build_access_key_request_headers_without_origin_cookie() -> None:
    headers = _build_access_key_request_headers({}, auth_token="token-1")

    assert headers["auth_token"] == "token-1"
    assert headers["cookie"] == "auth_token=token-1"
    assert headers["accept"] == "*/*"
    assert headers["content-type"] == "application/json"


def test_build_access_key_request_headers_merges_origin_cookie() -> None:
    headers = _build_access_key_request_headers(
        {
            "cookie": "session=sess-1; auth_token=old-token",
            "x-cluster": "us-west-1#pre",
        },
        auth_token="new-token",
    )

    assert headers["x-cluster"] == "us-west-1#pre"
    assert _parse_cookie_pairs(headers["cookie"]) == {
        "session": "sess-1",
        "auth_token": "new-token",
    }


def test_build_access_key_request_headers_merges_case_insensitive_cookie() -> None:
    headers = _build_access_key_request_headers(
        {"Cookie": "lang=en"},
        auth_token="token-1",
    )

    assert _serialize_cookie_pairs(_parse_cookie_pairs(headers["cookie"])) == (
        "lang=en; auth_token=token-1"
    )


def test_build_sensitive_access_key_request_contains_expected_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pyromind_api_client,
        "_find_secret_key",
        lambda: TEST_SECRET,
    )
    auth_token = jwt.encode({"user_id": 42}, TEST_SECRET, algorithm="HS256")
    issued_at = int(time.time())

    request_key = _build_sensitive_access_key_request(auth_token, now=issued_at)
    payload = jwt.decode(
        request_key,
        TEST_SECRET,
        algorithms=["HS256"],
        options={"verify_exp": False},
    )

    assert payload == {
        "client": "pyromind-server",
        "event": "sensitive_data.find_access_key",
        "exp": issued_at + 60,
        "iat": issued_at,
        "user_id": 42,
    }


def test_build_sensitive_data_request_key_uses_requested_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pyromind_api_client,
        "_find_secret_key",
        lambda: TEST_SECRET,
    )
    auth_token = jwt.encode({"user_id": 42}, TEST_SECRET, algorithm="HS256")

    request_key = _build_sensitive_data_request_key(
        auth_token,
        event="sensitive_data.future_event",
        now=1_700_000_000,
    )
    payload = jwt.decode(
        request_key,
        TEST_SECRET,
        algorithms=["HS256"],
        options={"verify_exp": False},
    )

    assert payload["client"] == "pyromind-server"
    assert payload["event"] == "sensitive_data.future_event"


def test_get_api_key_posts_sensitive_data_request_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pyromind_api_client,
        "_find_secret_key",
        lambda: TEST_SECRET,
    )
    auth_token = jwt.encode({"user_id": 42}, TEST_SECRET, algorithm="HS256")
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200

        def json(self) -> dict[str, object]:
            return {"success": True, "data": {"accessKey": "long-lived-access-key"}}

    def _post(url: str, **kwargs: object) -> _Response:
        captured["url"] = url
        captured.update(kwargs)
        return _Response()

    monkeypatch.setattr(pyromind_api_client.httpx, "post", _post)

    access_key = pyromind_api_client.get_api_key(
        env="pre",
        auth_token=auth_token,
        origin_headers={},
    )

    assert access_key == "long-lived-access-key"
    assert captured["url"] == (
        "https://pre-api-portal.pyromind.ai/account/sensitive_data/find_access_key"
    )
    request_body = captured["json"]
    assert isinstance(request_body, dict)
    assert isinstance(request_body.get("request_key"), str)
