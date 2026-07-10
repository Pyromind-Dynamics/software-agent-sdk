from openhands.tools.utils.pyromind_api_client import (
    _build_access_key_request_headers,
    _parse_cookie_pairs,
    _serialize_cookie_pairs,
)


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
