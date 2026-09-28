"""Tests for the portal-issued per-user Label Studio token."""

import httpx
import pytest

from openhands.tools.label_studio.converter import ConversionError
from openhands.tools.label_studio.token_provider import PortalTokenProvider


class _Response:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _provider(**overrides) -> PortalTokenProvider:
    params = {
        "portal_base_url": "https://pre-api-portal.pyromind.ai",
        "headers": {"cookie": "session"},
    }
    params.update(overrides)
    return PortalTokenProvider(**params)


def test_fetch_asks_the_portal_with_the_callers_credential(monkeypatch):
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_get(url, *, headers, timeout):
        calls.append((url, headers))
        return _Response(200, {"token": "user-token"})

    monkeypatch.setattr(httpx, "get", fake_get)

    assert _provider().fetch() == "user-token"
    assert calls == [
        (
            "https://pre-api-portal.pyromind.ai/label_studio/token",
            {"cookie": "session"},
        )
    ]


def test_fetch_reports_a_response_without_a_token(monkeypatch):
    """An empty token would surface later as an unexplained 401 from Label Studio."""
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response(200, {}))

    with pytest.raises(ConversionError, match="missing the token"):
        _provider().fetch()


def test_fetch_reports_http_failures(monkeypatch):
    monkeypatch.setattr(
        httpx, "get", lambda *a, **k: _Response(401, {}, text="Unauthorized")
    )

    with pytest.raises(ConversionError, match="HTTP 401"):
        _provider().fetch()


def test_fetch_names_the_endpoint_that_answered(monkeypatch):
    """A portal_base_url naming the console answers 405 from its static host."""
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **k: _Response(405, {}, text="405 Not Allowed"),
    )

    with pytest.raises(
        ConversionError, match=r"https://console.example.com/label_studio/token"
    ):
        _provider(portal_base_url="https://console.example.com").fetch()


@pytest.mark.parametrize("portal_base_url", ["http://host:notaport", "http://[::1"])
def test_fetch_reports_a_malformed_portal_url(portal_base_url):
    """httpx raises InvalidURL, which is not a RequestError, for a bad base URL."""
    with pytest.raises(ConversionError, match="InvalidURL"):
        _provider(portal_base_url=portal_base_url).fetch()


def test_fetch_reports_unreachable_portal(monkeypatch):
    def fake_get(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(ConversionError, match="unreachable"):
        _provider().fetch()
