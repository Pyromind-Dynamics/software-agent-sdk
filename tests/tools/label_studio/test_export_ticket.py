"""Tests for the portal-issued export capability ticket."""

import httpx
import pytest

from openhands.tools.label_studio.converter import ConversionError
from openhands.tools.label_studio.export_ticket import PortalExportTicketProvider


class _Response:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _provider(**overrides) -> PortalExportTicketProvider:
    params = {
        "portal_base_url": "https://pre-api-portal.pyromind.ai",
        "headers": {"cookie": "session"},
    }
    params.update(overrides)
    return PortalExportTicketProvider(**params)


def test_fetch_posts_the_project_ref_with_the_callers_credential(monkeypatch):
    calls: list[tuple[str, dict[str, str], dict]] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append((url, headers, json))
        return _Response(
            200,
            {
                "token": "capability-ticket",
                "path": "/.pyromind-agent/label-studio/abc/export/"
                "label_studio_export.json",
                "expires_at": 1790000000,
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    assert _provider().fetch("abc") == "capability-ticket"
    assert calls == [
        (
            "https://pre-api-portal.pyromind.ai/label_studio/export_token",
            {"cookie": "session"},
            {"project_ref": "abc"},
        )
    ]


def test_fetch_forwards_the_cluster(monkeypatch):
    bodies: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        bodies.append(json)
        return _Response(200, {"token": "capability-ticket"})

    monkeypatch.setattr(httpx, "post", fake_post)
    _provider(cluster="us-west-1#pre").fetch("abc")

    assert bodies == [{"project_ref": "abc", "cluster": "us-west-1#pre"}]


def test_fetch_omits_an_empty_cluster(monkeypatch):
    bodies: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        bodies.append(json)
        return _Response(200, {"token": "capability-ticket"})

    monkeypatch.setattr(httpx, "post", fake_post)
    _provider().fetch("abc")

    assert bodies == [{"project_ref": "abc"}]


def test_fetch_reports_a_response_without_a_ticket(monkeypatch):
    """An unticketed project silently skips the push, so fail loudly here instead."""
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Response(200, {}))

    with pytest.raises(ConversionError, match="missing the token"):
        _provider().fetch("abc")


def test_fetch_reports_an_integration_that_is_switched_off(monkeypatch):
    monkeypatch.setattr(
        httpx, "post", lambda *a, **k: _Response(404, {}, text="Not Found")
    )

    with pytest.raises(ConversionError, match="HTTP 404"):
        _provider().fetch("abc")


def test_fetch_names_the_endpoint_that_answered(monkeypatch):
    """A wrong portal_base_url answers 404 like a switched-off integration."""
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _Response(404, {}, text='{"detail": "Not Found"}'),
    )

    with pytest.raises(
        ConversionError, match=r"http://localhost:8000/label_studio/export_token"
    ):
        _provider(portal_base_url="http://localhost:8000").fetch("abc")


def test_fetch_reports_a_rejected_project_ref(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _Response(400, {}, text="unknown cluster"),
    )

    with pytest.raises(ConversionError, match="HTTP 400"):
        _provider().fetch("abc")


def test_fetch_reports_unreachable_portal(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(ConversionError, match="unreachable"):
        _provider().fetch("abc")


@pytest.mark.parametrize("portal_base_url", ["http://host:notaport", "http://[::1"])
def test_fetch_reports_a_malformed_portal_url(portal_base_url):
    """httpx raises InvalidURL, which is not a RequestError, for a bad base URL.

    The real httpx call is the point: the exception has to come back as a
    ConversionError, or it escapes every caller that treats ConversionError as
    the failure contract -- including the create that must not be abandoned.
    """
    with pytest.raises(ConversionError, match="InvalidURL"):
        _provider(portal_base_url=portal_base_url).fetch("abc")


def test_fetch_reports_invalid_json(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _Response(200, ValueError("Expecting value")),
    )

    with pytest.raises(ConversionError, match="invalid JSON"):
        _provider().fetch("abc")
