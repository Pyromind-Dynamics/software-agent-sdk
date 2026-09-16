"""Tests for portal media URL signing."""

import httpx
import pytest

from openhands.tools.label_studio.converter import ConversionError
from openhands.tools.label_studio.media_signer import PortalMediaSigner


class _Response:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _signer(**overrides) -> PortalMediaSigner:
    params = {
        "portal_base_url": "https://console.example.com",
        "headers": {"cookie": "session"},
    }
    params.update(overrides)
    return PortalMediaSigner(**params)


def test_sign_many_posts_complete_paths_and_returns_absolute_urls(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    def fake_post(url, *, headers, json, timeout):
        calls.append((url, json["paths"]))
        return _Response(
            200,
            {
                "urls": {
                    path: f"https://console.example.com/label_studio/media?path={path}"
                    for path in json["paths"]
                }
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    urls = _signer(batch_limit=2).sign_many(["a", "b", "c"])

    assert calls[0][0] == "https://console.example.com/label_studio/media-urls"
    assert calls == [
        ("https://console.example.com/label_studio/media-urls", ["a", "b"]),
        ("https://console.example.com/label_studio/media-urls", ["c"]),
    ]
    assert urls["c"].startswith("https://console.example.com/label_studio/media")


def test_sign_many_rejects_a_partial_response(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _Response(200, {"urls": {"a": "https://portal/media"}}),
    )

    with pytest.raises(ConversionError, match="returned no url for 1 of 2"):
        _signer().sign_many(["a", "b"])


def test_sign_many_reports_http_failures(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _Response(403, {}, text="Invalid storage path"),
    )

    with pytest.raises(ConversionError, match="HTTP 403"):
        _signer().sign_many(["a"])


def test_http_failures_name_the_endpoint_that_answered(monkeypatch):
    """A wrong portal_base_url answers 404 like a missing route, so report the URL."""
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _Response(
            404, {"detail": "Not Found"}, text='{"detail": "Not Found"}'
        ),
    )

    with pytest.raises(
        ConversionError, match=r"http://localhost:8000/label_studio/media-urls"
    ):
        _signer(portal_base_url="http://localhost:8000").sign_many(["a"])


def test_sign_many_reports_unreachable_portal(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(ConversionError, match="unreachable"):
        _signer().sign_many(["a"])


@pytest.mark.parametrize("portal_base_url", ["http://host:notaport", "http://[::1"])
def test_sign_many_reports_a_malformed_portal_url(portal_base_url):
    """httpx raises InvalidURL, which is not a RequestError, for a bad base URL."""
    with pytest.raises(ConversionError, match="InvalidURL"):
        _signer(portal_base_url=portal_base_url).sign_many(["a"])


def test_sign_many_forwards_the_cluster(monkeypatch):
    """The portal pins the cluster the agent is routed to into every media URL."""
    bodies: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        bodies.append(json)
        return _Response(
            200,
            {"urls": {path: "https://console/media" for path in json["paths"]}},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    _signer(cluster="us-west-1#pre").sign_many(["a", "b"])

    assert bodies == [{"paths": ["a", "b"], "cluster": "us-west-1#pre"}]


def test_sign_many_omits_an_empty_cluster(monkeypatch):
    bodies: list[dict] = []

    def fake_post(url, *, headers, json, timeout):
        bodies.append(json)
        return _Response(200, {"urls": {"a": "https://console/media"}})

    monkeypatch.setattr(httpx, "post", fake_post)
    _signer().sign_many(["a"])

    assert bodies == [{"paths": ["a"]}]
