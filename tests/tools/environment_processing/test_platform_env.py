"""Tests for request-driven platform routing."""

from __future__ import annotations

import pytest

from openhands.tools.utils.platform_env import (
    platform_cluster_target,
    platform_headers,
    platform_portal_url,
    resolve_platform_env,
)


@pytest.mark.parametrize(
    ("route", "target", "header", "host"),
    [
        (None, ("prod", "us-west-1"), "us-west-1", "api-portal.pyromind.ai"),
        (" ", ("prod", "us-west-1"), "us-west-1", "api-portal.pyromind.ai"),
        ("us-west-2", ("prod", "us-west-2"), "us-west-2", "api-portal.pyromind.ai"),
        (
            "us-west-2#pre",
            ("pre", "us-west-2"),
            "us-west-2#pre",
            "pre-api-portal.pyromind.ai",
        ),
        (
            "us-west-1#pre2",
            ("pre2", "us-west-1"),
            "us-west-1#pre2",
            "pre2-api-portal.pyromind.ai",
        ),
    ],
)
def test_platform_route_uses_request_header(monkeypatch, route, target, header, host):
    monkeypatch.setenv("APP_ENV", "pre2")
    monkeypatch.setenv("PYROMIND_CLUSTER", "unrelated")
    headers = platform_headers({"X-Cluster": route} if route is not None else {})
    assert headers == {"x-cluster": header}
    assert platform_cluster_target(route) == target
    assert platform_portal_url(
        "https://pre-api-portal.pyromind.ai/storage", headers
    ) == (f"https://{host}/storage")
    assert platform_portal_url("https://custom.example/storage", headers) == (
        "https://custom.example/storage"
    )


def test_explicit_env_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    assert resolve_platform_env("  pre2  ") == "pre2"
