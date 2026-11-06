"""Tests for the shared control-plane env fallback used by the edp tools."""

from __future__ import annotations

import pytest

from openhands.tools.utils.platform_env import resolve_platform_env


def test_app_env_overrides_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "pre")
    assert resolve_platform_env("prod") == "pre"


def test_app_env_prod_maps_to_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    for app_env in ("prod", "production", "online", "PROD"):
        monkeypatch.setenv("APP_ENV", app_env)
        assert resolve_platform_env(None) == "prod"


def test_app_env_missing_defaults_to_pre(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    assert resolve_platform_env(None) == "pre"


def test_app_env_non_prod_maps_to_pre(monkeypatch: pytest.MonkeyPatch) -> None:
    for app_env in ("dev", "staging", "pre"):
        monkeypatch.setenv("APP_ENV", app_env)
        assert resolve_platform_env("") == "pre"


def test_blank_and_whitespace_env_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    assert resolve_platform_env("  ") == "prod"
    assert resolve_platform_env("") == "prod"


def test_explicit_env_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    assert resolve_platform_env("  pre2  ") == "pre2"
