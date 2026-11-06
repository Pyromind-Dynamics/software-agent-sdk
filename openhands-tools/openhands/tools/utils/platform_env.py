"""Resolve platform routing from the deployment environment."""

from __future__ import annotations

import os
from urllib.parse import urlsplit


_PROD_APP_ENVS = {"prod", "production", "online"}


def configured_platform_env() -> str | None:
    app_env = os.getenv("APP_ENV", "").strip().lower()
    if not app_env:
        return None
    if app_env in _PROD_APP_ENVS:
        return "prod"
    return "pre2" if app_env == "pre2" else "pre"


def resolve_platform_env(env: str | None = None) -> str:
    """Prefer APP_ENV, with an explicit fallback for standalone callers."""
    return configured_platform_env() or (env or "").strip().lower() or "pre"


def platform_cluster_route(route: str | None) -> str | None:
    """Keep the region while aligning an outgoing cluster header with APP_ENV."""
    env = configured_platform_env()
    if not route or env is None:
        return route
    cluster = route.partition("#")[0].strip()
    if not cluster:
        return None
    return cluster if env == "prod" else f"{cluster}#{env}"


def platform_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        name: platform_cluster_route(value) or ""
        if name.lower() == "x-cluster"
        else value
        for name, value in headers.items()
    }


def platform_endpoint(prod: str, pre: str, env: str | None = None) -> str:
    resolved = resolve_platform_env(env)
    if resolved in _PROD_APP_ENVS:
        return prod
    return (
        pre.replace("https://pre-", "https://pre2-", 1) if resolved == "pre2" else pre
    )


def platform_portal_url(url: str) -> str:
    """Align configured portal URLs without rewriting custom endpoints."""
    env = configured_platform_env()
    if env is None:
        return url
    parsed = urlsplit(url)
    domain = "api-portal.pyromind.ai"
    if parsed.netloc not in {domain, f"pre-{domain}", f"pre2-{domain}"}:
        return url
    return parsed._replace(
        netloc=domain if env == "prod" else f"{env}-{domain}"
    ).geturl()
