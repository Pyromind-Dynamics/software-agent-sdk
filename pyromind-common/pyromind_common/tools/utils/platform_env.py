"""Resolve platform destinations from the request cluster route."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit


DEFAULT_PLATFORM_CLUSTER = "us-west-1"
_PROD_APP_ENVS = {"prod", "production", "online"}


def resolve_platform_env(env: str | None = None) -> str:
    return (env or "").strip().lower() or "prod"


def platform_cluster_route(route: str | None) -> str:
    route = (route or "").strip()
    return route if route.partition("#")[0] else DEFAULT_PLATFORM_CLUSTER


def platform_cluster_target(route: str | None) -> tuple[str, str]:
    cluster, _, env = platform_cluster_route(route).partition("#")
    return resolve_platform_env(env), cluster


def cluster_header(headers: Mapping[str, str]) -> str | None:
    return next(
        (
            value.strip()
            for name, value in headers.items()
            if name.lower() == "x-cluster" and value.strip()
        ),
        None,
    )


def platform_client_route(
    headers: Mapping[str, str], *, cluster: str | None, env: str | None
) -> str:
    """Prefer the request route to the standalone client's explicit arguments."""
    if route := cluster_header(headers):
        return route
    route = platform_cluster_route(cluster)
    env = resolve_platform_env(env)
    if (cluster or "").strip() and "#" not in route and env in {"pre", "pre2"}:
        return f"{route}#{env}"
    return route


def platform_headers(
    headers: Mapping[str, str], *, route: str | None = None
) -> dict[str, str]:
    forwarded = {
        name: value for name, value in headers.items() if name.lower() != "x-cluster"
    }
    forwarded["x-cluster"] = platform_cluster_route(route or cluster_header(headers))
    return forwarded


def platform_endpoint(prod: str, pre: str, env: str | None = None) -> str:
    resolved = resolve_platform_env(env)
    if resolved in _PROD_APP_ENVS:
        return prod
    return (
        pre.replace("https://pre-", "https://pre2-", 1) if resolved == "pre2" else pre
    )


def platform_portal_url(url: str, headers: Mapping[str, str]) -> str:
    """Select the portal environment while preserving custom endpoints."""
    route = cluster_header(headers)
    if route is None:
        return url
    env, _ = platform_cluster_target(route)
    parsed = urlsplit(url)
    domain = "api-portal.pyromind.ai"
    if parsed.netloc not in {domain, f"pre-{domain}", f"pre2-{domain}"}:
        return url
    prefix = f"{env}-" if env in {"pre", "pre2"} else ""
    return parsed._replace(netloc=f"{prefix}{domain}").geturl()
