"""Minimal sandbox client factory for the CustomCommandCPUNode pod runtime.

Mirrors ``openhands.tools.sandbox.create_sandbox_api_client`` using only
pyromind-sdk + httpx, so the frozen ``sandbox_runner.py`` can run inside the
pod's Python 3.10 without the openhands distributions (they require Python
>= 3.12). Keep behaviour identical: auth_token -> access key -> client.
"""

from __future__ import annotations

from openhands.tools.utils.pyromind_api_client import (
    get_api_key,
    get_pyromind_api_client,
)


def create_sandbox_api_client(
    *,
    env: str | None,
    cluster: str | None,
    auth_token: str | None,
    headers: dict[str, str],
) -> object:
    route = (cluster or "").strip() or "us-west-1"
    if "#" in route:
        env = route.partition("#")[2].strip().lower()
    elif cluster and env in {"pre", "pre2"}:
        route = f"{route}#{env}"
    access_key = get_api_key(
        env=env,
        auth_token=auth_token,
        origin_headers={**headers, "x-cluster": route},
    )
    # Mirror openhands.tools.sandbox.create_sandbox_api_client: return the
    # PyroMindAPIClient; the frozen runner then accesses `.sandboxes` itself.
    return get_pyromind_api_client(
        env=env,
        cluster=route,
        api_key=access_key,
    )
