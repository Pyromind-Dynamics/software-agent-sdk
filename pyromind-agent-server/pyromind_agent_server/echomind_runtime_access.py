from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.config import Config
from openhands.agent_server.echomind_projects import ProjectInfo
from openhands.agent_server.pyromind_auth import (
    get_current_login_user_from_token,
    parse_auth_token_from_cookie_header,
)
from openhands.tools.utils.platform_env import (
    platform_cluster_route,
    platform_cluster_target,
)
from openhands.tools.utils.pyromind_api_client import get_api_key
from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore


logger = logging.getLogger(__name__)
_ACCESS_KEY_REFRESH_SECONDS = 60


def build_runtime_credential_store(
    projects_root: Path, config: Config | None
) -> RuntimeCredentialStore | None:
    if config is None or config.secret_key is None:
        return None
    return RuntimeCredentialStore(
        projects_root.parent / ".echomind-runtime-credentials", cipher=config.cipher
    )


def platform_request_token(context: RequestContext | None) -> str | None:
    if context is None:
        return None
    if token := parse_auth_token_from_cookie_header(context.cookie):
        return token
    scheme, _, value = (context.authorization or "").partition(" ")
    return (value.strip() or None) if scheme.lower() == "bearer" else None


def refresh_runtime_access(
    store: RuntimeCredentialStore | None,
    project: ProjectInfo,
    context: RequestContext,
    *,
    force: bool = False,
) -> None:
    owner = project.project.owner_user_id
    if store is None or owner is None or owner != context.user_id:
        return
    route = platform_cluster_route(project.project.cluster)
    if context.x_cluster and platform_cluster_route(context.x_cluster) != route:
        return
    token = platform_request_token(context)
    if token is None:
        return
    login = get_current_login_user_from_token(token)
    if login is None or str(login.user_id) != owner:
        return
    platform_user_id = str(login.user_id)
    try:
        if not store.needs_refresh(
            owner,
            route,
            max_age_seconds=0 if force else _ACCESS_KEY_REFRESH_SECONDS,
            platform_user_id=platform_user_id,
        ):
            return
        env, _ = platform_cluster_target(route)
        headers = {"x-cluster": route}
        if context.cookie:
            headers["cookie"] = context.cookie
        if context.authorization:
            headers["authorization"] = context.authorization
        observed_at = datetime.now(UTC)
        access_key = get_api_key(env, token, headers, timeout=5)
        store.save(
            owner,
            route,
            access_key,
            platform_user_id=platform_user_id,
            observed_at=observed_at,
        )
    except (httpx.HTTPError, RuntimeError, ValueError, OSError) as exc:
        logger.warning(
            "Could not refresh EchoMind runtime access for project %s (%s)",
            project.project.project_id,
            type(exc).__name__,
        )
