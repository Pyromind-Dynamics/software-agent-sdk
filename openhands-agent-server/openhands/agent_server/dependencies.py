from collections.abc import Mapping
from typing import Any
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyCookie, APIKeyHeader

from openhands.agent_server.bash_service import BashEventService
from openhands.agent_server.config import Config
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.event_service import EventService
from openhands.agent_server.pyromind_auth import (
    LOGIN_REQUIRED_DETAIL,
    PYROMIND_AUTH_COOKIE_NAME,
    CurrentLoginUser,
    add_request_context_to_user,
    get_current_login_user_from_token,
    get_dev_login_user_from_headers,
    parse_auth_token_from_cookie_header,
)


# Cookie name used to authenticate the workspace static-file routes.
# Intentionally distinct from the header name: the cookie is ONLY honored
# by the workspace router (so iframes / <img> can load workspace files),
# and is rejected by every other API endpoint.
WORKSPACE_SESSION_COOKIE_NAME = "oh_workspace_session_key"

_SESSION_API_KEY_HEADER = APIKeyHeader(name="X-Session-API-Key", auto_error=False)
_WORKSPACE_SESSION_COOKIE = APIKeyCookie(
    name=WORKSPACE_SESSION_COOKIE_NAME, auto_error=False
)


def is_session_api_key_valid(config: Config, session_api_key: str | None) -> bool:
    if not config.enable_session_api_key_auth:
        return False
    return bool(config.session_api_keys and session_api_key in config.session_api_keys)


def is_pyromind_jwt_auth_configured(config: Config) -> bool:
    return config.enable_pyromind_jwt_auth is True


def is_auth_configured(config: Config) -> bool:
    return bool(config.session_api_keys) or is_pyromind_jwt_auth_configured(config)


def get_pyromind_jwt_token(
    *,
    cookies: Mapping[str, str],
) -> str | None:
    return cookies.get(PYROMIND_AUTH_COOKIE_NAME)


def get_pyromind_jwt_token_from_request(request: Request) -> str | None:
    return resolve_pyromind_auth_token(
        cookies=request.cookies,
        cookie_header=request.headers.get("cookie"),
    )


def resolve_pyromind_auth_token(
    *,
    cookies: Mapping[str, str] | None = None,
    cookie_header: str | None = None,
) -> str | None:
    """Resolve the Pyromind JWT from parsed cookies or a raw Cookie header."""
    if cookies:
        if token := get_pyromind_jwt_token(cookies=cookies):
            return token
    return parse_auth_token_from_cookie_header(cookie_header)


def verify_pyromind_jwt_token(
    config: Config,
    token: str | None,
) -> CurrentLoginUser | None:
    if config.enable_pyromind_jwt_auth is not True:
        return None
    return get_current_login_user_from_token(token)


def authenticate_request(request: Request, session_api_key: str | None) -> None:
    config: Config = request.app.state.config

    dev_user = get_dev_login_user_from_headers(request.headers)

    if dev_user is not None:
        request.state.auth_method = "pyromind_dev"
        request.state.current_user = dev_user
        return

    if is_session_api_key_valid(config, session_api_key):
        request.state.auth_method = "session_api_key"
        return

    pyromind_token = get_pyromind_jwt_token_from_request(request)
    pyromind_user = verify_pyromind_jwt_token(config, pyromind_token)
    if pyromind_user is not None:
        request.state.auth_method = "pyromind_jwt"
        request.state.current_user = add_request_context_to_user(
            pyromind_user, request.headers
        )
        return

    if not is_auth_configured(config):
        return
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=LOGIN_REQUIRED_DETAIL)


def load_base_env(request: Request):
    def _parse_env_from_x_cluster(_x_cluster: str | None) -> str | None:
        _DEFAULT_ENV = "prod"
        if not _x_cluster:
            return _DEFAULT_ENV
        if "#" not in _x_cluster:
            return _DEFAULT_ENV
        env = _x_cluster.rsplit("#", 1)[-1].strip().lower()
        ## 这里返回的env 可能是pre 也可能是pre2
        return env or _DEFAULT_ENV

    def _resolve_cluster_from_conversation(_x_cluster: str | None) -> str:
        """
        根据 x-cluster 头解析集群信息
        x_cluster 可能的值： us-west-1 us-west-2 us-west-1#pre us-west-1#pre2 us-west-2#pre us-west-2#pre2
        """
        if _x_cluster and "#" in _x_cluster:
            region = _x_cluster.split("#", 1)[0].strip()
            if region:
                return region
        if _x_cluster:
            return _x_cluster
        raise ValueError(f"Can't resolve cluster from x-cluster: {_x_cluster}.")

    if x_cluster := _get_validation_cluster_header(request):
        request.state.env = _parse_env_from_x_cluster(x_cluster)
        request.state.cluster = _resolve_cluster_from_conversation(x_cluster)


def _get_validation_cluster_header(
    http_request: Request,
    extra: dict[str, Any] = {},
) -> str | None:
    current_user = getattr(http_request.state, "current_user", None)
    if isinstance(current_user, CurrentLoginUser) and current_user.x_cluster:
        return current_user.x_cluster
    if extra:
        cluster = extra.get("x_cluster", extra.get("x-cluster"))
        if cluster and isinstance(cluster, str):
            return cluster
    return http_request.headers.get("x-cluster")


def get_current_user_id(request: Request) -> str | None:
    current_user = getattr(request.state, "current_user", None)
    if isinstance(current_user, CurrentLoginUser):
        return str(current_user.user_id)
    return None


def check_session_api_key(
    request: Request,
    session_api_key: str | None = Depends(_SESSION_API_KEY_HEADER),
) -> None:
    """Reject the request unless it matches one configured auth mechanism.

    Reads config from ``request.app.state.config`` at request time so that keys
    or JWT settings delivered via ``POST /api/init`` take effect immediately
    without restarting the server or re-registering routes.
    """
    authenticate_request(request, session_api_key)
    ## 加载环境信息
    load_base_env(request)


def check_workspace_session(
    request: Request,
    header_key: str | None = Depends(_SESSION_API_KEY_HEADER),
    cookie_key: str | None = Depends(_WORKSPACE_SESSION_COOKIE),
) -> None:
    """Auth dependency for the workspace static-file routes.

    Accepts EITHER the standard ``X-Session-API-Key`` header OR the
    ``oh_workspace_session_key`` cookie (minted by
    ``POST /api/auth/workspace-session``).
    The cookie is required because browsers cannot attach custom headers to
    ``<iframe src>`` or ``<img src>`` requests, which is how the canvas
    frontend embeds workspace artifacts. The cookie is deliberately scoped
    to this router only; no other endpoint honors it.
    """
    config: Config = request.app.state.config
    dev_user = get_dev_login_user_from_headers(request.headers)
    if dev_user is not None:
        request.state.auth_method = "pyromind_dev"
        request.state.current_user = dev_user
        return

    for candidate in (header_key, cookie_key):
        if is_session_api_key_valid(config, candidate):
            return

    for candidate in (
        request.cookies.get(PYROMIND_AUTH_COOKIE_NAME),
        cookie_key,
    ):
        pyromind_user = verify_pyromind_jwt_token(config, candidate)
        if pyromind_user is not None:
            request.state.auth_method = "pyromind_jwt"
            request.state.current_user = add_request_context_to_user(
                pyromind_user, request.headers
            )
            return

    if not is_auth_configured(config):
        return
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=LOGIN_REQUIRED_DETAIL)


def get_workspace_session_cookie_value(request: Request) -> str:
    config: Config = request.app.state.config
    session_api_key = request.headers.get("x-session-api-key")
    if is_session_api_key_valid(config, session_api_key):
        return session_api_key or ""

    token = get_pyromind_jwt_token_from_request(request)
    if verify_pyromind_jwt_token(config, token) is not None:
        return token or ""

    return ""


def get_conversation_service(request: Request) -> ConversationService:
    service = getattr(request.app.state, "conversation_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation service is not available",
        )
    return service


def get_bash_event_service(request: Request) -> BashEventService:
    service = getattr(request.app.state, "bash_event_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bash event service is not available",
        )
    return service


async def get_event_service(
    conversation_id: UUID,
    request: Request,
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> EventService:
    user_id = get_current_user_id(request)
    if user_id is None:
        event_service = await conversation_service.get_event_service(conversation_id)
    else:
        event_service = await conversation_service.get_event_service(
            conversation_id,
            user_id=user_id,
        )
    if event_service is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}",
        )
    return event_service
