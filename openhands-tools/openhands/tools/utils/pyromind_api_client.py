"""Factory helpers for the Pyromind SDK HTTP client."""

from __future__ import annotations

import base64
import gzip
import json
import os
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import httpx
import jwt


if TYPE_CHECKING:
    from pyromind_sdk.client.client import PyroMindAPIClient


_PROD_ENVS = frozenset({"prod", "production", "online"})

_BASE_DOMAIN = "api-portal.pyromind.ai"
_ACCESS_KEY_URI = "/account/agent/find_access_key"
_API_URI = "/api/v1"
_AGENT_ACCESS_KEY_AUDIENCE = "pyromind-portal/agent-access-key"
_AGENT_ACCESS_KEY_CLIENT = "pyromind-agent"
_AGENT_ACCESS_KEY_EVENT = "agent.find_access_key"
_AGENT_ACCESS_KEY_TOKEN_TYPE = "agent_access_key_request"
_AGENT_ACCESS_KEY_TTL_SECONDS = 60
_AGENT_ACCESS_KEY_SECRET_ENV = "PYROMIND_AGENT_ACCESS_KEY_REQUEST_SECRET"
_DEV_SECRET_KEY = "Kij823420JITRE21i21248cbsxhuexvS"


def _get_domain(env: str | None, api_uri: str) -> str:
    if env in _PROD_ENVS:
        return f"https://{_BASE_DOMAIN}{api_uri}"
    if env == "pre2":
        return f"https://pre2-{_BASE_DOMAIN}{api_uri}"
    if env == "pre":
        return f"https://pre-{_BASE_DOMAIN}{api_uri}"
    return f"https://{_BASE_DOMAIN}{api_uri}"


def _access_key_url(env: str | None) -> str:
    return _get_domain(env, _ACCESS_KEY_URI)


def _api_url(env: str | None) -> str:
    return _get_domain(env, _API_URI)


def _parse_cookie_pairs(cookie_header: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in cookie_header.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        if name:
            cookies[name] = value.strip()
    return cookies


def _serialize_cookie_pairs(cookies: dict[str, str]) -> str:
    return "; ".join(f"{name}={value}" for name, value in cookies.items())


def _extract_cookie_header(
    headers: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    remaining = dict(headers)
    cookies: dict[str, str] = {}
    for key in list(remaining):
        if key.lower() == "cookie":
            cookies.update(_parse_cookie_pairs(remaining.pop(key)))
    return remaining, cookies


@lru_cache
def _find_secret_key() -> str:
    env_secret = os.getenv(_AGENT_ACCESS_KEY_SECRET_ENV, "").strip()
    if env_secret:
        return env_secret
    if os.getenv("APP_ENV", "dev") == "dev":
        return _DEV_SECRET_KEY
    secret_path = os.getenv("PYROMIND_SECRET_PATH", "/etc/secrets")
    try:
        with open(f"{secret_path}/web_secret_key") as secret_file:
            return secret_file.read().strip()
    except FileNotFoundError as exc:
        raise RuntimeError("Secret file not found: web_secret_key") from exc
    except PermissionError as exc:
        raise RuntimeError("Permission denied reading secret: web_secret_key") from exc


def _user_id_from_auth_token(auth_token: str) -> int:
    try:
        payload = jwt.decode(
            auth_token,
            options={"verify_signature": False, "verify_exp": False},
        )
    except jwt.InvalidTokenError as exc:
        raise RuntimeError("Invalid auth token") from exc
    user_id = payload.get("user_id")
    if type(user_id) is not int:
        raise RuntimeError("Auth token is missing user_id")
    return user_id


def _build_agent_access_key_request(
    auth_token: str,
    *,
    now: int | None = None,
) -> str:
    issued_at = int(time.time()) if now is None else now
    return jwt.encode(
        {
            "aud": _AGENT_ACCESS_KEY_AUDIENCE,
            "client": _AGENT_ACCESS_KEY_CLIENT,
            "event": _AGENT_ACCESS_KEY_EVENT,
            "exp": issued_at + _AGENT_ACCESS_KEY_TTL_SECONDS,
            "iat": issued_at,
            "token_type": _AGENT_ACCESS_KEY_TOKEN_TYPE,
            "user_id": _user_id_from_auth_token(auth_token),
        },
        _find_secret_key(),
        algorithm="HS256",
    )


def get_api_key(
    env: str | None,
    auth_token: str,
    origin_headers: dict[str, str],
    timeout: int = 30,
) -> str:
    """Exchange a short-lived agent request for a Pyromind access key.

    Parses the platform response shape ``{"success": true, "data": {"accessKey": ...}}``
    and returns the access key. Raises on HTTP errors or unexpected payloads.

    用短期 agent 请求凭证换取 Pyromind access key。解析平台返回的
    ``{"success": true, "data": {"accessKey": ...}}`` 格式并返回 access key。
    HTTP 错误或响应格式异常时抛出异常。
    """
    # Construct the request
    endpoint_url = _access_key_url(env)

    # 构建请求头
    headers = _build_access_key_request_headers(origin_headers, auth_token=auth_token)

    # 发送请求
    response = httpx.post(
        endpoint_url,
        headers=headers,
        json={"request_key": _build_agent_access_key_request(auth_token)},
        timeout=timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Failed to get api key: HTTP {response.status_code}")

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid api key response: expected JSON object")
    if not payload.get("success"):
        message = payload.get("message") or payload.get("error") or "unknown error"
        raise RuntimeError(f"Api key request failed: {message}")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Invalid api key response: missing data object")

    access_key = data.get("accessKey")
    if not access_key:
        raise RuntimeError("Invalid api key response: missing accessKey")

    return str(access_key)


def decompress_gzip_base64_data(value: str) -> Any:
    """Decode a gzip+base64 compressed payload back into JSON data.

    还原 middleware 以 gzip+base64 压缩下发的 JSON 载荷。
    """
    raw = gzip.decompress(base64.b64decode(value))
    return json.loads(raw.decode("utf-8"))


def _build_access_key_request_headers(
    origin_headers: dict[str, str],
    *,
    auth_token: str,
) -> dict[str, str]:
    base_headers, existing_cookies = _extract_cookie_header(origin_headers)
    cookies = {**existing_cookies, "auth_token": auth_token}
    return {
        "accept": "*/*",
        "content-type": "application/json",
        **base_headers,
        "auth_token": auth_token,
        "cookie": _serialize_cookie_pairs(cookies),
    }


def get_pyromind_api_client(
    *,
    env: str | None,
    cluster: str | None,
    api_key: str | None,
    timeout: int = 30,
    max_retries: int = 3,
) -> PyroMindAPIClient:
    """Create a :class:`PyroMindAPIClient` from conversation context and environment.

    ``base_url`` is derived from the user's ``x-cluster`` routing header stored on
    the conversation. ``api_key`` and ``cluster`` still fall back to process env
    vars when not yet wired from conversation secrets.

    根据会话上下文与环境变量创建 :class:`PyroMindAPIClient`。

    ``base_url`` 由会话中持久化的 ``x-cluster`` 环境标识解析；``api_key`` 与
    ``cluster`` 暂仍可通过进程环境变量提供。
    """
    base_url = _api_url(env)
    client_type = _load_pyromind_api_client_type()
    return client_type(
        api_key=api_key,
        base_url=base_url,
        cluster=cluster,
        timeout=timeout,
        max_retries=max_retries,
    )


def _load_pyromind_api_client_type() -> type[PyroMindAPIClient]:
    try:
        from pyromind_sdk import PyroMindAPIClient
    except ImportError as exc:
        raise RuntimeError(
            "pyromind-sdk with pyromind_sdk.PyroMindAPIClient is required."
        ) from exc
    return PyroMindAPIClient
