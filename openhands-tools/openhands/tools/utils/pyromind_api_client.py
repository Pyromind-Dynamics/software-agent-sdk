"""Factory helpers for the Pyromind SDK HTTP client."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx


if TYPE_CHECKING:
    from pyromind_sdk.client.client import PyroMindAPIClient


_PROD_ENVS = frozenset({"prod", "production", "online"})

_BASE_DOMAIN = "api-portal.pyromind.ai"
_ACCESS_KEY_URI = "/account/find_access_key"
_API_URI = "/api/v1"


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


def get_api_key(
    env: str | None,
    auth_token: str,
    origin_headers: dict[str, str],
    timeout: int = 30,
) -> str:
    """Exchange an auth token for a Pyromind access key.

    Parses the platform response shape ``{"success": true, "data": {"accessKey": ...}}``
    and returns the access key. Raises on HTTP errors or unexpected payloads.

    用 auth token 换取 Pyromind access key。解析平台返回的
    ``{"success": true, "data": {"accessKey": ...}}`` 格式并返回 access key。
    HTTP 错误或响应格式异常时抛出异常。
    """
    ## Construct the request
    endpoint_url = _access_key_url(env)

    ## 构建请求头
    headers = _build_access_key_request_headers(origin_headers, auth_token=auth_token)

    ## 发送请求
    response = httpx.post(
        endpoint_url,
        headers=headers,
        json={},
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
