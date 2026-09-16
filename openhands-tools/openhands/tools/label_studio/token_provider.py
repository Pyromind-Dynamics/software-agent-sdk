"""Per-user Label Studio API tokens, issued by the portal.

Label Studio shows an account only the projects of its own organization, and the
portal gives each user their own organization. A token shared by every
conversation would therefore file each user's projects under a single account,
where the user who asked for them could not see them. The portal hands each
caller the token of their own account, over the credential that caller already
authenticated with.
"""

from __future__ import annotations

from typing import Any

import httpx

from openhands.tools.label_studio.converter import ConversionError


TOKEN_ROUTE = "/label_studio/token"


class PortalTokenProvider:
    """Reads the calling user's own Label Studio token from the portal."""

    def __init__(
        self,
        *,
        portal_base_url: str,
        headers: dict[str, str],
        timeout: float = 30.0,
    ) -> None:
        self._portal_base_url = portal_base_url.rstrip("/")
        self._headers = dict(headers)
        self._timeout = timeout

    def fetch(self) -> str:
        # The URL is part of every failure: a portal_base_url that names the
        # console instead of this API answers with the console's SPA shell.
        endpoint = f"{self._portal_base_url}{TOKEN_ROUTE}"
        try:
            response = httpx.get(endpoint, headers=self._headers, timeout=self._timeout)
        # InvalidURL is not a RequestError -- httpx derives it straight from
        # Exception -- so a mistyped portal_base_url would otherwise escape
        # every caller that catches ConversionError.
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            raise ConversionError(
                f"Portal token API {endpoint} unreachable: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise ConversionError(
                f"Portal token API {endpoint} returned HTTP "
                f"{response.status_code}: {response.text[:300]}"
            )
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise ConversionError(
                f"Portal token API returned invalid JSON: {exc}"
            ) from exc
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token.strip():
            raise ConversionError(
                f"Portal token API response is missing the token: {payload}"
            )
        return token.strip()
