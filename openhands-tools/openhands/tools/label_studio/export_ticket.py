"""Export capability tickets, minted by the portal for one project.

The Label Studio deployment's export button has to push the export into the
project owner's Storage, but that deployment serves every user at once, so it
cannot hold a storage credential that would let any one of them write anywhere.
The portal mints a ticket instead: a capability limited to writing that one
export key for that one user, over the credential the caller already
authenticated with. The project description carries it to the button's plugin,
which never sees any other credential.
"""

from __future__ import annotations

from typing import Any

import httpx

from openhands.tools.label_studio.converter import ConversionError


EXPORT_TOKEN_ROUTE = "/label_studio/export_token"


class PortalExportTicketProvider:
    """Mints the calling user's export ticket for one of their own projects."""

    def __init__(
        self,
        *,
        portal_base_url: str,
        headers: dict[str, str],
        cluster: str = "",
        timeout: float = 30.0,
    ) -> None:
        self._portal_base_url = portal_base_url.rstrip("/")
        self._headers = dict(headers)
        self._cluster = cluster
        self._timeout = timeout

    def fetch(self, project_ref: str) -> str:
        body: dict[str, Any] = {"project_ref": project_ref}
        if self._cluster:
            body["cluster"] = self._cluster
        # The URL is part of every failure: pointing portal_base_url at the wrong
        # host answers with a plain 404 that is otherwise indistinguishable from
        # the integration being switched off on the right host.
        endpoint = f"{self._portal_base_url}{EXPORT_TOKEN_ROUTE}"
        try:
            response = httpx.post(
                endpoint,
                headers=self._headers,
                json=body,
                timeout=self._timeout,
            )
        # InvalidURL is not a RequestError -- httpx derives it straight from
        # Exception -- so a mistyped portal_base_url would otherwise escape
        # every caller that catches ConversionError.
        except (httpx.RequestError, httpx.InvalidURL) as exc:
            raise ConversionError(
                f"Portal export-token API {endpoint} unreachable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise ConversionError(
                f"Portal export-token API {endpoint} returned HTTP "
                f"{response.status_code}: {response.text[:300]}"
            )
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise ConversionError(
                f"Portal export-token API {endpoint} returned invalid JSON: {exc}"
            ) from exc
        token = payload.get("token") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token.strip():
            raise ConversionError(
                f"Portal export-token API {endpoint} response is missing the "
                f"token: {payload}"
            )
        return token.strip()
