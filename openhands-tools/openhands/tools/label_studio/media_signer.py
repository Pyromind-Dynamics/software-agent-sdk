"""Portal media URL signing for Label Studio task data.

The portal owns the browser-facing media route: it validates the caller's
session cookie, binds the returned URL to that user's bucket, and redirects to a
freshly presigned object URL on every render. Tasks therefore store stable
portal URLs instead of short-lived storage signatures.
"""

from __future__ import annotations

from typing import Any

import httpx

from openhands.tools.label_studio.converter import ConversionError


MEDIA_URLS_ROUTE = "/label_studio/media-urls"
# The portal caps one request at 1000 paths.
MEDIA_URL_BATCH_LIMIT = 1000


class PortalMediaSigner:
    """Signs many Storage paths into portal media URLs in one request."""

    def __init__(
        self,
        *,
        portal_base_url: str,
        headers: dict[str, str],
        cluster: str = "",
        batch_limit: int = MEDIA_URL_BATCH_LIMIT,
        timeout: float = 30.0,
    ) -> None:
        self._portal_base_url = portal_base_url.rstrip("/")
        self._headers = dict(headers)
        self._cluster = cluster
        self._batch_limit = batch_limit
        self._timeout = timeout
        self._expires_in: int | None = None

    @property
    def expires_in(self) -> int | None:
        """Lifetime the portal gave the URLs it last signed, in seconds.

        Task data stores these URLs permanently, so the caller has to know when
        they stop working.
        """
        return self._expires_in

    def sign_many(self, paths: list[str]) -> dict[str, str]:
        urls: dict[str, str] = {}
        for start in range(0, len(paths), self._batch_limit):
            chunk = paths[start : start + self._batch_limit]
            urls.update(self._sign_chunk(chunk))
        return urls

    def _sign_chunk(self, paths: list[str]) -> dict[str, str]:
        if not paths:
            return {}
        body: dict[str, Any] = {"paths": paths}
        if self._cluster:
            body["cluster"] = self._cluster
        # The URL is part of every failure: pointing portal_base_url at the wrong
        # host answers with a plain 404 that is otherwise indistinguishable from a
        # missing route on the right host.
        endpoint = f"{self._portal_base_url}{MEDIA_URLS_ROUTE}"
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
                f"Portal media-urls API {endpoint} unreachable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise ConversionError(
                f"Portal media-urls API {endpoint} returned HTTP "
                f"{response.status_code}: {response.text[:300]}"
            )
        try:
            payload: Any = response.json()
        except ValueError as exc:
            raise ConversionError(
                f"Portal media-urls API returned invalid JSON: {exc}"
            ) from exc
        urls = payload.get("urls") if isinstance(payload, dict) else None
        if not isinstance(urls, dict):
            raise ConversionError(
                f"Portal media-urls API response is missing the urls mapping: {payload}"
            )
        missing = [path for path in paths if not urls.get(path)]
        if missing:
            raise ConversionError(
                f"Portal media-urls API returned no url for {len(missing)} of "
                f"{len(paths)} paths, first: {missing[0]}"
            )
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, int) and expires_in > 0:
            self._expires_in = expires_in
        return {path: str(urls[path]) for path in paths}
