"""Bounded browser-equivalent image checks before project mutations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import httpx

from openhands.tools.label_studio.converter import ConversionError


def validate_task_media(
    tasks: Iterable[dict[str, Any]],
    timeout: float,
    path_diagnostics: Mapping[str, str] | None = None,
) -> None:
    """Check the first actual URL for each slot, including late optional slots.

    Signed media must work without the caller's session credentials, just as in
    an annotator's browser. Never attach portal credentials to media requests.
    """
    checked: set[str] = set()
    errors: list[str] = []
    for task in tasks:
        data = task.get("data", {})
        if not isinstance(data, dict):
            continue
        for key, path in data.items():
            if not key.endswith("_path") or not isinstance(path, str):
                continue
            field = key[:-5]
            if field in checked or field not in data:
                continue
            checked.add(field)
            error = _image_error(data[field], timeout)
            if error:
                diagnostic = (path_diagnostics or {}).get(
                    path,
                    f"original={path!r} resolved={path!r} path_type=storage_object",
                )
                errors.append(f"field={field!r} {diagnostic}: {error}")
    if errors:
        raise ConversionError("Media preflight failed: " + "; ".join(errors))


def _image_error(url: Any, timeout: float) -> str | None:
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        return "invalid_media_url (expected absolute HTTP URL)"
    try:
        with httpx.stream(
            "GET",
            url,
            headers={"Range": "bytes=0-4095"},
            timeout=timeout,
            follow_redirects=True,
        ) as response:
            status = response.status_code
            if status in {401, 403}:
                return f"authentication_failed HTTP {status}"
            if status == 404:
                return "path_not_found HTTP 404"
            if status not in {200, 206}:
                return f"unexpected_response HTTP {status}"
            if not response.headers.get("content-type", "").lower().startswith("image/"):
                return "non_image_response"
            if not next(response.iter_bytes(chunk_size=64), b""):
                return "empty_image_response"
    except (httpx.RequestError, httpx.InvalidURL):
        # Exception strings may include signed query tokens.
        return "media_unreachable"
    return None
