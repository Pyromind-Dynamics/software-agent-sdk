"""Reject request credentials at durable execution boundaries."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit


_CREDENTIAL_KEYS = frozenset(
    {
        "cookie",
        "setcookie",
        "authorization",
        "proxyauthorization",
        "accesstoken",
        "refreshtoken",
        "authtoken",
        "apikey",
        "password",
        "xamzsignature",
        "xamzcredential",
        "xgoogsignature",
        "signature",
    }
)


def validate_persisted_payload(value: object, *, path: str | None = None) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = None if path is None else f"{path}.{key}" if path else str(key)
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in _CREDENTIAL_KEYS:
                location = f" at {child}" if child else ""
                raise ValueError(f"request credentials cannot be persisted{location}")
            validate_persisted_payload(item, path=child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_persisted_payload(
                item, path=None if path is None else f"{path}[{index}]"
            )
    elif isinstance(value, str) and "://" in value:
        url = urlsplit(value)
        location = f" at {path}" if path else ""
        if url.username or url.password:
            raise ValueError(f"credential-bearing URLs cannot be persisted{location}")
        for key, _ in parse_qsl(url.query):
            if re.sub(r"[^a-z0-9]", "", key.lower()) in _CREDENTIAL_KEYS:
                raise ValueError(f"signed URLs cannot be persisted{location}")
