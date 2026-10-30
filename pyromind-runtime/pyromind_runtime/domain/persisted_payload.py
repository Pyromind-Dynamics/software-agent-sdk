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


def validate_persisted_payload(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in _CREDENTIAL_KEYS:
                raise ValueError("request credentials cannot be persisted")
            validate_persisted_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_persisted_payload(item)
    elif isinstance(value, str) and "://" in value:
        url = urlsplit(value)
        if url.username or url.password:
            raise ValueError("credential-bearing URLs cannot be persisted")
        for key, _ in parse_qsl(url.query):
            if re.sub(r"[^a-z0-9]", "", key.lower()) in _CREDENTIAL_KEYS:
                raise ValueError("signed URLs cannot be persisted")
