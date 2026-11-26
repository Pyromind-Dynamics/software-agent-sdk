"""Pyromind server composition package."""

import os
import ssl
from pathlib import Path

import certifi


def _configure_platform_ca_bundle() -> None:
    if "SSL_CERT_FILE" in os.environ or "SSL_CERT_DIR" in os.environ:
        return
    defaults = ssl.get_default_verify_paths()
    if defaults.cafile or (
        defaults.capath and any(Path(defaults.capath).glob("????????.[0-9]*"))
    ):
        return
    if ssl.create_default_context().cert_store_stats()["x509_ca"]:
        return
    os.environ["SSL_CERT_FILE"] = (
        os.getenv("REQUESTS_CA_BUNDLE")
        or os.getenv("CURL_CA_BUNDLE")
        or certifi.where()
    )


# aiohttp builds the SDK's WebSocket TLS context at import time.
_configure_platform_ca_bundle()
