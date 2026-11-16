from __future__ import annotations

import os

import httpx
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.storage_paths import to_storage_api_path

from openhands.tools.pyromind_dataset.definition import (
    _default_storage_base_url,
    request_storage_url,
)
from openhands.tools.utils.platform_env import platform_cluster_route


class StorageResourceInspector:
    def read_media(self, path: str, context: RequestContext) -> tuple[bytes, str]:
        storage_path = to_storage_api_path(path, field="path")
        if not context.cookie and not context.authorization:
            raise ValueError("Storage preview requires current request credentials")
        headers = {
            name: value
            for name, value in (
                ("cookie", context.cookie),
                ("authorization", context.authorization),
                ("x-cluster", platform_cluster_route(context.x_cluster)),
            )
            if value
        }
        try:
            url = request_storage_url(
                storage_path=storage_path,
                storage_base_url=os.getenv("PYROMIND_STORAGE_BASE_URL")
                or _default_storage_base_url(),
                headers=headers,
                timeout=30,
            )
            body = bytearray()
            with httpx.stream(
                "GET", url, timeout=30, follow_redirects=True
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=8192):
                    body.extend(chunk)
                    if len(body) > 8 * 1024 * 1024:
                        raise ValueError("sample image exceeds the 8 MB preview limit")
        except Exception as exc:
            raise ValueError("sample image preview is unavailable") from exc
        value = bytes(body)
        if value.startswith(b"\xff\xd8\xff"):
            mime_type = "image/jpeg"
        elif value.startswith(b"\x89PNG\r\n\x1a\n"):
            mime_type = "image/png"
        elif value.startswith(b"RIFF") and value[8:12] == b"WEBP":
            mime_type = "image/webp"
        else:
            raise ValueError("sample preview supports JPEG, PNG and WebP images")
        return value, mime_type
