from __future__ import annotations

import hashlib
import os
from pathlib import PurePosixPath

import httpx
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.preparation import InspectResource

from openhands.tools.pyromind_dataset.definition import (
    _default_storage_base_url,
    request_storage_url,
)
from pyromind_agent_server.evaluation_storage import build_evaluation_storage


class StorageResourceInspector:
    def inspect(self, request: InspectResource, context: RequestContext) -> JsonObject:
        path = PurePosixPath(request.path)
        if (
            ".." in path.parts
            or "\\" in request.path
            or ":" in request.path
            or "?" in request.path
            or "#" in request.path
        ):
            raise ValueError("use a Storage path, not a URL or a parent traversal")
        validate_persisted_payload({"path": request.path})
        if not context.cookie and not context.authorization:
            raise ValueError("Storage inspection requires current request credentials")
        if request.kind == "directory":
            names = sorted(build_evaluation_storage(context).list(request.path))
            directory: JsonObject = {
                "path": request.path,
                "kind": "directory",
                "entries": [name for name in names[:200]],
                "truncated": len(names) > 200,
                "scope": "immediate children",
            }
            return directory
        headers = {
            name: value
            for name, value in (
                ("cookie", context.cookie),
                ("authorization", context.authorization),
                ("x-cluster", context.x_cluster),
            )
            if value
        }
        try:
            url = request_storage_url(
                storage_path=request.path,
                storage_base_url=os.getenv("PYROMIND_STORAGE_BASE_URL")
                or _default_storage_base_url(),
                headers=headers,
                timeout=30,
            )
            data = bytearray()
            with httpx.stream(
                "GET",
                url,
                headers={"Range": f"bytes=0-{request.max_bytes}"},
                timeout=30,
                follow_redirects=True,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=8192):
                    data.extend(chunk[: request.max_bytes + 1 - len(data)])
                    if len(data) > request.max_bytes:
                        break
            truncated = len(data) > request.max_bytes
            body = bytes(data[: request.max_bytes])
            result = {
                "path": request.path,
                "kind": "file",
                "content": body.decode("utf-8", errors="replace"),
                "bytes_read": len(body),
                "truncated": truncated,
                "scope": "file prefix",
                "preview_sha256": hashlib.sha256(body).hexdigest(),
            }
            validate_persisted_payload(result)
            return result
        except (httpx.HTTPError, ValueError):
            raise ValueError(
                "Storage preview unavailable; "
                "request the missing information from the user"
            ) from None
