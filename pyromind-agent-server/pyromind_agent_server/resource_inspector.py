from __future__ import annotations

import hashlib
import os
from pathlib import PurePosixPath

import httpx
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.persisted_payload import validate_persisted_payload
from pyromind_runtime.domain.preparation import InspectResource
from pyromind_runtime.ports.storage import EvaluationStorageUnavailableError

from openhands.tools.pyromind_dataset.definition import (
    _default_storage_base_url,
    request_storage_url,
)
from pyromind_agent_server.evaluation_storage import build_evaluation_storage


class StorageResourceInspector:
    def read_media(self, path: str, context: RequestContext) -> tuple[bytes, str]:
        if ".." in PurePosixPath(path).parts or any(
            c in path for c in ("\\", ":", "?", "#")
        ):
            raise ValueError("use a Storage path")
        if not context.cookie and not context.authorization:
            raise ValueError("Storage preview requires current request credentials")
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
                storage_path=path,
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
            digest = hashlib.sha256()
            total = 0
            lines = 0
            nonempty = False
            with httpx.stream(
                "GET",
                url,
                headers={}
                if request.compute_digest
                else {"Range": f"bytes=0-{request.max_bytes}"},
                timeout=30,
                follow_redirects=True,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=8192):
                    total += len(chunk)
                    digest.update(chunk)
                    if request.compute_digest:
                        for index, part in enumerate(chunk.split(b"\n")):
                            if index:
                                lines += int(nonempty)
                                nonempty = False
                            nonempty = nonempty or bool(part.strip())
                    data.extend(chunk[: max(0, request.max_bytes + 1 - len(data))])
                    if len(data) > request.max_bytes and not request.compute_digest:
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
            if request.compute_digest:
                result.update(
                    sha256=digest.hexdigest(),
                    total_bytes=total,
                    digest_scope="complete file",
                    nonempty_lines=lines + int(nonempty),
                )
            validate_persisted_payload(result)
            return result
        except (httpx.HTTPError, ValueError) as exc:
            raise EvaluationStorageUnavailableError(
                f"Storage preview unavailable for {request.path}"
            ) from exc
