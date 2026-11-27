"""Request-scoped Pyromind Storage adapter for evaluation artifacts."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path, PurePosixPath

import httpx

from pyromind_common.tools.pyromind_dataset.definition import (
    StorageFileNotFoundError,
    StorageFileTooLargeError,
    _default_storage_base_url,
    _extract_api_data,
    download_file_from_pyromind,
    request_storage_url,
    upload_local_file_to_pyromind,
)
from pyromind_common.tools.utils.platform_env import (
    platform_headers,
    platform_portal_url,
)
from pyromind_runtime.application.evaluation_annotations import RUN_MANIFEST_NAME
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.storage import (
    WORKFLOW_ARTIFACT_MAX_BYTES,
    EvaluationStorage,
    EvaluationStorageMissingError,
    EvaluationStorageTooLargeError,
    EvaluationStorageUnavailableError,
)


logger = logging.getLogger(__name__)

REPORT_MAX_BYTES = WORKFLOW_ARTIFACT_MAX_BYTES
STORAGE_TIMEOUT_SECONDS = 60.0


class PyromindEvaluationStorage(EvaluationStorage):
    """EvaluationStorage over the Pyromind Storage API, caller's identity.

    Objects are read through pre-signed GETs and written through the
    presigned-upload flow under their final basename. Deletion is not
    implemented (the Storage API has no delete contract here), so protocol
    GC quietly skips — objects accumulate.

    Failure classification follows the port contract: only an explicit
    "not found" from the backend becomes
    :class:`EvaluationStorageMissingError`; every transport, permission, or
    payload failure becomes :class:`EvaluationStorageUnavailableError` with
    the cause logged and kept out of the exception message.
    """

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str],
        timeout: float = STORAGE_TIMEOUT_SECONDS,
        max_read_bytes: int = REPORT_MAX_BYTES,
        read_attempts: int = 3,
        max_manifest_bytes: int | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = headers
        self._timeout = timeout
        self._max_read_bytes = max_read_bytes
        self._read_attempts = read_attempts
        self._max_manifest_bytes = max_manifest_bytes

    def preview_url(self, path: str) -> str:
        return request_storage_url(
            storage_path=path,
            storage_base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
            force_download=False,
        )

    def read(self, path: str) -> bytes:
        max_bytes = self._max_read_bytes
        if self._max_manifest_bytes is not None and path.endswith(
            f"/{RUN_MANIFEST_NAME}"
        ):
            max_bytes = self._max_manifest_bytes
        try:
            return download_file_from_pyromind(
                storage_path=path,
                storage_base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout,
                max_bytes=max_bytes,
                attempts=self._read_attempts,
            )
        except StorageFileNotFoundError as exc:
            raise EvaluationStorageMissingError(
                f"storage object {path} does not exist"
            ) from exc
        except StorageFileTooLargeError as exc:
            raise EvaluationStorageTooLargeError(
                "Storage object exceeds the read limit", retryable=False
            ) from exc
        except Exception as exc:
            logger.error("evaluation storage read failed for %s: %s", path, exc)
            raise EvaluationStorageUnavailableError(
                f"storage read failed for {path}"
            ) from exc

    def read_evidence_range(self, path: str, offset: int, size: int) -> bytes:
        if offset < 0 or not 0 < size <= 64 * 1024:
            raise ValueError("Invalid evidence range")
        url = request_storage_url(
            storage_path=path,
            storage_base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
        )
        with httpx.stream(
            "GET",
            url,
            headers={
                "range": f"bytes={offset}-{offset + size - 1}",
                "accept-encoding": "identity",
            },
            timeout=self._timeout,
            follow_redirects=True,
        ) as response:
            if response.status_code == 416:
                return b""
            if response.status_code == 404:
                raise EvaluationStorageMissingError(path)
            if response.status_code == 206:
                if not response.headers.get("content-range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise EvaluationStorageUnavailableError(
                        "Invalid evidence range response"
                    )
            elif response.status_code != 200 or offset:
                raise EvaluationStorageUnavailableError(
                    "Storage did not serve the requested evidence range"
                )
            data = bytearray()
            for chunk in response.iter_bytes(chunk_size=8192):
                data.extend(chunk[: size - len(data)])
                if len(data) == size:
                    break
            return bytes(data)

    def write(self, path: str, payload: bytes) -> None:
        target_dir = str(PurePosixPath(path).parent)
        try:
            with tempfile.TemporaryDirectory(prefix="eval-annotation-") as staged_dir:
                staged = Path(staged_dir) / PurePosixPath(path).name
                # The uploader stores files under their local basename, so the
                # staged name must equal the canonical artifact name.
                staged.write_bytes(payload)
                upload_local_file_to_pyromind(
                    local_path=staged,
                    target_dir=target_dir,
                    storage_base_url=self._base_url,
                    headers=self._headers,
                    timeout=self._timeout,
                )
        except Exception as exc:
            logger.error("evaluation storage write failed for %s: %s", path, exc)
            raise EvaluationStorageUnavailableError(
                f"storage write failed for {path}"
            ) from exc

    def list(self, prefix: str) -> set[str]:
        # Unlike the advisory storage helper, failures raise:
        # an unreachable listing must never be mistaken for an empty one.
        try:
            response = httpx.post(
                f"{self._base_url}/file_list",
                headers=self._headers,
                json={"path": prefix, "search": ""},
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            logger.error("storage file_list failed for %s: %s", prefix, exc)
            raise EvaluationStorageUnavailableError(
                f"storage listing failed for {prefix}",
                retryable=isinstance(exc, httpx.RequestError)
                or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code in {408, 429, 502, 503, 504}
                ),
            ) from exc
        if not isinstance(payload, dict):
            raise EvaluationStorageUnavailableError(
                f"storage file_list returned a non-object payload for {prefix}"
            )
        # Business failures ride HTTP 200 with success=false / isLoggedIn
        # flags; the repo-wide checks make those raise like transport errors.
        data = _extract_api_data("storage file_list", payload)
        if isinstance(data, str):
            raise EvaluationStorageUnavailableError(
                f"storage file_list failed for {prefix}: {data}"
            )
        entries = data.get("list")
        if not isinstance(entries, list):
            raise EvaluationStorageUnavailableError(
                f"storage file_list returned an unexpected payload for {prefix}"
            )
        names: set[str] = set()
        for item in entries:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("path")
                if isinstance(name, str):
                    names.add(name.rsplit("/", 1)[-1])
        return names


def build_evaluation_storage(
    context: RequestContext,
    *,
    timeout: float = STORAGE_TIMEOUT_SECONDS,
    max_read_bytes: int = REPORT_MAX_BYTES,
    read_attempts: int = 3,
    max_manifest_bytes: int | None = None,
) -> PyromindEvaluationStorage:
    headers = {"accept": "*/*"}
    if context.cookie:
        headers["cookie"] = context.cookie
    if context.authorization:
        headers["authorization"] = context.authorization
    headers = platform_headers(headers, route=context.x_cluster)
    base_url = os.getenv("PYROMIND_STORAGE_BASE_URL") or _default_storage_base_url()
    return PyromindEvaluationStorage(
        base_url=platform_portal_url(base_url, headers),
        headers=headers,
        timeout=timeout,
        max_read_bytes=max_read_bytes,
        read_attempts=read_attempts,
        max_manifest_bytes=max_manifest_bytes,
    )


def build_observation_evaluation_storage(context: RequestContext) -> EvaluationStorage:
    return build_evaluation_storage(
        context,
        timeout=5,
        max_read_bytes=48 * 1024,
        max_manifest_bytes=4 * 1024 * 1024,
        read_attempts=2,
    )
