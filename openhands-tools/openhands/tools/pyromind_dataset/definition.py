"""Definitions of the upload_file_to_pyromind and get_storage_url tools."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

import httpx
from pydantic import BaseModel, Field
from rich.text import Text

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.utils import default_path_access_policy
from openhands.tools.utils.workspace_staging import (
    WorkspaceStagingError,
    is_remote_workspace,
    resolve_workspace_path,
    staged_remote_path,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


PRE_STORAGE_API_BASE_URL = "https://pre-api-portal.pyromind.ai/storage_api"
PROD_STORAGE_API_BASE_URL = "https://api-portal.pyromind.ai/storage_api"
PYROMIND_STORAGE_AUTH_COOKIE_SECRET = "PYROMIND_STORAGE_AUTH_COOKIE"
PYROMIND_STORAGE_HEADERS_STATE_KEY = "pyromind_storage_headers"
PYROMIND_AGENT_STORAGE_ROOT = "/.pyromind-agent"

_UPLOAD_ATTEMPTS = 3
_UPLOAD_BACKOFF_SECONDS = 1.0
# Presigned upload URLs are short-lived, so a 403 is worth retrying: the retry
# mints a fresh URL. 404 and other 4xx mean the request itself is wrong.
_TRANSIENT_STORAGE_STATUSES = frozenset({403, 408, 425, 429})


class StorageFileNotFoundError(ValueError):
    """Raised when a Storage object genuinely does not exist.

    Callers use this to tell "absent" apart from a transient or server-side
    failure; those must never be mistaken for a missing file.
    """


class _TransientStorageError(ValueError):
    """Internal marker for a Storage failure that is worth retrying."""


def _is_transient_storage_status(status_code: int) -> bool:
    return status_code >= 500 or status_code in _TRANSIENT_STORAGE_STATUSES


_WORKSPACE_PATH_PREFIXES = ("/workspace/", "workspace/")

_PROD_APP_ENVS = {"prod", "production", "online"}


def _default_storage_base_url() -> str:
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    if app_env in _PROD_APP_ENVS:
        return PROD_STORAGE_API_BASE_URL
    return PRE_STORAGE_API_BASE_URL


# ---------------------------------------------------------------------------
# upload_file_to_pyromind
# ---------------------------------------------------------------------------


class UploadFileToPyromindAction(Action):
    """Upload a local workspace file to Pyromind storage."""

    file_path: str = Field(
        description=(
            "Path of the file in the current conversation workspace to "
            "upload (e.g. 'acc.py')."
        ),
    )
    target_dir: str | None = Field(
        default=None,
        description=(
            "Storage directory to upload into. Defaults to the "
            "conversation-scoped /.pyromind-agent/<conversation_id>/ directory."
        ),
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Upload file to Pyromind: ", style="bold blue")
        content.append(self.file_path)
        return content


class UploadFileToPyromindObservation(Observation):
    """Result of a storage upload."""

    storage_path: str | None = Field(
        default=None,
        description=(
            "Absolute storage path of the uploaded file, usable in node "
            "parameters (e.g. /.pyromind-agent/<conversation_id>/acc.py)."
        ),
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Upload failed", style="bold red")
        else:
            content.append("Uploaded: ", style="bold green")
            content.append(self.storage_path or "")
        return content


_UPLOAD_FILE_DESCRIPTION = """Upload a workspace file to Pyromind storage.

Use this when a server-side process needs a file that only exists in the
conversation workspace. Two common cases:

- A workflow node parameter that takes a server-side path, most commonly a
  custom evaluation metric or reward script for MetricsConfigBuilderCustomNode:
  write the Python file locally first, upload it with this tool, then use the
  returned storage path in the node's `entry` parameter as
  `<storage_path>:<function_name>`
  (e.g. /.pyromind-agent/<conversation_id>/acc.py:acc_func).
- A pipeline input sidecar for `df_submit_pipeline`, for example an image
  manifest. Pass target_dir=<the storage directory that holds the data> so the
  manifest lands beside the files it references, then pass the returned storage
  path as `input_path`. Without target_dir the file goes to the conversation
  directory.

Returns the absolute storage path of the uploaded file.
"""


class UploadFileToPyromindExecutor(
    ToolExecutor[UploadFileToPyromindAction, UploadFileToPyromindObservation]
):
    """Upload workspace files through the Pyromind storage API."""

    def __init__(
        self,
        storage_base_url: str | None = None,
        headers: dict[str, str] | None = None,
        secret_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        base_url = storage_base_url or _default_storage_base_url()
        self._storage_base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})
        self._secret_headers = dict(secret_headers or {})
        self._timeout = timeout

    def __call__(
        self,
        action: UploadFileToPyromindAction,
        conversation: BaseConversation | None = None,
    ) -> UploadFileToPyromindObservation:
        try:
            if conversation is None:
                raise ValueError(
                    "upload_file_to_pyromind requires an active conversation."
                )
            headers = self._resolve_headers(conversation)
        except ValueError as exc:
            return UploadFileToPyromindObservation.from_text(
                text=str(exc),
                is_error=True,
            )

        target_dir = (
            action.target_dir or f"{PYROMIND_AGENT_STORAGE_ROOT}/{conversation.id}"
        )
        overwritten = False
        try:
            with _workspace_upload_path(action.file_path, conversation) as local_path:
                overwritten = local_path.name in storage_file_names(
                    directory=target_dir,
                    storage_base_url=self._storage_base_url,
                    headers=headers,
                    timeout=self._timeout,
                    search=local_path.name,
                )
                storage_path = upload_local_file_to_pyromind(
                    local_path=local_path,
                    target_dir=target_dir,
                    storage_base_url=self._storage_base_url,
                    headers=headers,
                    timeout=self._timeout,
                )
        except ValueError as exc:
            return UploadFileToPyromindObservation.from_text(
                text=str(exc),
                is_error=True,
            )
        except OSError as exc:
            return UploadFileToPyromindObservation.from_text(
                text=f"Failed to read file for upload: {exc}",
                is_error=True,
            )

        text = f"File uploaded to Pyromind storage: {storage_path}"
        if overwritten:
            text += "\nWarning: an existing file at that Storage path was overwritten."
        return UploadFileToPyromindObservation.from_text(
            text=text,
            storage_path=storage_path,
        )

    def _resolve_headers(
        self,
        conversation: BaseConversation | None,
    ) -> dict[str, str]:
        headers = {"accept": "*/*", **self._headers}
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(_resolve_secret_headers(conversation, self._secret_headers))
        return headers


def upload_local_file_to_pyromind(
    *,
    local_path: Path,
    target_dir: str,
    storage_base_url: str,
    headers: dict[str, str],
    timeout: float,
) -> str:
    """Upload a trusted local runtime file and return its Storage path."""
    last_error: _TransientStorageError | None = None
    for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
        try:
            return _upload_local_file_once(
                local_path=local_path,
                target_dir=target_dir,
                storage_base_url=storage_base_url,
                headers=headers,
                timeout=timeout,
            )
        except _TransientStorageError as exc:
            last_error = exc
            if attempt < _UPLOAD_ATTEMPTS:
                time.sleep(_UPLOAD_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise ValueError(f"{last_error} (after {_UPLOAD_ATTEMPTS} attempts)") from None


def _upload_local_file_once(
    *,
    local_path: Path,
    target_dir: str,
    storage_base_url: str,
    headers: dict[str, str],
    timeout: float,
) -> str:
    filename = local_path.name
    storage_path = str(PurePosixPath(target_dir) / filename)
    file_size = local_path.stat().st_size
    try:
        response = httpx.post(
            f"{storage_base_url.rstrip('/')}/presigned_upload_url",
            headers=headers,
            json={
                "filename": filename,
                "path": target_dir,
                "content_type": "application/octet-stream",
                "size": file_size,
            },
            timeout=timeout,
        )
    except httpx.RequestError as exc:
        raise _TransientStorageError(
            "Failed to call Pyromind storage presigned_upload_url API: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if _is_transient_storage_status(response.status_code):
        raise _TransientStorageError(
            "Pyromind storage presigned_upload_url API returned HTTP "
            f"{response.status_code}: {_truncate_text(response.text)}"
        )

    payload_result = _decode_json_response(
        response, "Pyromind storage presigned_upload_url API"
    )
    if isinstance(payload_result, str):
        raise ValueError(payload_result)
    data_result = _extract_api_data("presigned_upload_url", payload_result)
    if isinstance(data_result, str):
        raise ValueError(data_result)

    if data_result.get("multipart") is True:
        _upload_local_file_multipart(
            local_path=local_path,
            data_result=data_result,
            target_dir=target_dir,
            filename=filename,
            file_size=file_size,
            storage_base_url=storage_base_url,
            headers=headers,
            timeout=timeout,
        )
        return storage_path

    upload_url = data_result.get("upload_url")
    if not isinstance(upload_url, str) or not upload_url.strip():
        raise ValueError(
            "Pyromind storage presigned_upload_url API response is missing "
            "upload_url data."
        )
    _presigned_put(
        upload_url,
        content=local_path.open("rb"),
        method=str(data_result.get("method") or "PUT").upper(),
        extra_headers=data_result.get("headers"),
        timeout=timeout,
        error_label="Pyromind storage presigned upload",
    )
    return storage_path


def _presigned_put(
    url: str,
    *,
    content: Any,
    method: str,
    extra_headers: Any,
    timeout: float,
    error_label: str,
) -> None:
    """PUT bytes/file-object to one presigned upload URL, raising ValueError."""
    upload_headers = extra_headers if isinstance(extra_headers, dict) else {}
    try:
        upload_response = httpx.request(
            method, url, content=content, headers=upload_headers, timeout=timeout
        )
    except httpx.RequestError as exc:
        raise _TransientStorageError(
            f"Failed to upload file via {error_label}: {type(exc).__name__}: {exc}"
        ) from exc
    if upload_response.status_code >= 400:
        message = (
            f"{error_label} returned HTTP {upload_response.status_code}: "
            f"{_truncate_text(upload_response.text)}"
        )
        if _is_transient_storage_status(upload_response.status_code):
            raise _TransientStorageError(message)
        raise ValueError(message)


def _upload_local_file_multipart(
    *,
    local_path: Path,
    data_result: dict[str, Any],
    target_dir: str,
    filename: str,
    file_size: int,
    storage_base_url: str,
    headers: dict[str, str],
    timeout: float,
) -> None:
    """Upload a presigned multipart payload part-by-part, then complete it."""
    upload_id = data_result.get("upload_id")
    part_size = data_result.get("part_size")
    part_count = data_result.get("part_count")
    part_urls = data_result.get("part_urls")
    if not isinstance(upload_id, str) or not upload_id:
        raise ValueError(
            "Pyromind storage presigned_upload_url API multipart response is "
            "missing upload_id data."
        )
    if not isinstance(part_size, int) or part_size < 1:
        raise ValueError(
            "Pyromind storage presigned_upload_url API multipart response is "
            "missing part_size data."
        )
    if not isinstance(part_count, int) or part_count < 1:
        raise ValueError(
            "Pyromind storage presigned_upload_url API multipart response is "
            "missing part_count data."
        )
    if not isinstance(part_urls, list) or len(part_urls) != part_count:
        raise ValueError(
            "Pyromind storage presigned_upload_url API multipart response "
            f"has {len(part_urls) if isinstance(part_urls, list) else 0} "
            f"part_urls, expected {part_count}."
        )

    part_label = "Pyromind storage multipart part upload"
    with local_path.open("rb") as file_obj:
        for part_url in part_urls:
            chunk = file_obj.read(part_size)
            if not chunk:
                break
            _presigned_put(
                str(part_url),
                content=chunk,
                method="PUT",
                extra_headers=None,
                timeout=timeout,
                error_label=part_label,
            )

    try:
        complete_response = httpx.post(
            f"{storage_base_url.rstrip('/')}/multipart_complete",
            headers=headers,
            json={
                "path": target_dir,
                "filename": filename,
                "size": file_size,
                "upload_id": upload_id,
                "part_count": part_count,
            },
            timeout=timeout,
        )
    except httpx.RequestError as exc:
        raise ValueError(
            "Failed to call Pyromind storage multipart_complete API: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    payload_result = _decode_json_response(
        complete_response, "Pyromind storage multipart_complete API"
    )
    if isinstance(payload_result, str):
        raise ValueError(payload_result)
    complete_result = _extract_api_data("multipart_complete", payload_result)
    if isinstance(complete_result, str):
        raise ValueError(complete_result)


def request_storage_url(
    *,
    storage_path: str,
    storage_base_url: str,
    headers: dict[str, str],
    timeout: float,
    force_download: bool | None = None,
) -> str:
    """Return the pre-signed Storage URL for one object.

    ``force_download`` is omitted unless the caller sets it, so a request that
    only needs the bytes keeps the Storage API default.
    """
    body: dict[str, Any] = {"path": storage_path}
    if force_download is not None:
        body["force_download"] = force_download
    try:
        response = httpx.post(
            f"{storage_base_url.rstrip('/')}/get_url",
            headers=headers,
            json=body,
            timeout=timeout,
        )
    except httpx.RequestError as exc:
        raise ValueError(
            f"Failed to request Pyromind storage download URL: {exc}"
        ) from exc
    payload = _decode_json_response(response, "Pyromind storage get_url API")
    if isinstance(payload, str):
        raise ValueError(payload)
    data = _extract_api_data("get_url", payload)
    if isinstance(data, str):
        raise ValueError(data)
    url = data.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Pyromind storage get_url API response is missing url data.")
    return url


def download_file_from_pyromind(
    *,
    storage_path: str,
    storage_base_url: str,
    headers: dict[str, str],
    timeout: float,
    max_bytes: int,
) -> bytes:
    """Download one bounded storage file for non-data control-plane checks."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be greater than 0")
    url = request_storage_url(
        storage_path=storage_path,
        storage_base_url=storage_base_url,
        headers=headers,
        timeout=timeout,
    )

    content = bytearray()
    try:
        with httpx.stream(
            "GET",
            url,
            # Signed-URL GETs are CDN-cached; bypass so freshly written
            # control-plane files are visible immediately.
            headers={"cache-control": "no-cache", "pragma": "no-cache"},
            timeout=timeout,
            follow_redirects=True,
        ) as download:
            if download.status_code == 404:
                raise StorageFileNotFoundError(
                    f"Pyromind storage object not found: {storage_path}"
                )
            if download.status_code >= 400:
                body = download.read().decode("utf-8", errors="replace")
                raise ValueError(
                    "Pyromind storage download URL returned HTTP "
                    f"{download.status_code}: {_truncate_text(body)}"
                )
            for chunk in download.iter_bytes():
                if len(content) + len(chunk) > max_bytes:
                    raise ValueError(
                        f"Storage file exceeds the {max_bytes}-byte preflight limit."
                    )
                content.extend(chunk)
    except httpx.RequestError as exc:
        raise ValueError(f"Failed to download Pyromind storage file: {exc}") from exc
    return bytes(content)


def download_tail_from_pyromind(
    *,
    storage_path: str,
    storage_base_url: str,
    headers: dict[str, str],
    timeout: float,
    tail_bytes: int,
) -> tuple[bytes, int]:
    """Range-download the last ``tail_bytes`` of a storage file.

    Uses a suffix byte range (``bytes=-N``) so the caller never needs the
    file size up front. Returns ``(tail, total_size)``. Raises when storage
    ignores range requests for files larger than ``tail_bytes``.
    """
    if tail_bytes < 1:
        raise ValueError("tail_bytes must be greater than 0")
    url = request_storage_url(
        storage_path=storage_path,
        storage_base_url=storage_base_url,
        headers=headers,
        timeout=timeout,
    )

    try:
        with httpx.stream(
            "GET",
            url,
            headers={"range": f"bytes=-{tail_bytes}"},
            timeout=timeout,
            follow_redirects=True,
        ) as download:
            if download.status_code >= 400:
                body = download.read().decode("utf-8", errors="replace")
                raise ValueError(
                    "Pyromind storage download URL returned HTTP "
                    f"{download.status_code}: {_truncate_text(body)}"
                )
            if download.status_code == 206:
                total = _parse_content_range_total(
                    download.headers.get("content-range", "")
                )
                if total is None:
                    raise ValueError(
                        "Pyromind storage ranged download is missing "
                        "content-range in its response."
                    )
                return download.read(), total
            length_header = download.headers.get("content-length")
            if length_header is not None and length_header.isdigit():
                if int(length_header) > tail_bytes:
                    raise ValueError(
                        "Storage ignored the range request (HTTP 200 with "
                        f"content-length {length_header}); refusing unbounded "
                        "download."
                    )
                return download.read(), int(length_header)
            body = download.read()
            if len(body) > tail_bytes:
                raise ValueError(
                    "Storage ignored the range request and streamed "
                    f"{len(body)} bytes; refusing unbounded tail download."
                )
            return body, len(body)
    except httpx.RequestError as exc:
        raise ValueError(
            f"Failed to download Pyromind storage file tail: {exc}"
        ) from exc


def _parse_content_range_total(content_range: str) -> int | None:
    total = content_range.rsplit("/", 1)[-1].strip() if "/" in content_range else ""
    return int(total) if total.isdigit() else None


class UploadFileToPyromindTool(
    ToolDefinition[UploadFileToPyromindAction, UploadFileToPyromindObservation]
):
    """Tool for uploading workspace files to Pyromind storage."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[ToolDefinition]:
        storage_base_url = str(
            params.pop("storage_base_url", _default_storage_base_url())
        )
        headers = params.pop("headers", None)
        secret_headers = params.pop("secret_headers", None)
        timeout = float(params.pop("timeout", 30.0))
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"UploadFileToPyromindTool got unknown params: {names}")
        _validate_storage_tool_params(
            storage_base_url,
            headers,
            secret_headers,
            timeout,
        )
        return [
            cls(
                description=_UPLOAD_FILE_DESCRIPTION,
                action_type=UploadFileToPyromindAction,
                observation_type=UploadFileToPyromindObservation,
                executor=UploadFileToPyromindExecutor(
                    storage_base_url=storage_base_url,
                    headers=_normalize_headers(headers),
                    secret_headers=_normalize_headers(secret_headers),
                    timeout=timeout,
                ),
                annotations=ToolAnnotations(
                    title="upload_file_to_pyromind",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


_MAX_URL_PATHS = 20

_GET_STORAGE_URL_DESCRIPTION = (
    "Return pre-signed Storage URLs that open files from the conversation.\n\n"
    "Use this whenever the user asks to see an image, HTML page, PDF, or "
    "another artifact: call it with the file paths, then embed the returned "
    "URLs in your reply as Markdown (`![description](url)` for images, "
    "`[label](url)` for other files). Showing a file to the user never depends "
    "on image support in your own model.\n\n"
    "Paths may be workspace paths (`public_data/report.html`), the Storage "
    "alias (`storage/datasets/img.png`), absolute sandbox paths, or Storage "
    "paths (`/datasets/img.png`). The file must live under the mounted "
    "Storage; sandbox paths outside the mount, such as `/tmp`, have no URL, so "
    "copy those into the workspace first (for example under `public_data/`) "
    "and call this tool again. "
    "URLs are temporary pre-signed links, so embed them in the reply instead "
    "of keeping them for later.\n"
)


def _storage_path_for(
    workspace: Any,
    conversation_id: str,
    path: str,
) -> str:
    """Map a model-facing path to the path the Storage API expects.

    Workspace paths resolve under the conversation directory inside Storage,
    while the ``storage/`` alias, absolute mount paths and Storage paths
    resolve against the Storage root.
    """
    try:
        relative, from_storage = resolve_workspace_path(workspace, path)
    except WorkspaceStagingError:
        # Outside the execution workspace, so the path already names Storage.
        return _storage_path(PurePosixPath(path))
    if not relative.parts:
        raise ValueError(f"Path must name a file or directory: {path!r}")
    if from_storage:
        return _storage_path(relative)
    if getattr(workspace, "storage_path", None) is None:
        raise ValueError(
            f"{path!r} is a workspace path and this session has no Storage "
            "mount, so Storage cannot serve it."
        )
    return _storage_path(
        PurePosixPath(PYROMIND_AGENT_STORAGE_ROOT, conversation_id, *relative.parts)
    )


def _storage_path(path: PurePosixPath) -> str:
    parts = [part for part in path.parts if part not in {"", "/", "."}]
    if not parts or ".." in parts:
        raise ValueError(f"Storage path must not be the root or contain '..': {path}")
    return "/" + "/".join(parts)


class GetStorageUrlAction(Action):
    """Resolve file paths into pre-signed Storage preview URLs."""

    paths: list[str] = Field(
        description=(
            "Files to resolve, addressed as workspace paths "
            "('public_data/report.html'), the Storage alias "
            "('storage/datasets/img.png'), or Storage paths "
            "('/datasets/img.png')."
        ),
    )


class StoragePreviewUrl(BaseModel):
    """One resolved Storage preview URL."""

    path: str = Field(description="Storage path the URL addresses.")
    url: str = Field(description="Pre-signed URL that opens the file inline.")


class GetStorageUrlObservation(Observation):
    """Pre-signed Storage URLs for the requested files."""

    urls: list[StoragePreviewUrl] = Field(
        default_factory=list,
        description="Resolved preview URLs, in request order.",
    )
    failures: list[str] = Field(
        default_factory=list,
        description="Requested paths that produced no URL, with the reason.",
    )


class GetStorageUrlExecutor(
    ToolExecutor[GetStorageUrlAction, GetStorageUrlObservation]
):
    """Turn workspace and Storage paths into pre-signed preview URLs."""

    def __init__(
        self,
        storage_base_url: str | None = None,
        headers: dict[str, str] | None = None,
        secret_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        base_url = storage_base_url or _default_storage_base_url()
        self._storage_base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})
        self._secret_headers = dict(secret_headers or {})
        self._timeout = timeout

    def __call__(
        self,
        action: GetStorageUrlAction,
        conversation: BaseConversation | None = None,
    ) -> GetStorageUrlObservation:
        try:
            if conversation is None:
                raise ValueError("get_storage_url requires an active conversation.")
            if len(action.paths) > _MAX_URL_PATHS:
                raise ValueError(
                    f"get_storage_url accepts at most {_MAX_URL_PATHS} paths per call."
                )
            headers = self._resolve_headers(conversation)
        except ValueError as exc:
            return GetStorageUrlObservation.from_text(text=str(exc), is_error=True)

        workspace = getattr(conversation, "workspace", None)
        conversation_id = str(getattr(conversation, "id", "") or "")
        urls: list[StoragePreviewUrl] = []
        failures: list[str] = []
        for path in action.paths:
            try:
                storage_path = _storage_path_for(workspace, conversation_id, path)
                url = request_storage_url(
                    storage_path=storage_path,
                    storage_base_url=self._storage_base_url,
                    headers=headers,
                    timeout=self._timeout,
                    force_download=False,
                )
            except ValueError as exc:
                failures.append(f"{path}: {exc}")
                continue
            urls.append(StoragePreviewUrl(path=storage_path, url=url))

        lines = [f"{entry.path} -> {entry.url}" for entry in urls]
        if failures:
            lines.append("--- failed ---")
            lines.extend(failures)
        return GetStorageUrlObservation.from_text(
            text="\n".join(lines) or "No paths were resolved.",
            is_error=not urls,
            urls=urls,
            failures=failures,
        )

    def _resolve_headers(
        self,
        conversation: BaseConversation,
    ) -> dict[str, str]:
        headers = {"accept": "*/*", **self._headers}
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(_resolve_secret_headers(conversation, self._secret_headers))
        return headers


class GetStorageUrlTool(ToolDefinition[GetStorageUrlAction, GetStorageUrlObservation]):
    """Tool for resolving file paths into Storage preview URLs."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[ToolDefinition]:
        storage_base_url = str(
            params.pop("storage_base_url", _default_storage_base_url())
        )
        headers = params.pop("headers", None)
        secret_headers = params.pop("secret_headers", None)
        timeout = float(params.pop("timeout", 30.0))
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"GetStorageUrlTool got unknown params: {names}")
        _validate_storage_tool_params(
            storage_base_url,
            headers,
            secret_headers,
            timeout,
        )
        return [
            cls(
                description=_GET_STORAGE_URL_DESCRIPTION,
                action_type=GetStorageUrlAction,
                observation_type=GetStorageUrlObservation,
                executor=GetStorageUrlExecutor(
                    storage_base_url=storage_base_url,
                    headers=_normalize_headers(headers),
                    secret_headers=_normalize_headers(secret_headers),
                    timeout=timeout,
                ),
                annotations=ToolAnnotations(
                    title="get_storage_url",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


def _validate_storage_tool_params(
    storage_base_url: str,
    headers: Any,
    secret_headers: Any,
    timeout: float,
) -> None:
    if not storage_base_url.strip():
        raise ValueError("storage_base_url must be a non-empty string")
    if headers is not None and not isinstance(headers, dict):
        raise ValueError("headers must be a dictionary when provided")
    if secret_headers is not None and not isinstance(secret_headers, dict):
        raise ValueError("secret_headers must be a dictionary when provided")
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")


def _normalize_headers(value: Any) -> dict[str, str] | None:
    if not value:
        return None
    return {str(k): str(v) for k, v in value.items()}


def _resolve_secret_headers(
    conversation: BaseConversation | None,
    configured_secret_headers: dict[str, str],
) -> dict[str, str]:
    secret_headers = dict(configured_secret_headers)
    if conversation is not None:
        state = cast("ConversationState", conversation.state)
        secret_registry = state.secret_registry
        if secret_registry.get_secret_value(PYROMIND_STORAGE_AUTH_COOKIE_SECRET):
            secret_headers.setdefault("cookie", PYROMIND_STORAGE_AUTH_COOKIE_SECRET)
    if not secret_headers:
        return {}
    if conversation is None:
        raise ValueError(
            "Cannot resolve Pyromind storage API header secrets without an "
            "active conversation."
        )

    resolved: dict[str, str] = {}
    state = cast("ConversationState", conversation.state)
    secret_registry = state.secret_registry
    for header_name, secret_name in secret_headers.items():
        value = secret_registry.get_secret_value(secret_name)
        if not value:
            raise ValueError(
                f"Secret '{secret_name}' required for Pyromind storage API "
                f"header '{header_name}' was not found."
            )
        resolved[header_name] = value
    return resolved


def _resolve_conversation_headers(
    conversation: BaseConversation | None,
) -> dict[str, str]:
    if conversation is None:
        return {}

    state = cast("ConversationState", conversation.state)
    headers = state.agent_state.get(PYROMIND_STORAGE_HEADERS_STATE_KEY)
    if not isinstance(headers, dict):
        return {}
    return {
        str(name): str(value) for name, value in headers.items() if value is not None
    }


def _resolve_workspace_file(
    file_path: str,
    conversation: BaseConversation | None,
) -> Path:
    if conversation is None:
        raise ValueError(
            "Cannot upload a workspace file without an active conversation."
        )

    workspace = cast(Any, conversation).workspace
    workspace_dir = Path(workspace.working_dir).resolve()
    path_policy = default_path_access_policy(workspace_dir)
    candidate = Path(file_path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (workspace_dir / candidate).resolve()
    )
    try:
        resolved.relative_to(workspace_dir)
    except ValueError as exc:
        raise ValueError(
            f"Cannot upload file outside the conversation workspace: {file_path}"
        ) from exc
    if not path_policy.check(resolved, "read") or not resolved.is_file():
        raise ValueError(f"Cannot upload missing workspace file: {file_path}")
    return resolved


@contextmanager
def _workspace_upload_path(
    file_path: str,
    conversation: BaseConversation,
) -> Iterator[Path]:
    """Yield a host path holding ``file_path`` for the duration of the upload.

    Local workspaces resolve in place. A remote (sandbox) workspace has no host
    filesystem, so the file is downloaded to a temporary host copy first.
    """
    workspace = cast(Any, conversation).workspace
    if not is_remote_workspace(workspace):
        yield _resolve_workspace_file(file_path, conversation)
        return
    with staged_remote_path(workspace, file_path) as staged:
        if not staged.is_file():
            raise ValueError(f"Cannot upload missing workspace file: {file_path}")
        yield staged


def _decode_json_response(
    response: httpx.Response,
    api_name: str,
) -> dict[str, Any] | str:
    if response.status_code >= 400:
        return (
            f"{api_name} returned HTTP {response.status_code}: "
            f"{_truncate_text(response.text)}"
        )

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        return f"{api_name} returned invalid JSON: {exc.msg}"
    if not isinstance(payload, dict):
        return f"{api_name} returned a non-object JSON payload."
    return payload


def _extract_api_data(api_name: str, payload: dict[str, Any]) -> dict[str, Any] | str:
    success = payload.get("success")
    data = payload.get("data")
    if success is not True:
        return _format_api_failure(api_name, payload)
    if not isinstance(data, dict):
        return f"Pyromind storage {api_name} API response is missing object data."
    if data.get("isLoggedIn") is False:
        message = _optional_str(data.get("message")) or "login required"
        return f"Pyromind storage {api_name} API requires login: {message}"
    if data.get("uploaded") is False:
        message = _optional_str(data.get("message")) or "upload failed"
        return f"Pyromind storage {api_name} API failed: {message}"
    return data


def _format_api_failure(api_name: str, payload: dict[str, Any]) -> str:
    message = _optional_str(payload.get("message")) or "unknown API failure"
    error_code = _optional_str(payload.get("error_code"))
    if error_code:
        return f"Pyromind storage {api_name} API failed with {error_code}: {message}"
    return f"Pyromind storage {api_name} API failed: {message}"


def storage_file_names(
    *,
    directory: str,
    storage_base_url: str,
    headers: dict[str, str],
    timeout: float,
    search: str = "",
) -> set[str]:
    """Names present in one Storage directory; empty when the lookup fails.

    Callers use this for advisory checks such as overwrite warnings, so an
    unreachable listing API must never block the operation it guards.
    """
    try:
        response = httpx.post(
            f"{storage_base_url.rstrip('/')}/file_list",
            headers=headers,
            json={"path": directory, "search": search},
            timeout=timeout,
        )
    except httpx.RequestError:
        return set()
    payload = _decode_json_response(response, "Pyromind storage file_list API")
    if isinstance(payload, str):
        return set()
    data = _extract_api_data("file_list", payload)
    if isinstance(data, str) or not isinstance(data.get("list"), list):
        return set()
    names: set[str] = set()
    for item in data["list"]:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            names.add(name)
            continue
        item_path = item.get("path")
        if isinstance(item_path, str) and item_path:
            names.add(PurePosixPath(item_path).name)
    return names


def _strip_workspace_prefix(path: str) -> str:
    """Strip a leading platform workspace prefix, yielding a storage path."""
    for prefix in _WORKSPACE_PATH_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _truncate_text(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} characters truncated]"


register_tool(UploadFileToPyromindTool.name, UploadFileToPyromindTool)
register_tool(GetStorageUrlTool.name, GetStorageUrlTool)
