"""Definitions of the preview_dataset and upload_file_to_pyromind tools."""

from __future__ import annotations

import csv
import json
import os
import random
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
from pydantic import AliasChoices, Field
from rich.text import Text

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


PRE_STORAGE_API_BASE_URL = "https://pre-api-portal.pyromind.ai/storage_api"
PROD_STORAGE_API_BASE_URL = "https://api-portal.pyromind.ai/storage_api"
PYROMIND_STORAGE_AUTH_COOKIE_SECRET = "PYROMIND_STORAGE_AUTH_COOKIE"
PYROMIND_STORAGE_HEADERS_STATE_KEY = "pyromind_storage_headers"
DEFAULT_UPLOAD_TARGET_DIR = "/agentTest"

_PROD_APP_ENVS = {"prod", "production", "online"}
_DEFAULT_PREVIEW_BYTES = 100 * 1024
_MAX_PREVIEW_BYTES = 100 * 1024
_MAX_REQUESTED_SAMPLES = 100
_MAX_RANGE_REQUESTS = 8
_MIN_RANGE_BYTES = 4096
_DELIMITED_HEADER_BYTES = 4096
_MAX_SAMPLE_STRING_CHARS = 2000
_MAX_SAMPLE_CONTAINER_ITEMS = 20
_TEXT_PREVIEW_SUFFIXES = {".txt", ".md", ".log", ".sh"}
_SUPPORTED_PREVIEW_SUFFIXES = {
    ".jsonl",
    ".json",
    ".csv",
    ".tsv",
    *_TEXT_PREVIEW_SUFFIXES,
}
_VISION_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
}
_VISION_FIELD_MARKERS = ("image", "video", "vision")
SampleStrategy = Literal["head", "tail", "random", "stratified"]


@dataclass(frozen=True)
class _PreviewChunk:
    content: bytes
    starts_at_zero: bool
    ends_at_eof: bool


def _default_storage_base_url() -> str:
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    if app_env in _PROD_APP_ENVS:
        return PROD_STORAGE_API_BASE_URL
    return PRE_STORAGE_API_BASE_URL


# ---------------------------------------------------------------------------
# preview_dataset
# ---------------------------------------------------------------------------


class PreviewDatasetAction(Action):
    """Preview a dataset stored on Pyromind storage."""

    dataset_path: str = Field(
        description=(
            "Relative path of the dataset on Pyromind storage, as pasted by "
            "the user (e.g. 'datasets/my_data/' or "
            "'datasets/my_data/train.jsonl'). Directories and single files "
            "are both accepted."
        ),
    )
    n: int = Field(
        default=10,
        description=(
            "Maximum number of sample rows or text lines to return (1-100). "
            "Defaults to 10; large truncated files may return fewer."
        ),
        ge=1,
        le=_MAX_REQUESTED_SAMPLES,
        validation_alias=AliasChoices("n", "max_samples"),
    )
    sample_strategy: SampleStrategy = Field(
        default="head",
        description=(
            "Sampling strategy: head for first rows, tail for last rows, random "
            "for random byte-range line samples, or stratified for head/middle/tail "
            "byte-range line samples."
        ),
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Preview dataset: ", style="bold blue")
        content.append(self.dataset_path)
        return content


class PreviewDatasetObservation(Observation):
    """Statistics and sample rows of a storage dataset."""

    dataset_path: str = Field(description="The storage path that was previewed.")
    files: list[str] = Field(
        default_factory=list,
        description="Data files found under the path (relative to it).",
    )
    num_rows: int | None = Field(
        default=None, description="Total row count across data files."
    )
    columns: list[str] = Field(
        default_factory=list, description="Top-level fields of each row."
    )
    p95_sequence_length: int | None = Field(
        default=None,
        description="P95 of per-row token length (prompt + response).",
    )
    has_vision: bool | None = Field(
        default=None, description="Whether rows contain image/video fields."
    )
    sample_rows: list[dict[str, Any]] = Field(
        default_factory=list, description="A few raw sample rows."
    )
    sample_file_path: str | None = Field(
        default=None,
        description=(
            "Absolute local path of the saved sample file under the conversation "
            "preview_dataset directory."
        ),
    )
    sample_size: int | None = Field(
        default=None,
        description="Size in bytes of the saved sample file.",
    )
    requested_rows: int | None = Field(
        default=None,
        description="Maximum sample rows requested by the caller.",
    )
    sample_strategy: str | None = Field(
        default=None,
        description="Sampling strategy used for the saved sample file.",
    )
    object_name: str | None = Field(
        default=None, description="Storage object name returned by the metadata API."
    )
    bucket_name: str | None = Field(
        default=None, description="Storage bucket returned by the metadata API."
    )
    size: int | None = Field(default=None, description="Object size in bytes.")
    content_type: str | None = Field(
        default=None, description="Object MIME type returned by storage."
    )
    is_dir: bool | None = Field(
        default=None, description="Whether the previewed storage path is a directory."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Object metadata returned by storage."
    )
    preview_file_path: str | None = Field(
        default=None,
        description=(
            "Concrete file path used for content sampling when dataset_path is a "
            "directory."
        ),
    )
    previewed_rows: int | None = Field(
        default=None,
        description=(
            "Rows or lines inspected in the bounded preview. For truncated large "
            "files this is not the total row count."
        ),
    )
    preview_truncated: bool = Field(
        default=False,
        description="Whether content sampling stopped at the preview byte limit.",
    )
    preview_error: str | None = Field(
        default=None,
        description="Non-fatal content parsing issue, when metadata lookup succeeded.",
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Dataset preview: ", style="bold green")
        content.append(self.dataset_path)
        if self.num_rows is not None:
            content.append(f"\nrows={self.num_rows}")
        if self.columns:
            content.append(f"\ncolumns={', '.join(self.columns)}")
        if self.sample_file_path:
            content.append(f"\nsample_file_path={self.sample_file_path}")
        return content


_PREVIEW_DATASET_DESCRIPTION = """Preview a Pyromind storage dataset.

Given the storage path the user pasted into the chat, this calls the Pyromind
storage API and returns:
- the data files found under the path
- storage metadata such as object name, bucket, size, content type, and is_dir
- exact row count when the file fits inside the bounded preview
- previewed row count and a saved sample file path for JSONL, JSON, CSV/TSV,
  and text
- configurable sampling with `sample_strategy`: `head`, `tail`, `random`, or
  `stratified`

Call this BEFORE generating any training workflow that uses a user-provided
dataset, so you can determine the data format (SFT messages / prompt-response
/ DPO chosen-rejected / GRPO prompt-only), pick the right dataset config
builder node, and fill in field-mapping parameters from real field names
instead of guessing.

Sample content is saved under the current conversation directory's
`preview_dataset/` folder, next to `workflow_canvas/`. Large files are sampled
with one or more HTTP byte-range requests under a total 100KB byte cap. When
preview_truncated=true, num_rows is left unset because the tool did not read the
full file; use previewed_rows and the saved sample file only as a format hint.
"""


class PreviewDatasetExecutor(
    ToolExecutor[PreviewDatasetAction, PreviewDatasetObservation]
):
    """Preview Pyromind storage files through the platform storage API."""

    def __init__(
        self,
        storage_base_url: str | None = None,
        headers: dict[str, str] | None = None,
        secret_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        max_preview_bytes: int = _DEFAULT_PREVIEW_BYTES,
    ) -> None:
        base_url = storage_base_url or _default_storage_base_url()
        self._storage_base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})
        self._secret_headers = dict(secret_headers or {})
        self._timeout = timeout
        self._max_preview_bytes = min(max_preview_bytes, _MAX_PREVIEW_BYTES)

    def __call__(
        self,
        action: PreviewDatasetAction,
        conversation: BaseConversation | None = None,
    ) -> PreviewDatasetObservation:
        try:
            headers = self._resolve_headers(conversation, json_content=True)
        except ValueError as exc:
            return PreviewDatasetObservation.from_text(text=str(exc), is_error=True)

        dataset_path = action.dataset_path.strip()
        if not dataset_path:
            return PreviewDatasetObservation.from_text(
                text="dataset_path must be a non-empty storage path.",
                is_error=True,
                dataset_path=action.dataset_path,
            )

        files: list[str] = []
        preview_path = dataset_path
        metadata: dict[str, Any] | None = None

        if _looks_like_directory(dataset_path):
            list_result = self._list_files(dataset_path, headers)
            if isinstance(list_result, PreviewDatasetObservation):
                return list_result
            files = list_result
            preview_path = _select_preview_file(files) or ""
            if not preview_path:
                return PreviewDatasetObservation.from_text(
                    text=f"No previewable files found under {dataset_path}.",
                    dataset_path=dataset_path,
                    files=files,
                    is_dir=True,
                )

        metadata_result = self._get_metadata(preview_path, headers)
        if isinstance(metadata_result, PreviewDatasetObservation):
            return metadata_result
        metadata = metadata_result

        if metadata.get("is_dir") is True:
            list_result = self._list_files(dataset_path, headers)
            if isinstance(list_result, PreviewDatasetObservation):
                return list_result
            files = list_result
            preview_path = _select_preview_file(files) or ""
            if not preview_path:
                return PreviewDatasetObservation.from_text(
                    text=f"No previewable files found under {dataset_path}.",
                    dataset_path=dataset_path,
                    files=files,
                    is_dir=True,
                    **_metadata_observation_fields(metadata),
                )
            metadata_result = self._get_metadata(preview_path, headers)
            if isinstance(metadata_result, PreviewDatasetObservation):
                return metadata_result
            metadata = metadata_result

        if not files:
            files = [preview_path]

        preview_url_result = self._get_download_url(preview_path, headers)
        if isinstance(preview_url_result, PreviewDatasetObservation):
            return preview_url_result

        size = _optional_int(metadata.get("size"))
        content_result = self._download_preview(
            preview_url_result,
            preview_path,
            size,
            str(metadata.get("content_type") or ""),
            action.sample_strategy,
            action.n,
        )
        if isinstance(content_result, PreviewDatasetObservation):
            return content_result
        preview_chunks, preview_truncated = content_result

        parsed = _parse_preview_chunks(
            preview_path,
            str(metadata.get("content_type") or ""),
            preview_chunks,
            max_samples=action.n,
            sample_strategy=action.sample_strategy,
            truncated=preview_truncated,
        )
        sample_result = _write_preview_sample(
            conversation,
            preview_path,
            str(metadata.get("content_type") or ""),
            parsed["sample_rows"],
            self._max_preview_bytes,
        )
        if isinstance(sample_result, PreviewDatasetObservation):
            return sample_result
        sample_file_path, sample_size, sample_rows = sample_result
        parsed["sample_rows"] = sample_rows
        parsed["columns"] = _collect_columns(sample_rows)
        text = _format_preview_text(
            dataset_path=dataset_path,
            preview_path=preview_path,
            files=files,
            metadata=metadata,
            parsed=parsed,
            sample_file_path=sample_file_path,
            sample_size=sample_size,
            requested_rows=action.n,
            sample_strategy=action.sample_strategy,
            preview_truncated=preview_truncated,
        )

        return PreviewDatasetObservation.from_text(
            text=text,
            dataset_path=dataset_path,
            files=files,
            num_rows=parsed["num_rows"],
            columns=parsed["columns"],
            p95_sequence_length=None,
            has_vision=_infer_has_vision(parsed["columns"], parsed["sample_rows"]),
            sample_rows=parsed["sample_rows"],
            sample_file_path=sample_file_path,
            sample_size=sample_size,
            requested_rows=action.n,
            sample_strategy=action.sample_strategy,
            preview_file_path=preview_path,
            previewed_rows=parsed["previewed_rows"],
            preview_truncated=preview_truncated,
            preview_error=parsed["preview_error"],
            **_metadata_observation_fields(metadata),
        )

    def _get_metadata(
        self,
        path: str,
        headers: dict[str, str],
    ) -> dict[str, Any] | PreviewDatasetObservation:
        payload_result = self._post_json("get_file_metadata", {"path": path}, headers)
        if isinstance(payload_result, str):
            return PreviewDatasetObservation.from_text(
                text=payload_result,
                is_error=True,
                dataset_path=path,
            )
        data_result = _extract_api_data("get_file_metadata", payload_result)
        if isinstance(data_result, str):
            return PreviewDatasetObservation.from_text(
                text=data_result,
                is_error=True,
                dataset_path=path,
            )
        return data_result

    def _list_files(
        self,
        path: str,
        headers: dict[str, str],
    ) -> list[str] | PreviewDatasetObservation:
        payload_result = self._post_json("file_list", {"path": path}, headers)
        if isinstance(payload_result, str):
            return PreviewDatasetObservation.from_text(
                text=payload_result,
                is_error=True,
                dataset_path=path,
            )
        data_result = _extract_api_data("file_list", payload_result)
        if isinstance(data_result, str):
            return PreviewDatasetObservation.from_text(
                text=data_result,
                is_error=True,
                dataset_path=path,
            )

        raw_files = data_result.get("list")
        if not isinstance(raw_files, list):
            return PreviewDatasetObservation.from_text(
                text="Pyromind storage file_list API response is missing list data.",
                is_error=True,
                dataset_path=path,
            )

        files: list[str] = []
        for item in raw_files:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").lower() != "file":
                continue
            item_path = item.get("path")
            if item_path is not None:
                files.append(str(item_path))
        return files

    def _get_download_url(
        self,
        path: str,
        headers: dict[str, str],
    ) -> str | PreviewDatasetObservation:
        payload_result = self._post_json("get_url", {"path": path}, headers)
        if isinstance(payload_result, str):
            return PreviewDatasetObservation.from_text(
                text=payload_result,
                is_error=True,
                dataset_path=path,
            )
        data_result = _extract_api_data("get_url", payload_result)
        if isinstance(data_result, str):
            return PreviewDatasetObservation.from_text(
                text=data_result,
                is_error=True,
                dataset_path=path,
            )

        url = data_result.get("url")
        if not isinstance(url, str) or not url.strip():
            return PreviewDatasetObservation.from_text(
                text="Pyromind storage get_url API response is missing url data.",
                is_error=True,
                dataset_path=path,
            )
        return url

    def _post_json(
        self,
        route: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any] | str:
        try:
            response = httpx.post(
                f"{self._storage_base_url}/{route}",
                headers=headers,
                json=body,
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            return (
                f"Failed to call Pyromind storage {route} API: "
                f"{type(exc).__name__}: {exc}"
            )
        return _decode_json_response(response, f"Pyromind storage {route} API")

    def _download_preview(
        self,
        url: str,
        dataset_path: str,
        file_size: int | None,
        content_type: str,
        sample_strategy: SampleStrategy,
        requested_rows: int,
    ) -> tuple[list[_PreviewChunk], bool] | PreviewDatasetObservation:
        kind = _preview_kind(dataset_path, content_type)
        ranges = _build_preview_ranges(
            file_size=file_size,
            max_bytes=self._max_preview_bytes,
            sample_strategy=sample_strategy,
            requested_rows=requested_rows,
            preview_kind=kind,
        )
        chunks: list[_PreviewChunk] = []
        truncated = file_size is None or file_size > self._max_preview_bytes
        for start, end in ranges:
            chunk_result = self._download_range(url, dataset_path, start, end)
            if isinstance(chunk_result, PreviewDatasetObservation):
                return chunk_result
            chunks.append(
                _PreviewChunk(
                    content=chunk_result,
                    starts_at_zero=start == 0,
                    ends_at_eof=file_size is not None and end >= file_size - 1,
                )
            )
        return chunks, truncated

    def _download_range(
        self,
        url: str,
        dataset_path: str,
        start: int,
        end: int,
    ) -> bytes | PreviewDatasetObservation:
        chunks = bytearray()
        max_bytes = end - start + 1
        headers = {"range": f"bytes={start}-{end}"}
        try:
            with httpx.stream(
                "GET",
                url,
                headers=headers,
                timeout=self._timeout,
                follow_redirects=True,
            ) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    return PreviewDatasetObservation.from_text(
                        text=(
                            "Pyromind storage download URL returned HTTP "
                            f"{response.status_code}: {_truncate_text(body)}"
                        ),
                        is_error=True,
                        dataset_path=dataset_path,
                    )
                for chunk in response.iter_bytes():
                    remaining = max_bytes - len(chunks)
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunks.extend(chunk[:remaining])
                        break
                    chunks.extend(chunk)
        except httpx.RequestError as exc:
            return PreviewDatasetObservation.from_text(
                text=(
                    "Failed to download Pyromind storage preview bytes: "
                    f"{type(exc).__name__}: {exc}"
                ),
                is_error=True,
                dataset_path=dataset_path,
            )
        return bytes(chunks)

    def _resolve_headers(
        self,
        conversation: BaseConversation | None,
        *,
        json_content: bool,
    ) -> dict[str, str]:
        headers = {"accept": "*/*", **self._headers}
        if json_content:
            headers["content-type"] = "application/json"
        headers.update(_resolve_conversation_headers(conversation))
        headers.update(_resolve_secret_headers(conversation, self._secret_headers))
        return headers


class PreviewDatasetTool(
    ToolDefinition[PreviewDatasetAction, PreviewDatasetObservation]
):
    """Tool for previewing datasets on Pyromind storage."""

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
        max_preview_bytes = int(params.pop("max_preview_bytes", _DEFAULT_PREVIEW_BYTES))
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"PreviewDatasetTool got unknown params: {names}")
        _validate_storage_tool_params(
            storage_base_url,
            headers,
            secret_headers,
            timeout,
            max_preview_bytes,
        )
        return [
            cls(
                description=_PREVIEW_DATASET_DESCRIPTION,
                action_type=PreviewDatasetAction,
                observation_type=PreviewDatasetObservation,
                executor=PreviewDatasetExecutor(
                    storage_base_url=storage_base_url,
                    headers=_normalize_headers(headers),
                    secret_headers=_normalize_headers(secret_headers),
                    timeout=timeout,
                    max_preview_bytes=max_preview_bytes,
                ),
                annotations=ToolAnnotations(
                    title="preview_dataset",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


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
    target_dir: str = Field(
        default=DEFAULT_UPLOAD_TARGET_DIR,
        description="Storage directory to upload into. Defaults to /agentTest.",
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
            "parameters (e.g. /agentTest/acc.py)."
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

Use this when a workflow node needs a server-side file path, most commonly a
custom evaluation metric or reward script for MetricsConfigBuilderCustomNode:
write the Python file locally first, upload it with this tool, then use the
returned storage path in the node's `entry` parameter as
`<storage_path>:<function_name>` (e.g. /agentTest/acc.py:acc_func).

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
            local_path = _resolve_workspace_file(action.file_path, conversation)
            headers = self._resolve_headers(conversation)
        except ValueError as exc:
            return UploadFileToPyromindObservation.from_text(
                text=str(exc),
                is_error=True,
            )

        filename = local_path.name
        storage_path = str(PurePosixPath(action.target_dir) / filename)
        try:
            with local_path.open("rb") as file_obj:
                response = httpx.post(
                    f"{self._storage_base_url}/upload_file",
                    headers=headers,
                    data={
                        "name": filename,
                        "path": action.target_dir,
                        "bucket": "",
                    },
                    files={"file": (filename, file_obj, "application/octet-stream")},
                    timeout=self._timeout,
                )
        except httpx.RequestError as exc:
            return UploadFileToPyromindObservation.from_text(
                text=(
                    "Failed to call Pyromind storage upload_file API: "
                    f"{type(exc).__name__}: {exc}"
                ),
                is_error=True,
            )
        except OSError as exc:
            return UploadFileToPyromindObservation.from_text(
                text=f"Failed to read file for upload: {exc}",
                is_error=True,
            )

        payload_result = _decode_json_response(
            response,
            "Pyromind storage upload_file API",
        )
        if isinstance(payload_result, str):
            return UploadFileToPyromindObservation.from_text(
                text=payload_result,
                is_error=True,
            )
        data_result = _extract_api_data("upload_file", payload_result)
        if isinstance(data_result, str):
            return UploadFileToPyromindObservation.from_text(
                text=data_result,
                is_error=True,
            )

        failed_files = data_result.get("failed_files")
        if isinstance(failed_files, list) and failed_files:
            return UploadFileToPyromindObservation.from_text(
                text=(
                    "Pyromind storage upload_file API reported failed files: "
                    f"{_truncate_text(json.dumps(failed_files, ensure_ascii=False))}"
                ),
                is_error=True,
            )

        success_count = data_result.get("success_count")
        if success_count != 1:
            return UploadFileToPyromindObservation.from_text(
                text=(
                    "Pyromind storage upload_file API did not report one uploaded "
                    f"file: {json.dumps(data_result, ensure_ascii=False)}"
                ),
                is_error=True,
            )

        return UploadFileToPyromindObservation.from_text(
            text=f"File uploaded to Pyromind storage: {storage_path}",
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
            _DEFAULT_PREVIEW_BYTES,
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


def _validate_storage_tool_params(
    storage_base_url: str,
    headers: Any,
    secret_headers: Any,
    timeout: float,
    max_preview_bytes: int,
) -> None:
    if not storage_base_url.strip():
        raise ValueError("storage_base_url must be a non-empty string")
    if headers is not None and not isinstance(headers, dict):
        raise ValueError("headers must be a dictionary when provided")
    if secret_headers is not None and not isinstance(secret_headers, dict):
        raise ValueError("secret_headers must be a dictionary when provided")
    if timeout <= 0:
        raise ValueError("timeout must be greater than 0")
    if max_preview_bytes <= 0:
        raise ValueError("max_preview_bytes must be greater than 0")


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
    if not resolved.is_file():
        raise ValueError(f"Cannot upload missing workspace file: {file_path}")
    return resolved


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


def _metadata_observation_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    raw_metadata = metadata.get("metadata")
    return {
        "object_name": _optional_str(metadata.get("object_name")),
        "bucket_name": _optional_str(metadata.get("bucket_name")),
        "size": _optional_int(metadata.get("size")),
        "content_type": _optional_str(metadata.get("content_type")),
        "is_dir": (
            metadata.get("is_dir") if isinstance(metadata.get("is_dir"), bool) else None
        ),
        "metadata": raw_metadata if isinstance(raw_metadata, dict) else {},
    }


def _looks_like_directory(path: str) -> bool:
    return path.endswith("/")


def _select_preview_file(files: list[str]) -> str | None:
    for file_path in files:
        if PurePosixPath(file_path).suffix.lower() in _SUPPORTED_PREVIEW_SUFFIXES:
            return file_path
    return files[0] if files else None


def _build_preview_ranges(
    *,
    file_size: int | None,
    max_bytes: int,
    sample_strategy: SampleStrategy,
    requested_rows: int,
    preview_kind: str | None,
) -> list[tuple[int, int]]:
    if file_size is None or file_size <= max_bytes:
        return [(0, max_bytes - 1)]

    if sample_strategy == "head":
        return [(0, max_bytes - 1)]

    include_header = preview_kind in {"csv", "tsv"}
    header_range: list[tuple[int, int]] = []
    remaining_budget = max_bytes
    min_sample_start = 0
    if include_header:
        header_bytes = min(_DELIMITED_HEADER_BYTES, max(1, max_bytes // 4), file_size)
        header_range = [(0, header_bytes - 1)]
        remaining_budget -= header_bytes
        min_sample_start = header_bytes

    if remaining_budget <= 0:
        return header_range or [(0, max_bytes - 1)]

    if sample_strategy == "tail":
        start = max(min_sample_start, file_size - remaining_budget)
        return header_range + [(start, file_size - 1)]

    range_count = min(
        _MAX_RANGE_REQUESTS,
        max(1, requested_rows),
        max(1, remaining_budget // _MIN_RANGE_BYTES),
    )
    range_size = max(1, remaining_budget // range_count)
    max_start = max(min_sample_start, file_size - range_size)

    if sample_strategy == "stratified":
        starts = _stratified_starts(min_sample_start, max_start, range_count)
    else:
        starts = _random_starts(min_sample_start, max_start, range_count)

    ranges = [(start, min(file_size - 1, start + range_size - 1)) for start in starts]
    return header_range + ranges


def _stratified_starts(start: int, end: int, count: int) -> list[int]:
    if count <= 1 or start >= end:
        return [start]
    return [
        start + round((end - start) * index / (count - 1)) for index in range(count)
    ]


def _random_starts(start: int, end: int, count: int) -> list[int]:
    if count <= 1 or start >= end:
        return [start]
    starts: set[int] = set()
    attempts = count * 4
    for _ in range(attempts):
        starts.add(random.randint(start, end))
        if len(starts) >= count:
            break
    while len(starts) < count:
        starts.add(start + round((end - start) * len(starts) / count))
    return sorted(starts)


def _parse_preview_bytes(
    file_path: str,
    content_type: str,
    content: bytes,
    *,
    max_samples: int,
    truncated: bool,
) -> dict[str, Any]:
    return _parse_preview_chunks(
        file_path,
        content_type,
        [
            _PreviewChunk(
                content=content, starts_at_zero=True, ends_at_eof=not truncated
            )
        ],
        max_samples=max_samples,
        sample_strategy="head",
        truncated=truncated,
    )


def _parse_preview_chunks(
    file_path: str,
    content_type: str,
    chunks: list[_PreviewChunk],
    *,
    max_samples: int,
    sample_strategy: SampleStrategy,
    truncated: bool,
) -> dict[str, Any]:
    kind = _preview_kind(file_path, content_type)
    if kind is None:
        return {
            "num_rows": None,
            "columns": [],
            "sample_rows": [],
            "previewed_rows": 0,
            "preview_error": (
                "Content preview skipped because the file type is not a supported "
                "text, JSON, JSONL, CSV, or TSV format."
            ),
        }

    lines = _complete_lines_from_chunks(chunks)
    sample_limit = _effective_sample_limit(max_samples, truncated)
    if kind == "json" and not truncated and len(chunks) == 1:
        text = chunks[0].content.decode("utf-8", errors="replace")
        return _parse_json_preview(text, sample_limit, sample_strategy)
    if kind == "json":
        preview = _parse_text_lines(lines, sample_limit, sample_strategy, truncated)
        preview["preview_error"] = (
            "JSON file was larger than the preview byte limit, so it was sampled "
            "as text chunks instead of parsed as complete JSON."
        )
        return preview
    if kind == "jsonl":
        return _parse_jsonl_lines(lines, sample_limit, sample_strategy, truncated)
    if kind in {"csv", "tsv"}:
        delimiter = "\t" if kind == "tsv" else ","
        return _parse_delimited_lines(
            lines, sample_limit, sample_strategy, truncated, delimiter
        )
    return _parse_text_lines(lines, sample_limit, sample_strategy, truncated)


def _preview_kind(file_path: str, content_type: str) -> str | None:
    suffix = PurePosixPath(file_path).suffix.lower()
    normalized_content_type = content_type.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json" or "json" in normalized_content_type:
        return "json"
    if suffix == ".csv" or "csv" in normalized_content_type:
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix in _TEXT_PREVIEW_SUFFIXES or normalized_content_type.startswith("text/"):
        return "text"
    return None


def _parse_jsonl_preview(
    text: str,
    sample_limit: int,
    truncated: bool,
) -> dict[str, Any]:
    lines = _complete_preview_lines(text, truncated)
    return _parse_jsonl_lines(lines, sample_limit, "head", truncated)


def _parse_jsonl_lines(
    lines: list[str],
    sample_limit: int,
    sample_strategy: SampleStrategy,
    truncated: bool,
) -> dict[str, Any]:
    non_empty_lines = [line for line in lines if line.strip()]
    indexed_lines = list(enumerate(non_empty_lines, start=1))
    sampled_lines = _sample_sequence(indexed_lines, sample_limit, sample_strategy)
    sample_rows: list[dict[str, Any]] = []
    preview_error = None
    for index, line in sampled_lines:
        try:
            sample_rows.append(_sample_row(json.loads(line)))
        except json.JSONDecodeError as exc:
            preview_error = f"Failed to parse JSONL line {index}: {exc.msg}"
            sample_rows.append(
                {"line": index, "raw": _truncate_text(line, _MAX_SAMPLE_STRING_CHARS)}
            )

    previewed_rows = len(non_empty_lines)
    return {
        "num_rows": previewed_rows if not truncated else None,
        "columns": _collect_columns(sample_rows),
        "sample_rows": sample_rows,
        "previewed_rows": previewed_rows,
        "preview_error": preview_error,
    }


def _parse_json_preview(
    text: str,
    sample_limit: int,
    sample_strategy: SampleStrategy,
) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = _parse_text_preview(text, sample_limit, False)
        preview["preview_error"] = f"Failed to parse JSON: {exc.msg}"
        return preview

    if isinstance(value, list):
        rows = [_sample_row(item) for item in value]
    else:
        rows = [_sample_row(value)]
    sample_rows = _sample_sequence(rows, sample_limit, sample_strategy)
    return {
        "num_rows": len(rows),
        "columns": _collect_columns(sample_rows),
        "sample_rows": sample_rows,
        "previewed_rows": len(rows),
        "preview_error": None,
    }


def _parse_delimited_preview(
    text: str,
    sample_limit: int,
    truncated: bool,
    delimiter: str,
) -> dict[str, Any]:
    lines = _complete_preview_lines(text, truncated)
    return _parse_delimited_lines(lines, sample_limit, "head", truncated, delimiter)


def _parse_delimited_lines(
    lines: list[str],
    sample_limit: int,
    sample_strategy: SampleStrategy,
    truncated: bool,
    delimiter: str,
) -> dict[str, Any]:
    if not lines:
        return {
            "num_rows": 0 if not truncated else None,
            "columns": [],
            "sample_rows": [],
            "previewed_rows": 0,
            "preview_error": None,
        }

    header = lines[0]
    data_lines = _sample_sequence(lines[1:], sample_limit, sample_strategy)
    reader = csv.DictReader([header, *data_lines], delimiter=delimiter)
    sample_rows = [
        _sample_row(dict(row))
        for index, row in enumerate(reader)
        if index < sample_limit
    ]
    previewed_rows = max(len(lines) - 1, 0)
    columns = [str(name) for name in reader.fieldnames or []]
    return {
        "num_rows": previewed_rows if not truncated else None,
        "columns": columns,
        "sample_rows": sample_rows,
        "previewed_rows": previewed_rows,
        "preview_error": None,
    }


def _parse_text_preview(
    text: str,
    sample_limit: int,
    truncated: bool,
) -> dict[str, Any]:
    lines = _complete_preview_lines(text, truncated)
    return _parse_text_lines(lines, sample_limit, "head", truncated)


def _parse_text_lines(
    lines: list[str],
    sample_limit: int,
    sample_strategy: SampleStrategy,
    truncated: bool,
) -> dict[str, Any]:
    indexed_lines = list(enumerate(lines, start=1))
    sampled_lines = _sample_sequence(indexed_lines, sample_limit, sample_strategy)
    sample_rows = [
        {"line": index, "text": _truncate_text(line, _MAX_SAMPLE_STRING_CHARS)}
        for index, line in sampled_lines
    ]
    previewed_rows = len(lines)
    return {
        "num_rows": previewed_rows if not truncated else None,
        "columns": ["line", "text"],
        "sample_rows": sample_rows,
        "previewed_rows": previewed_rows,
        "preview_error": None,
    }


def _complete_preview_lines(text: str, truncated: bool) -> list[str]:
    lines = text.splitlines()
    if truncated and text and not text.endswith(("\n", "\r")) and lines:
        return lines[:-1]
    return lines


def _complete_lines_from_chunks(chunks: list[_PreviewChunk]) -> list[str]:
    lines: list[str] = []
    for chunk in chunks:
        text = chunk.content.decode("utf-8", errors="replace")
        chunk_lines = text.splitlines()
        if not chunk.starts_at_zero and chunk_lines:
            chunk_lines = chunk_lines[1:]
        if not chunk.ends_at_eof and text and not text.endswith(("\n", "\r")):
            chunk_lines = chunk_lines[:-1]
        lines.extend(chunk_lines)
    return lines


def _effective_sample_limit(max_samples: int, _truncated: bool) -> int:
    return max_samples


def _sample_sequence(
    values: list[Any],
    sample_limit: int,
    sample_strategy: SampleStrategy,
) -> list[Any]:
    if sample_limit <= 0 or not values:
        return []
    if sample_strategy == "head":
        return values[:sample_limit]
    if sample_strategy == "tail":
        return values[-sample_limit:]
    if sample_strategy == "random":
        count = min(sample_limit, len(values))
        indexes = sorted(random.sample(range(len(values)), count))
        return [values[index] for index in indexes]
    return _stratified_values(values, sample_limit)


def _stratified_values(values: list[Any], sample_limit: int) -> list[Any]:
    count = min(sample_limit, len(values))
    if count <= 0:
        return []
    if count == 1:
        return [values[0]]
    indexes = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indexes]


def _sample_row(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        trimmed = _trim_value(value)
        return trimmed if isinstance(trimmed, dict) else {"value": trimmed}
    return {"value": _trim_value(value)}


def _trim_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, _MAX_SAMPLE_STRING_CHARS)
    if isinstance(value, list):
        trimmed = [_trim_value(item) for item in value[:_MAX_SAMPLE_CONTAINER_ITEMS]]
        if len(value) > _MAX_SAMPLE_CONTAINER_ITEMS:
            trimmed.append(f"... [{len(value) - _MAX_SAMPLE_CONTAINER_ITEMS} more]")
        return trimmed
    if isinstance(value, dict):
        items = list(value.items())
        trimmed_dict = {
            str(key): _trim_value(item_value)
            for key, item_value in items[:_MAX_SAMPLE_CONTAINER_ITEMS]
        }
        if len(items) > _MAX_SAMPLE_CONTAINER_ITEMS:
            trimmed_dict["..."] = f"[{len(items) - _MAX_SAMPLE_CONTAINER_ITEMS} more]"
        return trimmed_dict
    return value


def _collect_columns(sample_rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in sample_rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)
    return columns


def _infer_has_vision(
    columns: list[str],
    sample_rows: list[dict[str, Any]],
) -> bool | None:
    if not columns and not sample_rows:
        return None
    for column in columns:
        column_lower = column.lower()
        if any(marker in column_lower for marker in _VISION_FIELD_MARKERS):
            return True
    return any(_value_mentions_vision(row) for row in sample_rows)


def _value_mentions_vision(value: Any) -> bool:
    if isinstance(value, str):
        suffix = PurePosixPath(value.split("?", 1)[0]).suffix.lower()
        return suffix in _VISION_SUFFIXES
    if isinstance(value, list):
        return any(_value_mentions_vision(item) for item in value)
    if isinstance(value, dict):
        return any(_value_mentions_vision(item) for item in value.values())
    return False


def _write_preview_sample(
    conversation: BaseConversation | None,
    preview_path: str,
    content_type: str,
    sample_rows: list[dict[str, Any]],
    max_bytes: int,
) -> tuple[str | None, int | None, list[dict[str, Any]]] | PreviewDatasetObservation:
    if not sample_rows:
        return None, None, sample_rows
    if conversation is None:
        return None, None, sample_rows

    workspace = cast(Any, conversation).workspace
    workspace_dir = Path(workspace.working_dir).resolve()
    sample_dir = workspace_dir / "preview_dataset"

    sample_content, suffix, limited_rows = _render_sample_file_content(
        preview_path,
        content_type,
        sample_rows,
        max_bytes,
    )
    if not sample_content:
        return None, None, limited_rows

    sample_file = (
        sample_dir
        / f"{_safe_sample_file_stem(preview_path)}-sample-{uuid.uuid4().hex[:8]}"
        f"{suffix}"
    )
    try:
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_file.write_text(sample_content, encoding="utf-8")
    except OSError as exc:
        return PreviewDatasetObservation.from_text(
            text=f"Failed to write preview sample file: {exc}",
            is_error=True,
            dataset_path=preview_path,
        )
    return str(sample_file), len(sample_content.encode("utf-8")), limited_rows


def _safe_sample_file_stem(path: str) -> str:
    name = PurePosixPath(path.rstrip("/")).name or "dataset"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return (safe or "dataset")[:60]


def _render_sample_file_content(
    preview_path: str,
    content_type: str,
    sample_rows: list[dict[str, Any]],
    max_bytes: int,
) -> tuple[str, str, list[dict[str, Any]]]:
    suffix = ".txt" if _preview_kind(preview_path, content_type) == "text" else ".json"
    rows = list(sample_rows)
    while rows:
        content = _format_sample_file_content(rows, suffix)
        if len(content.encode("utf-8")) <= max_bytes:
            return content, suffix, rows
        rows = rows[:-1]
    return "", suffix, []


def _format_sample_file_content(sample_rows: list[dict[str, Any]], suffix: str) -> str:
    if suffix == ".txt":
        lines = [
            str(row["text"])
            for row in sample_rows
            if set(row) == {"line", "text"} and row.get("text") is not None
        ]
        return "\n".join(lines) + ("\n" if lines else "")
    return json.dumps(sample_rows, ensure_ascii=False) + "\n"


def _format_preview_text(
    *,
    dataset_path: str,
    preview_path: str,
    files: list[str],
    metadata: dict[str, Any],
    parsed: dict[str, Any],
    sample_file_path: str | None,
    sample_size: int | None,
    requested_rows: int,
    sample_strategy: SampleStrategy,
    preview_truncated: bool,
) -> str:
    fields = _metadata_observation_fields(metadata)
    parts = [
        f"Dataset preview: {dataset_path}",
        f"preview_file_path={preview_path}",
        f"files={len(files)}",
    ]
    if fields["size"] is not None:
        parts.append(f"size={fields['size']} bytes")
    if parsed["num_rows"] is not None:
        parts.append(f"rows={parsed['num_rows']}")
    else:
        parts.append(f"previewed_rows={parsed['previewed_rows']}")
    if parsed["columns"]:
        parts.append(f"columns={', '.join(parsed['columns'])}")
    parts.append(f"requested_rows={requested_rows}")
    parts.append(f"sample_strategy={sample_strategy}")
    parts.append(f"sample_rows={len(parsed['sample_rows'])}")
    if parsed["preview_error"]:
        parts.append(f"preview_error={parsed['preview_error']}")
    if sample_file_path:
        parts.append(f"sample_file_path={sample_file_path}")
    if sample_size is not None:
        parts.append(f"sample_size={sample_size} bytes")
    parts.append(f"preview_truncated={str(preview_truncated).lower()}")
    return "\n".join(parts)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truncate_text(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} characters truncated]"


register_tool(PreviewDatasetTool.name, PreviewDatasetTool)
register_tool(UploadFileToPyromindTool.name, UploadFileToPyromindTool)
