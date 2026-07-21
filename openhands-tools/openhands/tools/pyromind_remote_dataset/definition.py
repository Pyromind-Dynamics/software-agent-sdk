"""Preview remote datasets from HuggingFace or ModelScope."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import httpx
from pydantic import Field
from rich.text import Text

from openhands.sdk.llm.message import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


_SUPPORTED_DATA_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv", ".parquet"}
_TEXT_SUFFIXES = {".txt", ".md", ".log"}
_PREVIEW_SUFFIXES = _SUPPORTED_DATA_SUFFIXES | _TEXT_SUFFIXES
_SMALL_FILE_BYTES = 10 * 1024
_LARGE_FILE_PREVIEW_BYTES = 15 * 1024
_PARQUET_MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
_MAX_SAMPLE_ROWS = 100
_MAX_SAMPLE_STRING_CHARS = 2000
_HF_BASE_URL = "https://huggingface.co"
_MODELSCOPE_BASE_URL = "https://modelscope.cn/api/v1"


@dataclass(frozen=True)
class _RemoteFile:
    path: str
    size: int | None


class PreviewRemoteDatasetAction(Action):
    """Preview a dataset hosted on HuggingFace or ModelScope."""

    dataset_name: str = Field(
        description=(
            "Dataset identifier on the remote hub. For HuggingFace use "
            "'org/name' format (e.g. 'openai/gsm8k'). For ModelScope use "
            "the dataset id (e.g. 'modelscope/gsm8k')."
        ),
    )
    source: str | None = Field(
        default=None,
        description=(
            "Remote hub: 'huggingface' or 'modelscope'. If None (default), "
            "auto-detects by trying HuggingFace first, then ModelScope."
        ),
        pattern=r"^(huggingface|modelscope)$",
    )
    split: str | None = Field(
        default=None,
        description=(
            "Optional split name to preview (e.g. 'train', 'test'). "
            "If None, the first data file found is used."
        ),
    )
    n: int = Field(
        default=10,
        description="Maximum sample rows to return (1-100). Defaults to 10.",
        ge=1,
        le=_MAX_SAMPLE_ROWS,
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Preview remote dataset: ", style="bold blue")
        src = self.source or "auto"
        content.append(f"{src}/{self.dataset_name}")
        return content


class PreviewRemoteDatasetObservation(Observation):
    """Statistics and sample rows of a remote dataset."""

    dataset_name: str = Field(description="The dataset name that was previewed.")
    source: str = Field(description="Remote hub: 'huggingface' or 'modelscope'.")
    files: list[str] = Field(
        default_factory=list,
        description="Data files found in the dataset.",
    )
    splits: list[str] = Field(
        default_factory=list,
        description="Available splits (e.g. train, test, validation).",
    )
    num_rows: int | None = Field(
        default=None, description="Total row count if available."
    )
    columns: list[str] = Field(
        default_factory=list, description="Top-level fields of each row."
    )
    sample_rows: list[dict[str, Any]] = Field(
        default_factory=list, description="Sample rows from the dataset."
    )
    preview_file_path: str | None = Field(
        default=None,
        description="The specific file used for content sampling.",
    )
    previewed_rows: int | None = Field(
        default=None,
        description="Rows or lines inspected in the preview.",
    )
    preview_truncated: bool = Field(
        default=False,
        description="Whether content sampling was truncated.",
    )
    preview_error: str | None = Field(
        default=None,
        description="Non-fatal content parsing issue.",
    )
    has_vision: bool | None = Field(
        default=None, description="Whether rows contain image/video fields."
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Remote dataset preview: ", style="bold green")
        content.append(f"{self.source}/{self.dataset_name}")
        if self.num_rows is not None:
            content.append(f"\nrows={self.num_rows}")
        if self.columns:
            content.append(f"\ncolumns={', '.join(self.columns)}")
        return content


_PREVIEW_REMOTE_DESCRIPTION = """Preview a dataset hosted on HuggingFace or ModelScope.

Use this tool when the user wants to preview/inspect a dataset from HuggingFace
or ModelScope (e.g. 'openai/gsm8k', 'Qwen/Qwen2-Math-SFT'). If the dataset is
already in Pyromind shared space or user storage, use `preview_dataset` instead.

When `source` is not specified, the tool auto-detects by trying HuggingFace
first, then ModelScope. Use this auto mode when the user does not explicitly
mention the hub name.

This tool fetches dataset metadata and sample rows WITHOUT downloading the
full dataset. It uses:

- **HuggingFace**: `huggingface_hub` SDK to list files and read content.
- **ModelScope**: ModelScope REST API to list files and download content.

Supported file formats for content preview:
- Parquet (.parquet) — read via pyarrow, first N rows
- JSONL (.jsonl) — line-by-line parsing
- CSV (.csv) / TSV (.tsv) — header + data rows
- Text (.txt, .md, .log) — raw lines

When the dataset contains multiple data files:
- If `split` is specified, the first file matching that split is used.
- Otherwise, the first previewable file is auto-selected and the full file
  list is returned so you can call again with a specific file path.

Returns:
- files found in the dataset
- splits available (if detectable from file names)
- row count (exact for small files, estimated for large)
- column names and sample rows
"""


class PreviewRemoteDatasetExecutor(
    ToolExecutor[PreviewRemoteDatasetAction, PreviewRemoteDatasetObservation]
):
    """Preview datasets from HuggingFace or ModelScope."""

    def __init__(
        self,
        timeout: float = 60.0,
    ) -> None:
        self._timeout = timeout

    def __call__(
        self,
        action: PreviewRemoteDatasetAction,
        conversation: Any = None,  # noqa: ARG002
    ) -> PreviewRemoteDatasetObservation:
        dataset_name = action.dataset_name.strip()
        if not dataset_name:
            return PreviewRemoteDatasetObservation.from_text(
                text="dataset_name must be a non-empty string.",
                is_error=True,
                dataset_name=action.dataset_name,
                source=action.source or "auto",
            )

        source = action.source
        if source is None:
            hf_result = self._preview_huggingface(dataset_name, action.split, action.n)
            if not hf_result.is_error:
                return hf_result
            ms_result = self._preview_modelscope(dataset_name, action.split, action.n)
            if not ms_result.is_error:
                return ms_result
            return hf_result

        if source == "huggingface":
            return self._preview_huggingface(dataset_name, action.split, action.n)
        return self._preview_modelscope(dataset_name, action.split, action.n)

    # ------------------------------------------------------------------
    # HuggingFace
    # ------------------------------------------------------------------

    def _preview_huggingface(
        self,
        dataset_name: str,
        split: str | None,
        n: int,
    ) -> PreviewRemoteDatasetObservation:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return PreviewRemoteDatasetObservation.from_text(
                text="huggingface_hub is not installed.",
                is_error=True,
                dataset_name=dataset_name,
                source="huggingface",
            )

        api = HfApi()
        try:
            files = list(
                api.list_repo_tree(dataset_name, repo_type="dataset", recursive=True)
            )
        except Exception as exc:
            return PreviewRemoteDatasetObservation.from_text(
                text=(
                    f"Failed to list HuggingFace dataset '{dataset_name}': "
                    f"{type(exc).__name__}: {exc}"
                ),
                is_error=True,
                dataset_name=dataset_name,
                source="huggingface",
            )

        remote_files: list[_RemoteFile] = []
        for entry in files:
            path = getattr(entry, "path", None)
            if path is None:
                continue
            size = getattr(entry, "size", None)
            remote_files.append(_RemoteFile(path=str(path), size=size))

        return self._select_and_preview(
            remote_files, dataset_name, "huggingface", split, n
        )

    def _hf_download_file(
        self,
        dataset_name: str,
        file_path: str,
        max_bytes: int | None = None,
    ) -> bytes | str:
        """Download a file (or a portion) from HuggingFace.

        For small files or when max_bytes is None, uses hf_hub_download (cached).
        For large files with max_bytes, uses HTTP range request on the resolve URL.
        """
        if max_bytes is not None:
            url = f"{_HF_BASE_URL}/datasets/{dataset_name}/resolve/main/{file_path}"
            return self._http_range_download(url, max_bytes)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return "huggingface_hub is not installed."

        try:
            local_path = hf_hub_download(
                repo_id=dataset_name,
                filename=file_path,
                repo_type="dataset",
            )
        except Exception as exc:
            return f"Failed to download '{file_path}': {type(exc).__name__}: {exc}"

        try:
            with open(local_path, "rb") as f:
                return f.read()
        except OSError as exc:
            return f"Failed to read downloaded file: {exc}"

    # ------------------------------------------------------------------
    # ModelScope
    # ------------------------------------------------------------------

    def _preview_modelscope(
        self,
        dataset_name: str,
        split: str | None,
        n: int,
    ) -> PreviewRemoteDatasetObservation:
        try:
            response = httpx.get(
                f"{_MODELSCOPE_BASE_URL}/datasets/{dataset_name}/repo/tree",
                params={"Revision": "master", "Root": "/"},
                timeout=self._timeout,
            )
        except httpx.RequestError as exc:
            return PreviewRemoteDatasetObservation.from_text(
                text=(
                    f"Failed to list ModelScope dataset '{dataset_name}': "
                    f"{type(exc).__name__}: {exc}"
                ),
                is_error=True,
                dataset_name=dataset_name,
                source="modelscope",
            )

        if response.status_code >= 400:
            return PreviewRemoteDatasetObservation.from_text(
                text=(
                    f"ModelScope list API returned HTTP {response.status_code}: "
                    f"{_truncate_text(response.text)}"
                ),
                is_error=True,
                dataset_name=dataset_name,
                source="modelscope",
            )

        try:
            payload = response.json()
        except (json.JSONDecodeError, ValueError):
            return PreviewRemoteDatasetObservation.from_text(
                text="ModelScope list API returned invalid JSON.",
                is_error=True,
                dataset_name=dataset_name,
                source="modelscope",
            )

        data = payload.get("Data") or {}
        files_raw = data.get("Files") or []
        if not isinstance(files_raw, list):
            return PreviewRemoteDatasetObservation.from_text(
                text="ModelScope API response missing files list.",
                is_error=True,
                dataset_name=dataset_name,
                source="modelscope",
            )

        remote_files: list[_RemoteFile] = []
        for item in files_raw:
            if not isinstance(item, dict):
                continue
            path = item.get("Path")
            if path is None:
                continue
            size = item.get("Size")
            remote_files.append(
                _RemoteFile(
                    path=str(path),
                    size=int(size) if isinstance(size, (int, float)) else None,
                )
            )

        return self._select_and_preview(
            remote_files, dataset_name, "modelscope", split, n
        )

    def _ms_download_file(
        self,
        dataset_name: str,
        file_path: str,
        max_bytes: int | None = None,
    ) -> bytes | str:
        """Download a file (or a portion) from ModelScope."""
        url = (
            f"{_MODELSCOPE_BASE_URL}/datasets/{dataset_name}"
            f"/repo?Revision=master&FilePath={file_path}"
        )
        if max_bytes is not None:
            return self._http_range_download(url, max_bytes)
        try:
            response = httpx.get(url, timeout=self._timeout, follow_redirects=True)
            if response.status_code >= 400:
                return (
                    f"ModelScope download returned HTTP "
                    f"{response.status_code}: {_truncate_text(response.text)}"
                )
            return response.content
        except httpx.RequestError as exc:
            return f"Failed to download from ModelScope: {type(exc).__name__}: {exc}"

    def _http_range_download(self, url: str, max_bytes: int) -> bytes | str:
        """Download the first max_bytes of a file via HTTP range request."""
        headers = {"range": f"bytes=0-{max_bytes - 1}"}
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
                    return (
                        f"Download returned HTTP "
                        f"{response.status_code}: {_truncate_text(body)}"
                    )
                return response.read()
        except httpx.RequestError as exc:
            return f"Failed to download: {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # Shared selection + preview logic
    # ------------------------------------------------------------------

    def _select_and_preview(
        self,
        remote_files: list[_RemoteFile],
        dataset_name: str,
        source: str,
        split: str | None,
        n: int,
    ) -> PreviewRemoteDatasetObservation:
        data_files = [
            f
            for f in remote_files
            if PurePosixPath(f.path).suffix.lower() in _PREVIEW_SUFFIXES
        ]
        all_file_paths = [f.path for f in data_files]

        if not data_files:
            all_paths = [f.path for f in remote_files]
            return PreviewRemoteDatasetObservation.from_text(
                text=(
                    f"Dataset '{dataset_name}' contains no previewable data files.\n"
                    f"All files:\n" + "\n".join(f"  - {p}" for p in all_paths)
                ),
                dataset_name=dataset_name,
                source=source,
                files=all_paths,
            )

        splits = _detect_splits(all_file_paths)

        preview_file = _select_file(data_files, split)
        if preview_file is None:
            file_list_text = "\n".join(
                f"  - {f.path} ({_human_size(f.size)})" for f in data_files
            )
            return PreviewRemoteDatasetObservation.from_text(
                text=(
                    f"Dataset '{dataset_name}' has {len(data_files)} data files.\n"
                    f"Available files:\n{file_list_text}\n\n"
                    f"Specify a 'split' or use a different file path."
                ),
                dataset_name=dataset_name,
                source=source,
                files=all_file_paths,
                splits=splits,
            )

        max_bytes = None
        is_small = (
            preview_file.size is not None and preview_file.size <= _SMALL_FILE_BYTES
        )
        suffix = PurePosixPath(preview_file.path).suffix.lower()
        if not is_small:
            if suffix == ".parquet":
                if (
                    preview_file.size is not None
                    and preview_file.size > _PARQUET_MAX_DOWNLOAD_BYTES
                ):
                    return PreviewRemoteDatasetObservation.from_text(
                        text=(
                            f"Parquet file '{preview_file.path}' is "
                            f"{_human_size(preview_file.size)}, too large to "
                            f"preview (max "
                            f"{_human_size(_PARQUET_MAX_DOWNLOAD_BYTES)}). "
                            "Try specifying a different file or split."
                        ),
                        dataset_name=dataset_name,
                        source=source,
                        files=all_file_paths,
                        splits=splits,
                    )
            else:
                max_bytes = _LARGE_FILE_PREVIEW_BYTES

        download_fn = (
            self._hf_download_file
            if source == "huggingface"
            else self._ms_download_file
        )
        content = download_fn(dataset_name, preview_file.path, max_bytes)
        if isinstance(content, str):
            return PreviewRemoteDatasetObservation.from_text(
                text=content,
                is_error=True,
                dataset_name=dataset_name,
                source=source,
                files=all_file_paths,
                splits=splits,
            )

        parsed = _parse_content(preview_file.path, content, n, truncated=not is_small)

        file_list_text = "\n".join(
            f"  - {f.path} ({_human_size(f.size)})" for f in data_files
        )
        if len(data_files) > 1:
            header = (
                f"Dataset '{dataset_name}' has {len(data_files)} data files. "
                f"Auto-selected '{preview_file.path}' for preview.\n"
                f"Available files:\n{file_list_text}\n\n"
            )
        else:
            header = ""

        text = _format_preview_text(
            dataset_name=dataset_name,
            source=source,
            preview_file=preview_file.path,
            all_files=all_file_paths,
            splits=splits,
            parsed=parsed,
            n=n,
            header=header,
        )

        obs = PreviewRemoteDatasetObservation.from_text(
            text=text,
            dataset_name=dataset_name,
            source=source,
            files=all_file_paths,
            splits=splits,
            num_rows=parsed["num_rows"],
            columns=parsed["columns"],
            sample_rows=parsed["sample_rows"],
            preview_file_path=preview_file.path,
            previewed_rows=parsed["previewed_rows"],
            preview_truncated=parsed.get("truncated", False),
            preview_error=parsed.get("preview_error"),
            has_vision=_infer_has_vision(parsed["columns"], parsed["sample_rows"]),
        )
        if len(data_files) > 1 and header:
            new_content = [
                TextContent(text=header + item.text)
                if isinstance(item, TextContent)
                else item
                for item in obs.content
            ]
            obs = obs.model_copy(update={"content": new_content})
        return obs


class PreviewRemoteDatasetTool(
    ToolDefinition[PreviewRemoteDatasetAction, PreviewRemoteDatasetObservation]
):
    """Tool for previewing remote datasets on HuggingFace / ModelScope."""

    @classmethod
    def create(
        cls,
        conv_state: Any = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[ToolDefinition]:
        timeout = float(params.pop("timeout", 60.0))
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"PreviewRemoteDatasetTool got unknown params: {names}")
        return [
            cls(
                description=_PREVIEW_REMOTE_DESCRIPTION,
                action_type=PreviewRemoteDatasetAction,
                observation_type=PreviewRemoteDatasetObservation,
                executor=PreviewRemoteDatasetExecutor(timeout=timeout),
                annotations=ToolAnnotations(
                    title="preview_remote_dataset",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _select_file(
    files: list[_RemoteFile],
    split: str | None,
) -> _RemoteFile | None:
    """Select a file for preview, optionally filtered by split."""
    candidates = files
    if split:
        split_lower = split.lower()
        candidates = [f for f in files if split_lower in f.path.lower()]
        if not candidates:
            return None
    for f in candidates:
        if PurePosixPath(f.path).suffix.lower() in _SUPPORTED_DATA_SUFFIXES:
            return f
    for f in candidates:
        if PurePosixPath(f.path).suffix.lower() in _TEXT_SUFFIXES:
            return f
    return candidates[0] if candidates else None


def _detect_splits(file_paths: list[str]) -> list[str]:
    """Detect split names from file paths.

    Looks for patterns like 'train/xxx.parquet' or 'xxx-train.jsonl'.
    """
    splits: set[str] = set()
    known = {"train", "test", "validation", "val", "dev", "eval"}
    for path in file_paths:
        parts = path.lower().replace("\\", "/").split("/")
        for part in parts:
            stem = PurePosixPath(part).stem
            if stem in known:
                splits.add(stem)
        for kw in known:
            if kw in path.lower():
                splits.add(kw)
    return sorted(splits)


def _parse_content(
    file_path: str,
    content: bytes,
    max_samples: int,
    truncated: bool,
) -> dict[str, Any]:
    suffix = PurePosixPath(file_path).suffix.lower()
    if suffix == ".parquet":
        return _parse_parquet(content, max_samples, truncated)
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if suffix == ".jsonl":
        return _parse_jsonl(lines, max_samples, truncated)
    if suffix == ".csv":
        return _parse_delimited(lines, max_samples, truncated, ",")
    if suffix == ".tsv":
        return _parse_delimited(lines, max_samples, truncated, "\t")
    if suffix == ".json":
        return _parse_json(content, max_samples, truncated)
    return _parse_text(lines, max_samples, truncated)


def _parse_parquet(
    content: bytes,
    max_samples: int,
    truncated: bool,
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(io.BytesIO(content))
    except Exception as exc:
        return {
            "num_rows": None,
            "columns": [],
            "sample_rows": [],
            "previewed_rows": 0,
            "truncated": truncated,
            "preview_error": f"Parquet parse error: {type(exc).__name__}: {exc}",
        }

    columns = table.column_names
    total_rows = table.num_rows
    sample_count = min(max_samples, total_rows)
    sample_rows: list[dict[str, Any]] = []
    table_slice = table.slice(0, sample_count)
    for i in range(table_slice.num_rows):
        row: dict[str, Any] = {"row": i + 1}
        for col in columns:
            value = table_slice.column(col)[i].as_py()
            row[col] = _trim_value(value)
        sample_rows.append(row)

    return {
        "num_rows": total_rows if not truncated else None,
        "columns": columns,
        "sample_rows": sample_rows,
        "previewed_rows": sample_count,
        "truncated": truncated,
        "preview_error": None,
    }


def _parse_jsonl(
    lines: list[str],
    max_samples: int,
    truncated: bool,
) -> dict[str, Any]:
    non_empty = [line for line in lines if line.strip()]
    sample_rows: list[dict[str, Any]] = []
    for idx, line in enumerate(non_empty[:max_samples], start=1):
        row: dict[str, Any] = {
            "line": idx,
            "text": _truncate_text(line, _MAX_SAMPLE_STRING_CHARS),
        }
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                row.update(_trim_dict(parsed))
        except json.JSONDecodeError:
            pass
        sample_rows.append(row)
    return {
        "num_rows": len(non_empty) if not truncated else None,
        "columns": _collect_columns(sample_rows),
        "sample_rows": sample_rows,
        "previewed_rows": len(non_empty),
        "truncated": truncated,
        "preview_error": None,
    }


def _parse_delimited(
    lines: list[str],
    max_samples: int,
    truncated: bool,
    delimiter: str,
) -> dict[str, Any]:
    import csv

    if not lines or delimiter not in lines[0]:
        return _parse_text(lines, max_samples, truncated)
    header = lines[0]
    data_lines = lines[1 : max_samples + 1]
    reader = csv.DictReader([header, *data_lines], delimiter=delimiter)
    columns = [str(name) for name in reader.fieldnames or []]
    sample_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(reader, start=1):
        entry: dict[str, Any] = {"line": idx}
        for key, value in row.items():
            if key is not None and value is not None:
                entry[key] = _truncate_text(str(value), _MAX_SAMPLE_STRING_CHARS)
        sample_rows.append(entry)
    return {
        "num_rows": max(len(lines) - 1, 0) if not truncated else None,
        "columns": columns,
        "sample_rows": sample_rows,
        "previewed_rows": max(len(lines) - 1, 0),
        "truncated": truncated,
        "preview_error": None,
    }


def _parse_json(
    content: bytes,
    max_samples: int,
    truncated: bool,
) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            "num_rows": None,
            "columns": [],
            "sample_rows": [],
            "previewed_rows": 0,
            "truncated": truncated,
            "preview_error": f"JSON parse error: {exc}",
        }
    if isinstance(parsed, list):
        rows = parsed[:max_samples]
        sample_rows: list[dict[str, Any]] = []
        for idx, r in enumerate(rows, start=1):
            if isinstance(r, dict):
                sample_rows.append({"index": idx, **_trim_dict(r)})
            else:
                sample_rows.append({"index": idx, "value": _trim_value(r)})
        return {
            "num_rows": len(parsed) if not truncated else None,
            "columns": _collect_columns(sample_rows),
            "sample_rows": sample_rows,
            "previewed_rows": len(parsed),
            "truncated": truncated,
            "preview_error": None,
        }
    if isinstance(parsed, dict):
        return {
            "num_rows": 1,
            "columns": list(parsed.keys()),
            "sample_rows": [{"index": 1, **_trim_dict(parsed)}],
            "previewed_rows": 1,
            "truncated": truncated,
            "preview_error": None,
        }
    return {
        "num_rows": None,
        "columns": [],
        "sample_rows": [],
        "previewed_rows": 0,
        "truncated": truncated,
        "preview_error": "Unsupported JSON structure.",
    }


def _parse_text(
    lines: list[str],
    max_samples: int,
    truncated: bool,
) -> dict[str, Any]:
    sample_rows = [
        {"line": idx, "text": _truncate_text(text, _MAX_SAMPLE_STRING_CHARS)}
        for idx, text in enumerate(lines[:max_samples], start=1)
    ]
    return {
        "num_rows": len(lines) if not truncated else None,
        "columns": [],
        "sample_rows": sample_rows,
        "previewed_rows": len(lines),
        "truncated": truncated,
        "preview_error": None,
    }


def _trim_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, _MAX_SAMPLE_STRING_CHARS)
    if isinstance(value, list):
        return [_trim_value(item) for item in value]
    if isinstance(value, dict):
        return _trim_dict(value)
    return value


def _trim_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _trim_value(item) for key, item in value.items()}


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
    vision_markers = ("image", "video", "vision", "img")
    for column in columns:
        if any(marker in column.lower() for marker in vision_markers):
            return True
    return False


def _truncate_text(text: str, limit: int = 1000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} characters truncated]"


def _human_size(size: int | None) -> str:
    if size is None:
        return "size unknown"
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f}MB"
    return f"{size / (1024 * 1024 * 1024):.2f}GB"


def _format_preview_text(
    *,
    dataset_name: str,
    source: str,
    preview_file: str,
    all_files: list[str],
    splits: list[str],
    parsed: dict[str, Any],
    n: int,
    header: str,
) -> str:
    parts = [
        f"{source} dataset preview: {dataset_name}",
        f"preview_file={preview_file}",
        f"files={len(all_files)}",
    ]
    if splits:
        parts.append(f"splits={', '.join(splits)}")
    if parsed["num_rows"] is not None:
        parts.append(f"rows={parsed['num_rows']}")
    else:
        parts.append(f"previewed_rows={parsed['previewed_rows']}")
    if parsed["columns"]:
        parts.append(f"columns={', '.join(parsed['columns'])}")
    parts.append(f"requested_rows={n}")
    parts.append(f"sample_rows={len(parsed['sample_rows'])}")
    if parsed.get("preview_error"):
        parts.append(f"preview_error={parsed['preview_error']}")
    parts.append(f"preview_truncated={str(parsed.get('truncated', False)).lower()}")
    for row in parsed["sample_rows"]:
        row_id = row.get("line", row.get("row", row.get("index", "?")))
        parts.append(f"\n--- sample row {row_id} ---")
        row_text = {
            k: v for k, v in row.items() if k not in ("line", "row", "index", "text")
        }
        if row.get("text"):
            parts.append(str(row["text"]))
        elif row_text:
            parts.append(json.dumps(row_text, ensure_ascii=False, default=str))
    return header + "\n".join(parts)


register_tool(PreviewRemoteDatasetTool.name, PreviewRemoteDatasetTool)
