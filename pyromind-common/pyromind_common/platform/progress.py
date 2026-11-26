"""Read business progress directly from a mounted Storage run directory."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pyromind_common.platform.paths import (
    from_storage_api_path,
    to_storage_api_path,
)


class _ProgressCounters(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, strict=True)

    total: int | None = Field(default=None, ge=0)
    processed: int | None = Field(default=None, ge=0)
    succeeded: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)
    elapsed_ms: int | None = Field(default=None, ge=0)
    records_per_second: float | None = Field(default=None, ge=0)
    eta_ms: int | None = Field(default=None, ge=0)
    updated_at: str | None = None


class ProgressSnapshot(_ProgressCounters):
    output_dir: str
    progress_found: bool = False
    percent: float | None = None
    latest_records: list[dict[str, Any]] = Field(default_factory=list)
    tail_truncated: bool = False
    report_found: bool = False
    report: dict[str, Any] | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    evaluation_report_url: str | None = None
    errors: dict[str, str] = Field(default_factory=dict)


class _ReadOptions(BaseModel):
    tail_lines: int = Field(default=5, ge=0, le=50)
    tail_bytes: int = Field(default=65536, ge=1, le=4 * 1024 * 1024)
    max_json_bytes: int = Field(default=1024 * 1024, ge=1, le=16 * 1024 * 1024)


def _file(root: Path, name: str) -> Path:
    target = (root / name).resolve(strict=True)
    target.relative_to(root)
    if not target.is_file():
        raise ValueError("Not a regular run file")
    return target


def _json_file(
    root: Path, name: str, limit: int, errors: dict[str, str]
) -> dict[str, Any] | None:
    try:
        with _file(root, name).open("rb") as source:
            payload = source.read(limit + 1)
        if len(payload) > limit:
            errors[name] = "JSON file exceeds the configured byte limit"
            return None
        document = json.loads(payload.decode("utf-8-sig"))
        if not isinstance(document, dict):
            raise ValueError("Not a JSON object")
        return document
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, ValueError):
        errors[name] = "Run file is unreadable or is not a UTF-8 JSON object"
        return None


def _tail(
    root: Path, name: str, options: _ReadOptions, errors: dict[str, str]
) -> tuple[list[dict[str, Any]], bool]:
    if options.tail_lines == 0:
        return [], False
    try:
        with _file(root, name).open("rb") as source:
            source.seek(0, 2)
            end = source.tell()
            start = max(0, end - options.tail_bytes)
            if start:
                source.seek(start - 1)
                boundary = source.read(1)
            else:
                source.seek(0)
                boundary = b"\n"
            payload = source.read(min(end - start, options.tail_bytes))
    except FileNotFoundError:
        return [], False
    except (OSError, ValueError):
        errors[name] = "Cannot read the output tail inside the run directory"
        return [], False
    truncated = bool(start)
    if start and boundary != b"\n":
        first_newline = payload.find(b"\n")
        payload = payload[first_newline + 1 :] if first_newline >= 0 else b""
    records = []
    for line in reversed(payload.splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode("utf-8"))
            if not isinstance(record, dict):
                raise ValueError("Not a JSON object")
        except (UnicodeDecodeError, ValueError):
            errors[name] = "Skipped incomplete or invalid JSONL records"
            truncated = True
            continue
        if len(records) == options.tail_lines:
            truncated = True
            break
        records.append(record)
    return list(reversed(records)), truncated


def _artifact_paths(
    report: dict[str, Any] | None,
    root: Path,
    api_output: str,
    errors: dict[str, str],
) -> dict[str, str]:
    artifacts = report.get("artifacts") if report is not None else None
    if not isinstance(artifacts, dict):
        return {}
    result = {}
    output = PurePosixPath(api_output)
    for name, value in artifacts.items():
        try:
            if not isinstance(value, str):
                raise ValueError("Not a path")
            api_path = to_storage_api_path(
                value if value.startswith("/") else str(output / value)
            )
            relative = PurePosixPath(api_path).relative_to(output)
            _file(root, str(relative))
        except (OSError, ValueError):
            errors["artifacts"] = (
                "Some artifacts are missing or outside the run directory"
            )
            continue
        result[name] = from_storage_api_path(api_path)
    return result


def _confirmed_report_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(ord(character) < 33 for character in value)
    ):
        raise ValueError(
            "Supply an already confirmed HTTP report URL without credentials"
        )
    return value


def read_progress(
    output_dir: str,
    *,
    storage_root: Path = Path("/workspace"),
    tail_filename: str = "processed.jsonl",
    tail_lines: int = 5,
    tail_bytes: int = 65536,
    max_json_bytes: int = 1024 * 1024,
    stable_report_url: str | None = None,
) -> ProgressSnapshot:
    """Read one snapshot; neither missing files nor counters imply task completion.

    A stable report URL must already be confirmed by the calling product flow.
    Mounted files alone do not prove that a report has a registered Runtime run.
    """
    options = _ReadOptions(
        tail_lines=tail_lines, tail_bytes=tail_bytes, max_json_bytes=max_json_bytes
    )
    if (
        not tail_filename
        or PurePosixPath(tail_filename).name != tail_filename
        or "\\" in tail_filename
        or any(ord(character) < 32 for character in tail_filename)
        or tail_filename in {".", ".."}
    ):
        raise ValueError("tail_filename must be a filename inside the run directory")
    api_output = to_storage_api_path(output_dir)
    mount = storage_root.resolve(strict=True)
    root = (mount / api_output.lstrip("/")).resolve()
    root.relative_to(mount)
    result = ProgressSnapshot(
        output_dir=from_storage_api_path(api_output),
        evaluation_report_url=_confirmed_report_url(stable_report_url),
    )
    progress = _json_file(root, "progress.json", options.max_json_bytes, result.errors)
    if progress is not None:
        try:
            counters = _ProgressCounters.model_validate(progress)
            result = result.model_copy(
                update={**counters.model_dump(), "progress_found": True}
            )
        except ValidationError:
            result.errors["progress.json"] = "Progress counters are invalid"
    if result.total and result.processed is not None:
        result.percent = min(100.0, result.processed / result.total * 100)
    result.report = _json_file(
        root, "report.json", options.max_json_bytes, result.errors
    )
    result.report_found = result.report is not None
    result.latest_records, result.tail_truncated = _tail(
        root, tail_filename, options, result.errors
    )
    result.artifact_paths = _artifact_paths(
        result.report, root, api_output, result.errors
    )
    return result
