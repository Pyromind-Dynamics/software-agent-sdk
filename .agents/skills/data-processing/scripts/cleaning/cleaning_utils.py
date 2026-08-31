"""Small dependency-free helpers for dataset cleaning scripts.

The functions in this module are intentionally conservative: they prefer
streaming records, explicit validation errors, and simple stdlib-only repairs
over opaque dependencies. Cleaning scripts should import them with:

    PYTHONPATH=<skill>/scripts python clean_script.py ...
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ERROR_MISSING_FIELD = "missing_field"
ERROR_TYPE = "type_error"
ERROR_EMPTY_STRING = "empty_string"
ERROR_JSON_DECODE = "json_decode_error"
ERROR_INVALID_ROLE = "invalid_role"
ERROR_INVALID_FORMAT = "invalid_format"
ERROR_TRUNCATED_FIELD = "truncated_field"
ERROR_BENCHMARK_EXCLUSION = "benchmark_exclusion"
ERROR_EMPTY_DATASET = "empty_dataset"

ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "prompter": "user",
    "instruction": "user",
    "question": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "answer": "assistant",
    "response": "assistant",
    "reasoning": "assistant",
    "tool_call": "assistant",
    "system": "system",
    "developer": "system",
    "tool": "tool",
    "tool_result": "tool",
    "tool_output": "tool",
    "function": "tool",
    "observation": "tool",
    "environment": "tool",
    "executor": "tool",
}
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)")
WRAPPER_TAG_RE = re.compile(
    r"^\s*<(?:tool_call|tool_response|answer)>\s*|"
    r"\s*</(?:tool_call|tool_response|answer)>\s*$"
)
TRAINING_EXCLUSION_RE = re.compile(
    r"(?:should\s+never\s+appear\s+in\s+training\s+corpora|"
    r"do\s+not\s+(?:use\s+(?:this\s+)?(?:data|dataset)\s+to\s+)?train)",
    re.IGNORECASE,
)
SENSITIVE_KEY_RE = re.compile(
    r"(?:authorization|cookie|password|passwd|secret|token|api[_-]?key)",
    re.IGNORECASE,
)
SENSITIVE_INLINE_RE = re.compile(
    r"(?i)((?:authorization|cookie|password|passwd|secret|token|api[_-]?key)"
    r"\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^,\s}]+)"
)
BEARER_TOKEN_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")

__all__ = [
    "ALLOWED_MESSAGE_ROLES",
    "CleaningStats",
    "DEFAULT_SYSTEM_PROMPT",
    "DataCleaningError",
    "DropRecord",
    "ExactDeduper",
    "ParsedRecord",
    "ValidationError",
    "answer_from_choices",
    "content_to_text",
    "detect_output_format",
    "detect_training_format",
    "ensure_system_message",
    "first_validation_reason",
    "get_path",
    "is_too_long",
    "iter_csv",
    "iter_huggingface_rows",
    "iter_input_files",
    "iter_json",
    "iter_jsonl",
    "iter_records",
    "iter_text",
    "messages_from_record",
    "messages_from_trace_events",
    "messages_from_transcript",
    "missing_required_fields",
    "multiple_choice_prompt",
    "normalize_content",
    "normalize_messages",
    "normalize_role",
    "normalize_text",
    "normalize_tool_call",
    "normalize_tool_calls",
    "parse_json_loose",
    "redact_sensitive_data",
    "run_cleaning",
    "stable_json_dumps",
    "to_messages_record",
    "to_preference_record",
    "to_training_record",
    "training_exclusion_reason",
    "validate_messages_record",
    "validate_preference_record",
    "validate_record",
    "write_jsonl",
]


class DataCleaningError(ValueError):
    """Raised when a row cannot be parsed or normalized."""

    def __init__(
        self,
        message: str,
        *,
        code: str = ERROR_INVALID_FORMAT,
        detail: str | None = None,
    ) -> None:
        """Create a cleaning error with a machine-readable code."""
        super().__init__(message)
        self.code = code
        self.detail = detail


@dataclass
class ParsedRecord:
    """A record yielded by a tolerant reader."""

    data: Any | None
    line_number: int | None = None
    raw: str | None = None
    error: str | None = None
    error_type: str | None = None
    source_path: str | None = None

    @property
    def ok(self) -> bool:
        """Return whether the record parsed successfully."""
        return self.error is None


@dataclass
class ValidationError:
    """A schema validation error with a stable category."""

    code: str
    message: str
    path: str = "$"
    line_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the validation error to JSON-serializable data."""
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }
        if self.line_number is not None:
            data["line_number"] = self.line_number
        return data


@dataclass
class CleaningStats:
    """Collect stable counters written to the cleaning report."""

    read: int = 0
    written: int = 0
    dropped: int = 0
    errors: int = 0
    duplicates: int = 0
    drop_reasons: Counter[str] = field(default_factory=Counter)
    error_samples: list[dict[str, Any]] = field(default_factory=list)
    max_error_samples: int = 20
    status: str = "running"
    failure: str | None = None

    @property
    def total(self) -> int:
        """Backward-compatible alias for the number of source records read."""
        return self.read

    @property
    def kept(self) -> int:
        """Backward-compatible alias for the number of output rows written."""
        return self.written

    def record_input(self) -> None:
        """Increment the number of rows observed by a cleaning script."""
        self.read += 1

    def record_keep(self) -> None:
        """Increment the number of rows retained by a cleaning script."""
        self.written += 1

    def record_drop(
        self,
        reason: str,
        *,
        sample: Any | None = None,
        line_number: int | None = None,
        source_index: int | None = None,
        source_path: str | None = None,
        error: str | None = None,
        is_error: bool = False,
        is_duplicate: bool = False,
    ) -> None:
        """Increment a drop reason and keep a bounded sample for debugging."""
        self.dropped += 1
        self.drop_reasons[reason] += 1
        if is_error:
            self.errors += 1
        if is_duplicate:
            self.duplicates += 1
        if len(self.error_samples) >= self.max_error_samples:
            return
        item: dict[str, Any] = {"reason": reason}
        if source_path is not None:
            item["source_path"] = source_path
        if source_index is not None:
            item["source_index"] = source_index
        if line_number is not None:
            item["line_number"] = line_number
        if error:
            item["error"] = redact_sensitive_data(error)
        if sample is not None:
            item["sample"] = truncate_for_stats(sample)
        self.error_samples.append(item)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable stats payload."""
        return {
            "read": self.read,
            "written": self.written,
            "dropped": self.dropped,
            "errors": self.errors,
            "duplicates": self.duplicates,
            "drop_reasons": dict(self.drop_reasons),
            "error_samples": self.error_samples,
            "status": self.status,
            **({"failure": self.failure} if self.failure else {}),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CleaningStats:
        """Restore cumulative counters from a checkpoint payload."""
        return cls(
            read=int(value.get("read", value.get("total", 0))),
            written=int(value.get("written", value.get("kept", 0))),
            dropped=int(value.get("dropped", 0)),
            errors=int(value.get("errors", 0)),
            duplicates=int(value.get("duplicates", 0)),
            drop_reasons=Counter(value.get("drop_reasons", {})),
            error_samples=list(value.get("error_samples", [])),
            status=str(value.get("status", "running")),
            failure=value.get("failure"),
        )

    def write_json(self, path: str | Path) -> None:
        """Write the stats payload to a JSON file."""
        atomic_write_json(path, self.to_dict())


class DropRecord(DataCleaningError):
    """Signal an intentional row drop without failing the cleaner."""

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(detail or reason, code=reason, detail=detail)


class ExactDeduper:
    """Exact SHA-256 deduper for whole rows or selected field paths."""

    def __init__(self, field_path: str | None = None) -> None:
        """Create a deduper for a dotted field path or the full record."""
        self.field_path = field_path
        self._seen: set[str] = set()

    def hash_value(self, value: Any) -> str:
        """Return a stable hash for a JSON-compatible value."""
        text = stable_json_dumps(value)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def key_for_record(self, record: dict[str, Any]) -> str:
        """Return the dedupe key for a record."""
        value = get_path(record, self.field_path) if self.field_path else record
        return self.hash_value(value)

    def is_duplicate(self, record: dict[str, Any]) -> bool:
        """Return True after seeing the same key before."""
        key = self.key_for_record(record)
        if key in self._seen:
            return True
        self._seen.add(key)
        return False


def decode_text(blob: bytes) -> str:
    """Decode bytes with UTF-8 first and latin-1 fallback."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return blob.decode(encoding)
        except UnicodeDecodeError:
            continue
    return blob.decode("utf-8", errors="replace")


def normalize_text(value: Any, *, collapse_spaces: bool = False) -> str:
    """Normalize text, removing unsafe controls while preserving newlines."""
    text = "" if value is None else str(value)
    # NFC repairs canonically equivalent sequences without compatibility-folding
    # meaningful symbols such as 10⁴ into 104.
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    if collapse_spaces:
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return text.strip()


def strip_code_fence(text: str) -> str:
    """Remove a surrounding Markdown code fence when present."""
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return cleaned


def repair_json_text(text: str) -> str:
    """Apply small deterministic JSON repairs before parsing."""
    cleaned = strip_code_fence(text)
    cleaned = cleaned.lstrip("\ufeff").strip()
    cleaned = TRAILING_COMMA_RE.sub(r"\1", cleaned)
    cleaned = UNQUOTED_KEY_RE.sub(r'\1"\2"\3', cleaned)
    return cleaned


def parse_json_loose(text: str) -> Any:
    """Parse JSON with small repairs and Python-literal fallback."""
    cleaned = strip_code_fence(text).lstrip("\ufeff").strip()
    if not cleaned:
        raise DataCleaningError("empty JSON string", code=ERROR_EMPTY_STRING)
    attempts = [cleaned, repair_json_text(cleaned)]
    last_error: Exception | None = None
    for candidate in attempts:
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            last_error = exc
    try:
        value = ast.literal_eval(repair_json_text(cleaned))
    except (SyntaxError, ValueError) as exc:
        detail = str(last_error or exc)
        raise DataCleaningError(
            "could not parse JSON",
            code=ERROR_JSON_DECODE,
            detail=detail,
        ) from exc
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    raise DataCleaningError(
        "unsupported Python literal",
        code=ERROR_TYPE,
        detail=type(value).__name__,
    )


def iter_jsonl(
    path: str | Path,
    *,
    unwrap_huggingface: bool = True,
) -> Iterator[ParsedRecord]:
    """Yield parsed JSONL records without stopping on bad lines."""
    with Path(path).open("rb") as handle:
        for line_number, blob in enumerate(handle, start=1):
            raw = decode_text(blob).strip()
            if not raw:
                continue
            try:
                parsed = parse_json_loose(raw)
                if unwrap_huggingface and _is_huggingface_rows_envelope(parsed):
                    yield from _iter_huggingface_envelope(parsed)
                else:
                    yield ParsedRecord(parsed, line_number, raw)
            except DataCleaningError as exc:
                yield ParsedRecord(
                    None,
                    line_number,
                    raw,
                    str(exc),
                    exc.code,
                )


def _iter_json_array(path: Path) -> Iterator[ParsedRecord]:
    """Incrementally decode a top-level JSON array without loading it all."""
    decoder = json.JSONDecoder(strict=False)
    buffer = ""
    position = 0
    item_number = 0
    eof = False
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        while True:
            if position >= len(buffer) and not eof:
                buffer = handle.read(65536)
                position = 0
                eof = buffer == ""
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                if eof:
                    return
                continue
            if buffer[position] != "[":
                raise DataCleaningError(
                    "JSON input is not a top-level array",
                    code=ERROR_INVALID_FORMAT,
                )
            position += 1
            break

        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise DataCleaningError(
                        "unterminated JSON array",
                        code=ERROR_JSON_DECODE,
                    )
                buffer = buffer[position:] + handle.read(65536)
                position = 0
                eof = handle.tell() == path.stat().st_size
            if buffer[position] == "]":
                return

            while True:
                try:
                    item, end = decoder.raw_decode(buffer, position)
                    position = end
                    item_number += 1
                    yield ParsedRecord(item, item_number)
                    if position > 65536:
                        buffer = buffer[position:]
                        position = 0
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        yield ParsedRecord(
                            None,
                            item_number + 1,
                            buffer[position:],
                            f"could not parse JSON array item: {exc.msg}",
                            ERROR_JSON_DECODE,
                        )
                        return
                    buffer = buffer[position:] + handle.read(65536)
                    position = 0
                    eof = handle.tell() == path.stat().st_size


def iter_json(path: str | Path) -> Iterator[ParsedRecord]:
    """Yield records from a JSON array, object, or JSONL-looking file."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = ""
        while character := handle.read(1):
            if not character.isspace():
                first = character
                break
    if not first:
        return
    if first == "[":
        yield from _iter_json_array(source)
        return
    if first != "{":
        yield from iter_jsonl(source)
        return
    # A top-level object is one source record. If parsing fails, retry as JSONL
    # so newline-delimited objects with a .json suffix remain usable.
    text = decode_text(source.read_bytes()).strip()
    try:
        parsed = parse_json_loose(text)
    except DataCleaningError:
        yield from iter_jsonl(source)
        return
    if _is_huggingface_rows_envelope(parsed):
        yield from _iter_huggingface_envelope(parsed)
    else:
        yield ParsedRecord(parsed, 1)


def iter_delimited(
    path: str | Path,
    *,
    delimiter: str | None = None,
) -> Iterator[ParsedRecord]:
    """Yield CSV or TSV rows as dictionaries."""
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        sample = handle.read(8192)
        handle.seek(0)
        if delimiter is None:
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
        else:
            reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return
        for line_number, row in enumerate(reader, start=2):
            cleaned: dict[str, Any] = {}
            for key, value in row.items():
                target_key = "_extra" if key is None else normalize_text(key)
                cleaned[target_key] = value
            yield ParsedRecord(cleaned, line_number)


def iter_csv(path: str | Path) -> Iterator[ParsedRecord]:
    """Yield CSV rows as dictionaries with tolerant dialect detection."""
    yield from iter_delimited(path)


def iter_text(path: str | Path) -> Iterator[ParsedRecord]:
    """Yield each non-empty text line as ``{"text": line}``."""
    with Path(path).open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = normalize_text(line)
            if text:
                yield ParsedRecord({"text": text}, line_number, line.rstrip("\n"))


def _iter_json_like_text(path: Path) -> Iterator[ParsedRecord]:
    """Prefer JSON records, but preserve ordinary brace-prefixed text."""
    buffered_errors: list[ParsedRecord] = []
    records = iter(iter_json(path))
    for record in records:
        if not record.ok:
            buffered_errors.append(record)
            continue
        yield from buffered_errors
        yield record
        yield from records
        return
    yield from iter_text(path)


def _first_non_whitespace(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        while character := handle.read(1):
            if not character.isspace():
                return character
    return ""


_TEXT_INPUT_SUFFIXES = {".log", ".md", ".text", ".txt"}
_SUPPORTED_INPUT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".tsv",
    *_TEXT_INPUT_SUFFIXES,
}


def iter_input_files(path: str | Path) -> Iterator[Path]:
    """Yield one file or a directory tree in deterministic path order."""
    source = Path(path)
    if source.is_file():
        yield source
        return
    if not source.is_dir():
        raise FileNotFoundError(f"input path does not exist: {source}")
    files = sorted(
        candidate
        for candidate in source.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_INPUT_SUFFIXES
    )
    if not files:
        raise DataCleaningError(
            f"input directory contains no supported files: {source}",
            code=ERROR_EMPTY_DATASET,
        )
    yield from files


def iter_records(path: str | Path) -> Iterator[ParsedRecord]:
    """Stream one file or a directory using stable file order."""
    for source in iter_input_files(path):
        suffix = source.suffix.lower()
        if suffix == ".csv":
            records = iter_csv(source)
        elif suffix == ".tsv":
            records = iter_delimited(source, delimiter="\t")
        elif suffix == ".json":
            records = iter_json(source)
        elif suffix in _TEXT_INPUT_SUFFIXES:
            records = (
                _iter_json_like_text(source)
                if _first_non_whitespace(source) in {"{", "["}
                else iter_text(source)
            )
        else:
            records = iter_jsonl(source)
        for record in records:
            record.source_path = str(source)
            yield record


def _is_huggingface_rows_envelope(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("rows"), list)


def _iter_huggingface_envelope(
    envelope: dict[str, Any],
    *,
    reject_truncated: bool = True,
) -> Iterator[ParsedRecord]:
    for position, item in enumerate(envelope["rows"], start=1):
        if not isinstance(item, dict) or "row" not in item:
            yield ParsedRecord(
                None,
                position,
                error="invalid Hugging Face row wrapper",
                error_type=ERROR_INVALID_FORMAT,
            )
            continue
        row_number = item.get("row_idx")
        line_number = row_number + 1 if isinstance(row_number, int) else position
        truncated = item.get("truncated_cells")
        if reject_truncated and isinstance(truncated, list) and truncated:
            fields = ", ".join(normalize_text(field) for field in truncated)
            yield ParsedRecord(
                None,
                line_number,
                stable_json_dumps(item.get("row")),
                f"Hugging Face preview truncated fields: {fields}",
                ERROR_TRUNCATED_FIELD,
            )
            continue
        yield ParsedRecord(item["row"], line_number)


def iter_huggingface_rows(
    path: str | Path,
    *,
    reject_truncated: bool = True,
) -> Iterator[ParsedRecord]:
    """Yield row payloads from a Hugging Face rows/first-rows API response."""
    source = Path(path)
    text = decode_text(source.read_bytes()).strip()
    try:
        envelope = parse_json_loose(text)
    except DataCleaningError as exc:
        yield ParsedRecord(None, 1, text, str(exc), exc.code)
        return
    if not _is_huggingface_rows_envelope(envelope):
        yield ParsedRecord(
            None,
            1,
            error="missing Hugging Face rows envelope",
            error_type=ERROR_INVALID_FORMAT,
        )
        return
    yield from _iter_huggingface_envelope(
        envelope,
        reject_truncated=reject_truncated,
    )


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as UTF-8 JSONL and return the number written."""
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_json_dumps(value: Any) -> str:
    """Serialize a value deterministically for hashing or comparison."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Durably replace a JSON file without exposing a partial write."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def truncate_for_stats(value: Any, *, max_chars: int = 500) -> Any:
    """Return a compact sample suitable for the cleaning report."""
    redacted = redact_sensitive_data(value)
    text = redacted if isinstance(redacted, str) else stable_json_dumps(redacted)
    if len(text) <= max_chars:
        return redacted
    return text[: max_chars - 3] + "..."


def redact_sensitive_data(value: Any) -> Any:
    """Redact common credentials from bounded diagnostic samples."""
    if isinstance(value, dict):
        return {
            key: (
                "***REDACTED***"
                if SENSITIVE_KEY_RE.search(str(key))
                else redact_sensitive_data(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, str):
        redacted = SENSITIVE_INLINE_RE.sub(r"\1***REDACTED***", value)
        return BEARER_TOKEN_RE.sub("Bearer ***REDACTED***", redacted)
    return value


def get_path(record: dict[str, Any], path: str | None, default: Any = None) -> Any:
    """Read a dotted field path from a nested dictionary."""
    if not path:
        return record
    current: Any = record
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return default
    return current


def missing_required_fields(
    record: dict[str, Any],
    fields: Iterable[str],
) -> list[str]:
    """Return required dotted field paths that are absent or empty."""
    missing: list[str] = []
    for field_path in fields:
        value = get_path(record, field_path)
        if (
            value is None
            or (isinstance(value, str) and normalize_text(value) == "")
            or (isinstance(value, (list, dict)) and not value)
        ):
            missing.append(field_path)
    return missing


def training_exclusion_reason(record: dict[str, Any]) -> str | None:
    """Return a reason when explicit metadata excludes a row from training.

    Only known metadata fields are inspected so user text that merely discusses
    training-data policies is not accidentally filtered.
    """
    for key in ("canary", "benchmark_canary", "training_exclusion"):
        value = record.get(key)
        if value is not None and TRAINING_EXCLUSION_RE.search(normalize_text(value)):
            return f"{key} explicitly excludes this row from training"
    if record.get("do_not_train") is True:
        return "do_not_train is true"
    return None


def is_too_long(
    record: dict[str, Any],
    *,
    max_chars: int,
    fields: Iterable[str] | None = None,
) -> bool:
    """Return whether selected fields or the full row exceed max_chars."""
    if fields is None:
        text = stable_json_dumps(record)
        return len(text) > max_chars
    for path in fields:
        value = get_path(record, path, "")
        text = value if isinstance(value, str) else stable_json_dumps(value)
        if len(text) > max_chars:
            return True
    return False


def normalize_role(value: Any, *, fallback: str | None = None) -> str:
    """Map common dataset role labels to system/user/assistant/tool."""
    role = normalize_text(value, collapse_spaces=True).lower()
    if role in ROLE_ALIASES:
        return ROLE_ALIASES[role]
    if fallback is not None:
        return fallback
    return role


def normalize_content(value: Any) -> str | list[dict[str, Any]]:
    """Normalize message content while preserving multimodal content parts."""
    if isinstance(value, dict) and "type" in value:
        return normalize_content([value])
    if isinstance(value, list):
        parts: list[dict[str, Any]] = []
        for part in value:
            if isinstance(part, str):
                parts.append({"type": "text", "text": normalize_text(part)})
                continue
            if not isinstance(part, dict):
                parts.append({"type": "text", "text": normalize_text(part)})
                continue
            part_type = normalize_text(
                part.get("type", "text"), collapse_spaces=True
            ).lower()
            if part_type == "image_url":
                part_type = "image"
            normalized = {"type": part_type or "text"}
            if "text" in part:
                normalized["text"] = normalize_text(part["text"])
            if "content" in part and "text" not in normalized:
                normalized["text"] = normalize_text(part["content"])
            for key in ("url", "path", "image", "video", "audio"):
                if key in part:
                    normalized[key] = normalize_text(part[key])
            image_url = part.get("image_url")
            if part_type == "image" and "url" not in normalized:
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                if image_url:
                    normalized["url"] = normalize_text(image_url)
            parts.append(normalized)
        return parts
    if isinstance(value, str) or value is None:
        return normalize_text(value)
    return normalize_text(stable_json_dumps(value))


def content_to_text(value: Any) -> str:
    """Flatten normalized string or content parts to plain text."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return normalize_text(value)
    fragments: list[str] = []
    for part in value:
        if isinstance(part, dict):
            fragments.append(normalize_text(part.get("text", "")))
        else:
            fragments.append(normalize_text(part))
    return "\n".join(fragment for fragment in fragments if fragment)


def normalize_tool_call(value: Any) -> dict[str, Any] | None:
    """Normalize one tool call into OpenAI-style function-call shape."""
    if isinstance(value, str):
        try:
            value = parse_json_loose(value)
        except DataCleaningError:
            return None
    if not isinstance(value, dict):
        return None
    function = value.get("function")
    if isinstance(function, str):
        try:
            function = parse_json_loose(function)
        except DataCleaningError:
            function = {"name": function, "arguments": {}}
    if not isinstance(function, dict):
        function = {
            "name": value.get("name") or value.get("tool_name") or value.get("tool"),
            "arguments": value.get("arguments") or value.get("args") or {},
        }
    name = normalize_text(function.get("name", ""), collapse_spaces=True)
    if not name:
        return None
    arguments = function.get("arguments", {})
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = parse_json_loose(arguments)
        except DataCleaningError:
            arguments = {"raw": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    call: dict[str, Any] = {
        "type": normalize_text(value.get("type", "function")) or "function",
        "function": {"name": name, "arguments": arguments},
    }
    call_id = value.get("id") or value.get("tool_call_id")
    if call_id is not None:
        call["id"] = normalize_text(call_id, collapse_spaces=True)
    if isinstance(value.get("index"), int):
        call["index"] = value["index"]
    return call


def normalize_tool_calls(value: Any) -> list[dict[str, Any]]:
    """Normalize a tool_calls field into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = parse_json_loose(value)
        except DataCleaningError:
            return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in value:
        call = normalize_tool_call(item)
        if call is not None:
            calls.append(call)
    return calls


def strip_message_wrapper(value: Any) -> str:
    """Strip one outer tool_call/tool_response/answer XML-style wrapper."""
    return WRAPPER_TAG_RE.sub("", normalize_text(value)).strip()


def _raw_message_role(item: Any) -> str:
    """Return a normalized source-role label without alias conversion."""
    if not isinstance(item, dict):
        return ""
    value = (
        item.get("role")
        or item.get("from")
        or item.get("speaker")
        or item.get("author")
        or ""
    )
    return normalize_text(value, collapse_spaces=True).lower()


def _message_content_from_dict(item: dict[str, Any]) -> Any:
    """Return the first common content field in a message dictionary."""
    for key in ("content", "value", "text", "message", "output"):
        if key in item:
            return item[key]
    return ""


def _coerce_message_sequence(value: Any) -> list[Any]:
    """Coerce common message containers into a raw message list."""
    if isinstance(value, str):
        parsed = parse_json_loose(value)
        return _coerce_message_sequence(parsed)
    if isinstance(value, dict):
        for key in ("messages", "conversations", "conversation", "chosen"):
            if key in value:
                return _coerce_message_sequence(value[key])
        return [value]
    if isinstance(value, list):
        return value
    return []


def normalize_tool_trace_messages(value: Any) -> list[dict[str, Any]]:
    """Normalize reasoning/tool_call/tool_output/answer trace roles.

    Reasoning is merged into the following assistant tool call or final answer.
    XML-style wrappers are removed while the payload becomes a standard
    ``assistant.tool_calls`` entry and tool result.
    """
    raw_messages = _coerce_message_sequence(value)
    messages: list[dict[str, Any]] = []
    pending_reasoning: list[str] = []
    pending_tool_name = ""
    pending_tool_call_id = ""

    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        raw_role = _raw_message_role(item)
        content = _message_content_from_dict(item)
        if raw_role == "reasoning":
            text = normalize_text(content)
            if text:
                pending_reasoning.append(text)
            continue
        if raw_role == "tool_call":
            payload: Any = content
            if isinstance(payload, str):
                payload = parse_json_loose(strip_message_wrapper(payload))
            call = normalize_tool_call(payload)
            if call is None:
                raise DataCleaningError("invalid tool_call payload")
            pending_tool_name = call["function"]["name"]
            pending_tool_call_id = normalize_text(call.get("id", ""))
            message: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": [call],
            }
            reasoning = "\n".join(pending_reasoning)
            if reasoning:
                message["content"] = reasoning
            pending_reasoning.clear()
            messages.append(message)
            continue
        if raw_role in {"tool_output", "tool_result"}:
            message = {
                "role": "tool",
                "content": strip_message_wrapper(content),
            }
            if pending_tool_name:
                message["name"] = pending_tool_name
            if pending_tool_call_id:
                message["tool_call_id"] = pending_tool_call_id
            messages.append(message)
            pending_tool_name = ""
            pending_tool_call_id = ""
            continue
        if raw_role == "answer":
            answer = strip_message_wrapper(content)
            fragments = [*pending_reasoning, answer]
            messages.append(
                {
                    "role": "assistant",
                    "content": "\n".join(part for part in fragments if part),
                }
            )
            pending_reasoning.clear()
            continue

        role = normalize_role(raw_role)
        message = {"role": role, "content": normalize_content(content)}
        if role == "tool":
            message["content"] = content_to_text(message["content"])
        tool_calls = normalize_tool_calls(
            item.get("tool_calls") or item.get("tool_call")
        )
        if tool_calls:
            message["tool_calls"] = tool_calls
        for key in ("name", "tool_call_id"):
            if item.get(key) is not None:
                message[key] = normalize_text(item[key], collapse_spaces=True)
        messages.append(message)

    if pending_reasoning:
        messages.append({"role": "assistant", "content": "\n".join(pending_reasoning)})
    return messages


def normalize_messages(
    value: Any,
    *,
    assume_alternating_roles: bool = True,
) -> list[dict[str, Any]]:
    """Normalize chat records into role/content/tool_calls dictionaries."""
    if isinstance(value, str):
        try:
            raw_messages = _coerce_message_sequence(value)
        except DataCleaningError:
            prefix = value.lstrip()
            if prefix.startswith(("[", "{", '"[', '"{')):
                raise
            return messages_from_transcript(value)
    else:
        raw_messages = _coerce_message_sequence(value)
    if any(
        _raw_message_role(item)
        in {"reasoning", "tool_call", "tool_output", "tool_result", "answer"}
        for item in raw_messages
    ):
        return normalize_tool_trace_messages(raw_messages)
    messages: list[dict[str, Any]] = []
    tool_names_by_id: dict[str, str] = {}
    for index, item in enumerate(raw_messages):
        fallback = None
        if assume_alternating_roles:
            fallback = "user" if index % 2 == 0 else "assistant"
        if isinstance(item, str):
            role = fallback or "user"
            message: dict[str, Any] = {
                "role": role,
                "content": normalize_content(item),
            }
            messages.append(message)
            continue
        if not isinstance(item, dict):
            continue
        role_source = (
            item.get("role")
            or item.get("from")
            or item.get("speaker")
            or item.get("author")
            or item.get("name")
        )
        if role_source is None and ("tool_calls" in item or "tool_call" in item):
            role_source = "assistant"
        role = normalize_role(role_source, fallback=fallback)
        message = {
            "role": role,
            "content": normalize_content(_message_content_from_dict(item)),
        }
        if role == "tool":
            message["content"] = content_to_text(message["content"])
        tool_calls = normalize_tool_calls(
            item.get("tool_calls") or item.get("tool_call")
        )
        if tool_calls:
            message["tool_calls"] = tool_calls
            for call in tool_calls:
                call_id = normalize_text(call.get("id", ""), collapse_spaces=True)
                if call_id:
                    tool_names_by_id[call_id] = call["function"]["name"]
        for key in ("name", "tool_call_id"):
            if key in item and item[key] is not None:
                message[key] = normalize_text(item[key], collapse_spaces=True)
        raw_call_ids = item.get("tool_call_ids")
        if (
            "tool_call_id" not in message
            and isinstance(raw_call_ids, list)
            and len(raw_call_ids) == 1
        ):
            message["tool_call_id"] = normalize_text(
                raw_call_ids[0], collapse_spaces=True
            )
        if role == "tool" and "name" not in message:
            call_id = message.get("tool_call_id", "")
            if call_id in tool_names_by_id:
                message["name"] = tool_names_by_id[call_id]
        messages.append(message)
    return messages


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def ensure_system_message(
    messages: list[dict[str, Any]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Guarantee a non-empty leading system message.

    Datasets share one fixed system prompt, so a missing or empty system
    message is filled with ``system_prompt`` instead of per-row labeling.
    An existing non-empty system message is always kept as-is.
    """
    rest = list(messages)
    if rest and isinstance(rest[0], dict) and rest[0].get("role") == "system":
        existing = content_to_text(rest[0].get("content", "")).strip()
        if existing:
            return rest
        rest = rest[1:]
    return [{"role": "system", "content": system_prompt}, *rest]


def multiple_choice_prompt(
    question: Any,
    choices: Iterable[Any],
    *,
    subject: Any | None = None,
) -> str:
    """Build a compact user prompt from a multiple-choice row."""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines: list[str] = []
    subject_text = normalize_text(subject, collapse_spaces=True)
    if subject_text:
        lines.append(f"Subject: {subject_text}")
    lines.append(normalize_text(question))
    for index, choice in enumerate(choices):
        label = labels[index] if index < len(labels) else str(index + 1)
        lines.append(f"{label}. {normalize_text(choice)}")
    return "\n".join(lines)


def answer_from_choices(answer: Any, choices: list[Any]) -> str:
    """Return a text answer from an index, label, or raw answer value."""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if isinstance(answer, int) and 0 <= answer < len(choices):
        return f"{labels[answer]}. {normalize_text(choices[answer])}"
    text = normalize_text(answer, collapse_spaces=True)
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match and match.group(1) in labels[: len(choices)]:
        index = labels.index(match.group(1))
        return f"{match.group(1)}. {normalize_text(choices[index])}"
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(choices):
            return f"{labels[index]}. {normalize_text(choices[index])}"
    return text


def _best_response_from_responses(value: Any) -> str:
    """Return the best text response from a scored responses list."""
    if isinstance(value, str):
        try:
            value = parse_json_loose(value)
        except DataCleaningError:
            return normalize_text(value)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return ""
    best_text = ""
    best_score = float("-inf")
    for item in value:
        if isinstance(item, str):
            text = normalize_text(item)
            score = 0.0
        elif isinstance(item, dict):
            text = normalize_text(
                item.get("response")
                or item.get("content")
                or item.get("text")
                or item.get("output")
            )
            score_value = item.get("score") or item.get("rank") or 0
            try:
                score = float(score_value)
            except (TypeError, ValueError):
                score = 0.0
        else:
            continue
        if text and score >= best_score:
            best_text = text
            best_score = score
    return best_text


def _trace_role(record: dict[str, Any]) -> str:
    """Infer a role from one agent trace event."""
    for key in ("role", "from", "speaker", "source", "sender", "event_type", "type"):
        if key in record and record[key] is not None:
            role = normalize_role(record[key], fallback="")
            if role:
                return role
    return "assistant"


def message_from_trace_event(record: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one trace/raw-event row into a message when possible."""
    content = None
    for key in (
        "content",
        "message",
        "event_msg",
        "text",
        "observation",
        "stdout",
        "stderr",
        "action",
        "thought",
    ):
        if key in record and record[key] is not None:
            content = record[key]
            break
    tool_calls = normalize_tool_calls(
        record.get("tool_calls")
        or record.get("tool_call")
        or record.get("tool")
        or record.get("function_call")
    )
    if content is None and not tool_calls:
        return None
    role = _trace_role(record)
    if tool_calls:
        role = "assistant"
    message: dict[str, Any] = {"role": role, "content": normalize_content(content)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    for key in ("tool_call_id", "name"):
        if key in record and record[key] is not None:
            message[key] = normalize_text(record[key], collapse_spaces=True)
    return message


def messages_from_trace_events(value: Any) -> list[dict[str, Any]]:
    """Convert a list of agent trace/event rows into messages."""
    if isinstance(value, str):
        try:
            value = parse_json_loose(value)
        except DataCleaningError:
            return messages_from_transcript(value)
    if isinstance(value, dict):
        for key in ("events", "trace", "trajectory", "steps"):
            if key in value:
                return messages_from_trace_events(value[key])
        message = message_from_trace_event(value)
        return [message] if message else []
    if not isinstance(value, list):
        return []
    messages: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            message = message_from_trace_event(item)
            if message is not None:
                messages.append(message)
        elif isinstance(item, str):
            messages.extend(messages_from_transcript(item))
    return messages


def messages_from_transcript(value: Any) -> list[dict[str, Any]]:
    """Parse a loose role-prefixed transcript into messages."""
    text = normalize_text(value)
    if not text:
        return []
    pattern = re.compile(
        r"(?im)^\s*(system|user|human|assistant|gpt|tool|observation)\s*:\s*"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return [{"role": "user", "content": text}]
    messages: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            messages.append(
                {
                    "role": normalize_role(match.group(1)),
                    "content": normalize_content(content),
                }
            )
    return messages


def messages_from_record(
    record: dict[str, Any],
    *,
    preference: str | None = None,
) -> list[dict[str, Any]]:
    """Infer messages, requiring an explicit branch for preference rows."""
    preference_branches = {
        branch for branch in ("chosen", "rejected") if branch in record
    }
    if preference_branches:
        if preference not in preference_branches:
            raise DataCleaningError(
                "preference row requires an explicitly available branch",
                code=ERROR_INVALID_FORMAT,
            )
        assert preference is not None
        branch = record[preference]
        if not isinstance(branch, str):
            return normalize_messages(branch)
        prompt = _preference_prompt(record)
        if not isinstance(prompt, str) or not prompt:
            raise DataCleaningError(
                "preference-to-SFT conversion requires a non-empty text prompt",
                code=ERROR_MISSING_FIELD,
            )
        response = normalize_text(branch)
        if not response:
            raise DataCleaningError(
                f"preference branch {preference} must not be empty",
                code=ERROR_EMPTY_STRING,
            )
        messages: list[dict[str, Any]] = []
        system = record.get("system") or record.get("system_prompt")
        if system:
            messages.append({"role": "system", "content": normalize_content(system)})
        messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        )
        return messages
    for key in ("messages", "conversations", "conversation"):
        if key in record and record[key]:
            return normalize_messages(record[key])

    for key in ("events", "trace", "trajectory", "steps"):
        if key in record and record[key]:
            messages = messages_from_trace_events(record[key])
            if messages:
                return messages

    if any(
        key in record
        for key in (
            "event_msg",
            "event_type",
            "source",
            "sender",
            "thought",
            "action",
            "observation",
        )
    ):
        messages = messages_from_trace_events(record)
        if messages:
            return messages

    prompt = record.get("prompt") or record.get("instruction") or record.get("question")
    response = (
        record.get("response")
        or record.get("output")
        or record.get("answer_text")
        or record.get("completion")
        or record.get("ground_truth")
        or _best_response_from_responses(record.get("responses"))
    )
    choices = record.get("choices")
    if (
        not normalize_text(response)
        and isinstance(choices, list)
        and "answer" in record
    ):
        response = answer_from_choices(record["answer"], choices)
    if isinstance(choices, list) and "question" in record:
        prompt = multiple_choice_prompt(
            record["question"],
            choices,
            subject=record.get("subject"),
        )
    system = record.get("system") or record.get("system_prompt")

    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": normalize_content(system)})
    if prompt:
        user_content = normalize_text(prompt)
        extra_input = normalize_text(record.get("input", ""))
        if extra_input and extra_input not in user_content:
            user_content = f"{user_content}\n\n{extra_input}"
        messages.append({"role": "user", "content": user_content})
    if response:
        messages.append({"role": "assistant", "content": normalize_content(response)})
    return messages


def to_messages_record(
    record: dict[str, Any],
    *,
    preference: str | None = None,
) -> dict[str, Any]:
    """Convert a raw row to the top-level ``messages`` shape."""
    return {"messages": messages_from_record(record, preference=preference)}


def _preference_prompt(record: dict[str, Any]) -> Any:
    for key in ("prompt", "instruction", "question"):
        if key in record:
            prompt = record[key]
            if not isinstance(prompt, str):
                return prompt
            normalized = normalize_text(prompt)
            extra_input = normalize_text(record.get("input", ""))
            if extra_input and extra_input not in normalized:
                return f"{normalized}\n\n{extra_input}"
            return normalized
    return None


def detect_training_format(
    record: dict[str, Any],
) -> Literal["messages", "preference"]:
    """Choose DPO when a source row contains both preference branches."""
    if "chosen" in record and "rejected" in record:
        return "preference"
    return "messages"


def to_preference_record(record: dict[str, Any]) -> dict[str, str]:
    """Normalize a text DPO row without discarding either preference branch."""
    values = {
        "prompt": _preference_prompt(record),
        "chosen": record.get("chosen"),
        "rejected": record.get("rejected"),
    }
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(value, str):
            raise DataCleaningError(
                f"DPO {key} must be a string; conversational preference data "
                "requires an explicit lossless mapping",
                code=ERROR_TYPE,
            )
        text = normalize_text(value)
        if not text:
            raise DataCleaningError(
                f"DPO {key} must not be empty",
                code=ERROR_EMPTY_STRING,
            )
        normalized[key] = text
    if normalized["chosen"] == normalized["rejected"]:
        raise DataCleaningError(
            "DPO chosen and rejected must be different",
            code=ERROR_INVALID_FORMAT,
        )
    return normalized


def to_training_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a source row to messages or text DPO preference format."""
    if detect_training_format(record) == "preference":
        return to_preference_record(record)
    return to_messages_record(record)


def _validate_content_parts(
    content: list[Any],
    path: str,
    line_number: int | None,
) -> list[ValidationError]:
    """Validate explicit Transformers multimodal content parts."""
    errors: list[ValidationError] = []
    if not content:
        return [
            ValidationError(
                ERROR_EMPTY_STRING,
                "content parts must not be empty",
                path,
                line_number,
            )
        ]
    for index, part in enumerate(content):
        part_path = f"{path}[{index}]"
        if not isinstance(part, dict):
            errors.append(
                ValidationError(
                    ERROR_TYPE,
                    "content part must be an object",
                    part_path,
                    line_number,
                )
            )
            continue
        part_type = part.get("type")
        if not isinstance(part_type, str) or not part_type.strip():
            errors.append(
                ValidationError(
                    ERROR_MISSING_FIELD,
                    "content part requires type",
                    f"{part_path}.type",
                    line_number,
                )
            )
            continue
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                errors.append(
                    ValidationError(
                        ERROR_TYPE,
                        "text content part requires string text",
                        f"{part_path}.text",
                        line_number,
                    )
                )
            elif not text.strip():
                errors.append(
                    ValidationError(
                        ERROR_EMPTY_STRING,
                        "text content part must not be empty",
                        f"{part_path}.text",
                        line_number,
                    )
                )
        elif part_type in {"image", "video", "audio"}:
            location = part.get("url") or part.get("path")
            if not isinstance(location, str) or not location.strip():
                errors.append(
                    ValidationError(
                        ERROR_MISSING_FIELD,
                        f"{part_type} content part requires url or path",
                        part_path,
                        line_number,
                    )
                )
        else:
            errors.append(
                ValidationError(
                    ERROR_INVALID_FORMAT,
                    f"unsupported content part type: {part_type}",
                    f"{part_path}.type",
                    line_number,
                )
            )
    return errors


def _validate_tool_calls(
    value: Any,
    path: str,
    line_number: int | None,
) -> list[ValidationError]:
    """Validate Transformers/OpenAI-style function tool calls."""
    if not isinstance(value, list):
        return [
            ValidationError(ERROR_TYPE, "tool_calls must be a list", path, line_number)
        ]
    if not value:
        return [
            ValidationError(
                ERROR_EMPTY_STRING,
                "tool_calls must not be empty",
                path,
                line_number,
            )
        ]
    errors: list[ValidationError] = []
    for index, call in enumerate(value):
        call_path = f"{path}[{index}]"
        if not isinstance(call, dict):
            errors.append(
                ValidationError(
                    ERROR_TYPE, "tool call must be an object", call_path, line_number
                )
            )
            continue
        if call.get("type") != "function":
            errors.append(
                ValidationError(
                    ERROR_INVALID_FORMAT,
                    "tool call type must be function",
                    f"{call_path}.type",
                    line_number,
                )
            )
        function = call.get("function")
        if not isinstance(function, dict):
            errors.append(
                ValidationError(
                    ERROR_TYPE,
                    "tool call function must be an object",
                    f"{call_path}.function",
                    line_number,
                )
            )
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(
                ValidationError(
                    ERROR_EMPTY_STRING,
                    "tool function name must be a non-empty string",
                    f"{call_path}.function.name",
                    line_number,
                )
            )
        if not isinstance(function.get("arguments"), dict):
            errors.append(
                ValidationError(
                    ERROR_TYPE,
                    "tool function arguments must be an object",
                    f"{call_path}.function.arguments",
                    line_number,
                )
            )
        if not isinstance(call.get("id"), str) or not call["id"].strip():
            errors.append(
                ValidationError(
                    ERROR_EMPTY_STRING,
                    "tool call id must be a non-empty string",
                    f"{call_path}.id",
                    line_number,
                )
            )
    return errors


def validate_messages_record(
    record: Any,
    *,
    line_number: int | None = None,
) -> list[ValidationError]:
    """Validate a record in messages format."""
    errors: list[ValidationError] = []
    if not isinstance(record, dict):
        return [
            ValidationError(ERROR_TYPE, "record must be an object", "$", line_number)
        ]
    extra_fields = sorted(set(record) - {"messages"})
    if extra_fields:
        errors.append(
            ValidationError(
                ERROR_INVALID_FORMAT,
                f"top-level fields are not allowed: {', '.join(extra_fields)}",
                "$",
                line_number,
            )
        )
    messages = record.get("messages")
    if messages is None:
        return [
            ValidationError(
                ERROR_MISSING_FIELD,
                "missing messages",
                "$.messages",
                line_number,
            )
        ]
    if not isinstance(messages, list):
        return [
            ValidationError(
                ERROR_TYPE,
                "messages must be a list",
                "$.messages",
                line_number,
            )
        ]
    if not messages:
        errors.append(
            ValidationError(
                ERROR_EMPTY_STRING,
                "messages must not be empty",
                "$.messages",
                line_number,
            )
        )
    has_user = False
    tool_call_positions: dict[str, int] = {}
    tool_results: list[tuple[int, Any]] = []
    for index, message in enumerate(messages):
        path = f"$.messages[{index}]"
        if not isinstance(message, dict):
            errors.append(
                ValidationError(
                    ERROR_TYPE,
                    "message must be an object",
                    path,
                    line_number,
                )
            )
            continue
        role = message.get("role")
        if role is None:
            errors.append(
                ValidationError(
                    ERROR_MISSING_FIELD,
                    "missing role",
                    f"{path}.role",
                    line_number,
                )
            )
        elif role not in ALLOWED_MESSAGE_ROLES:
            errors.append(
                ValidationError(
                    ERROR_INVALID_ROLE,
                    f"invalid role: {role}",
                    f"{path}.role",
                    line_number,
                )
            )
        elif role == "user":
            has_user = True
        has_content = "content" in message
        has_tool_calls = bool(message.get("tool_calls"))
        if not has_content and not has_tool_calls:
            errors.append(
                ValidationError(
                    ERROR_MISSING_FIELD,
                    "missing content or tool_calls",
                    path,
                    line_number,
                )
            )
            continue
        if has_content:
            content = message.get("content")
            if not isinstance(content, (str, list)):
                errors.append(
                    ValidationError(
                        ERROR_TYPE,
                        "content must be string or list",
                        f"{path}.content",
                        line_number,
                    )
                )
            elif isinstance(content, list):
                errors.extend(
                    _validate_content_parts(
                        content,
                        f"{path}.content",
                        line_number,
                    )
                )
            elif not content.strip() and not has_tool_calls:
                errors.append(
                    ValidationError(
                        ERROR_EMPTY_STRING,
                        "content must not be empty",
                        f"{path}.content",
                        line_number,
                    )
                )
        if (
            role == "tool"
            and has_content
            and not isinstance(message.get("content"), str)
        ):
            errors.append(
                ValidationError(
                    ERROR_TYPE,
                    "tool content must be a string",
                    f"{path}.content",
                    line_number,
                )
            )
        if "tool_calls" in message:
            if role != "assistant":
                errors.append(
                    ValidationError(
                        ERROR_INVALID_FORMAT,
                        "tool_calls are only valid on assistant messages",
                        f"{path}.tool_calls",
                        line_number,
                    )
                )
            errors.extend(
                _validate_tool_calls(
                    message["tool_calls"],
                    f"{path}.tool_calls",
                    line_number,
                )
            )
            if isinstance(message["tool_calls"], list):
                for call in message["tool_calls"]:
                    if isinstance(call, dict) and isinstance(call.get("id"), str):
                        call_id = call["id"]
                        if call_id in tool_call_positions:
                            errors.append(
                                ValidationError(
                                    ERROR_INVALID_FORMAT,
                                    f"duplicate tool call id: {call_id}",
                                    f"{path}.tool_calls",
                                    line_number,
                                )
                            )
                        else:
                            tool_call_positions[call_id] = index
        if role == "tool":
            tool_results.append((index, message.get("tool_call_id")))
            if not isinstance(message.get("name"), str) or not message["name"].strip():
                errors.append(
                    ValidationError(
                        ERROR_MISSING_FIELD,
                        "tool message requires a non-empty name",
                        f"{path}.name",
                        line_number,
                    )
                )
        for key in ("name", "tool_call_id"):
            if key in message and (
                not isinstance(message[key], str) or not message[key].strip()
            ):
                errors.append(
                    ValidationError(
                        ERROR_TYPE,
                        f"{key} must be a non-empty string",
                        f"{path}.{key}",
                        line_number,
                    )
                )
    for index, tool_call_id in tool_results:
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            errors.append(
                ValidationError(
                    ERROR_MISSING_FIELD,
                    "tool message requires a non-empty tool_call_id",
                    f"$.messages[{index}].tool_call_id",
                    line_number,
                )
            )
        elif (
            tool_call_id not in tool_call_positions
            or tool_call_positions[tool_call_id] >= index
        ):
            errors.append(
                ValidationError(
                    ERROR_INVALID_FORMAT,
                    f"tool_call_id does not match an earlier call: {tool_call_id}",
                    f"$.messages[{index}].tool_call_id",
                    line_number,
                )
            )
    if messages and not has_user:
        errors.append(
            ValidationError(
                ERROR_MISSING_FIELD,
                "messages must contain at least one user message",
                "$.messages",
                line_number,
            )
        )
    return errors


def validate_preference_record(
    record: Any,
    *,
    line_number: int | None = None,
) -> list[ValidationError]:
    """Validate Pyromind's fixed text DPO preference shape."""
    if not isinstance(record, dict):
        return [
            ValidationError(
                ERROR_TYPE,
                "record must be an object",
                "$",
                line_number,
            )
        ]
    errors: list[ValidationError] = []
    allowed = {"prompt", "chosen", "rejected"}
    extra_fields = sorted(set(record) - allowed)
    if extra_fields:
        errors.append(
            ValidationError(
                ERROR_INVALID_FORMAT,
                f"top-level fields are not allowed: {', '.join(extra_fields)}",
                "$",
                line_number,
            )
        )
    for key in ("prompt", "chosen", "rejected"):
        if key not in record:
            errors.append(
                ValidationError(
                    ERROR_MISSING_FIELD,
                    f"missing {key}",
                    f"$.{key}",
                    line_number,
                )
            )
            continue
        value = record[key]
        if not isinstance(value, str):
            errors.append(
                ValidationError(
                    ERROR_TYPE,
                    f"{key} must be a string",
                    f"$.{key}",
                    line_number,
                )
            )
        elif not value.strip():
            errors.append(
                ValidationError(
                    ERROR_EMPTY_STRING,
                    f"{key} must not be empty",
                    f"$.{key}",
                    line_number,
                )
            )
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if (
        isinstance(chosen, str)
        and isinstance(rejected, str)
        and chosen.strip()
        and chosen.strip() == rejected.strip()
    ):
        errors.append(
            ValidationError(
                ERROR_INVALID_FORMAT,
                "chosen and rejected must be different",
                "$",
                line_number,
            )
        )
    return errors


def detect_output_format(record: Any) -> Literal["messages", "preference"] | None:
    """Detect one normalized output row without guessing arbitrary schemas."""
    if not isinstance(record, dict):
        return None
    if "messages" in record:
        return "messages"
    if any(key in record for key in ("prompt", "chosen", "rejected")):
        return "preference"
    return None


def validate_record(
    record: Any,
    *,
    line_number: int | None = None,
) -> list[ValidationError]:
    """Validate a normalized messages or text DPO output row."""
    output_format = detect_output_format(record)
    if output_format == "messages":
        return validate_messages_record(record, line_number=line_number)
    if output_format == "preference":
        return validate_preference_record(record, line_number=line_number)
    return [
        ValidationError(
            ERROR_INVALID_FORMAT,
            "record must use messages or prompt/chosen/rejected format",
            "$",
            line_number,
        )
    ]


def _write_json_line(handle: Any, value: dict[str, Any]) -> None:
    handle.write((stable_json_dumps(value) + "\n").encode("utf-8"))


def _flush_durable(*handles: Any) -> None:
    for handle in handles:
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_payload(
    *,
    source_records_committed: int,
    output_offset: int,
    stats: CleaningStats,
    source_path: str | None = None,
    source_line: int | None = None,
) -> dict[str, Any]:
    stats_payload = stats.to_dict()
    stats_payload.pop("error_samples", None)
    return {
        "version": 2,
        "source_records_committed": source_records_committed,
        "output_offset": output_offset,
        "error_samples_committed": len(stats.error_samples),
        "source": {
            "index": source_records_committed,
            "path": source_path,
            "line": source_line,
        },
        "stats": stats_payload,
    }


def _report_payload(
    stats: CleaningStats,
    checkpoint: dict[str, Any],
    *,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats_payload = stats.to_dict()
    errors = stats_payload.pop("error_samples", [])
    return {
        "status": stats.status,
        "format": None,
        "stats": stats_payload,
        "checkpoint": checkpoint,
        "validation": validation,
        "errors": errors,
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataCleaningError(
            f"resume requested but report is missing: {path}",
            code=ERROR_INVALID_FORMAT,
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataCleaningError(
            f"could not read report: {exc}",
            code=ERROR_JSON_DECODE,
        ) from exc
    checkpoint = value.get("checkpoint") if isinstance(value, dict) else None
    required = {"source_records_committed", "output_offset", "stats"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise DataCleaningError(
            "report checkpoint has an unsupported structure",
            code=ERROR_INVALID_FORMAT,
        )
    checkpoint = dict(checkpoint)
    stats = dict(checkpoint["stats"])
    errors = value.get("errors", [])
    if isinstance(errors, list):
        committed = int(checkpoint.get("error_samples_committed", len(errors)))
        stats["error_samples"] = errors[:committed]
    checkpoint["stats"] = stats
    return checkpoint


def _truncate_to_checkpoint(handle: Any, offset: int, label: str) -> None:
    if offset < 0:
        raise DataCleaningError(
            f"checkpoint {label} offset must not be negative",
            code=ERROR_INVALID_FORMAT,
        )
    handle.seek(0, os.SEEK_END)
    if offset > handle.tell():
        raise DataCleaningError(
            f"checkpoint {label} offset exceeds the file size",
            code=ERROR_INVALID_FORMAT,
        )
    handle.truncate(offset)
    handle.seek(offset)


def _restore_deduper(
    path: Path,
    deduper: ExactDeduper,
) -> Literal["messages", "preference"] | None:
    output_format: Literal["messages", "preference"] | None = None
    for parsed in iter_jsonl(path, unwrap_huggingface=False):
        if not parsed.ok or not isinstance(parsed.data, dict):
            raise DataCleaningError(
                "checkpoint output contains invalid JSONL",
                code=ERROR_JSON_DECODE,
            )
        record_format = detect_output_format(parsed.data)
        if record_format is None:
            raise DataCleaningError(
                "checkpoint output contains an unsupported format",
                code=ERROR_INVALID_FORMAT,
            )
        if output_format is not None and record_format != output_format:
            raise DataCleaningError(
                "checkpoint output mixes training formats",
                code=ERROR_INVALID_FORMAT,
            )
        output_format = record_format
        deduper.is_duplicate(parsed.data)
    return output_format


def run_cleaning(
    *,
    input_path: str | Path,
    output_path: str | Path,
    state_dir: str | Path,
    mapper: Callable[[Any], dict[str, Any] | None],
    resume: bool = False,
    limit: int | None = None,
) -> CleaningStats:
    """Run a resumable streaming cleaner around source-specific mapping logic.

    A checkpoint is committed after output is durable. Resume truncates any
    uncommitted output tail before rebuilding exact dedupe state.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than 0")

    output = Path(output_path)
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = state / "report.json"
    deduper = ExactDeduper()
    committed = 0
    target_format: Literal["messages", "preference"] | None = None

    if resume:
        checkpoint = _load_checkpoint(report_path)
        if not output.is_file():
            raise DataCleaningError(
                "resume requires output.jsonl",
                code=ERROR_INVALID_FORMAT,
            )
        output_handle = output.open("r+b")
        stats = CleaningStats.from_dict(checkpoint["stats"])
        stats.status = "running"
        stats.failure = None
        committed = int(checkpoint["source_records_committed"])
        _truncate_to_checkpoint(
            output_handle, int(checkpoint["output_offset"]), "output"
        )
        target_format = _restore_deduper(output, deduper)
    else:
        output_handle = output.open("w+b")
        stats = CleaningStats()

    invocation_read = 0
    committed_source_path: str | None = None
    committed_source_line: int | None = None
    source_checkpoint = checkpoint.get("source") if resume else None
    if isinstance(source_checkpoint, dict):
        committed_source_path = source_checkpoint.get("path")
        committed_source_line = source_checkpoint.get("line")
    checkpoint = _checkpoint_payload(
        source_records_committed=committed,
        output_offset=output_handle.tell(),
        stats=stats,
        source_path=committed_source_path,
        source_line=committed_source_line,
    )
    try:
        with output_handle:
            _flush_durable(output_handle)
            atomic_write_json(report_path, _report_payload(stats, checkpoint))
            for source_index, parsed in enumerate(iter_records(input_path), start=1):
                if source_index <= committed:
                    continue
                if limit is not None and invocation_read >= limit:
                    break
                invocation_read += 1
                stats.record_input()

                if not parsed.ok:
                    code = parsed.error_type or ERROR_JSON_DECODE
                    message = parsed.error or "source record could not be parsed"
                    stats.record_drop(
                        code,
                        sample=parsed.raw,
                        line_number=parsed.line_number,
                        source_index=source_index,
                        source_path=parsed.source_path,
                        error=message,
                        is_error=True,
                    )
                else:
                    try:
                        cleaned = mapper(parsed.data)
                        if cleaned is None:
                            raise DropRecord("filtered")
                        if not isinstance(cleaned, dict):
                            raise DataCleaningError(
                                "mapper must return an object or None",
                                code=ERROR_TYPE,
                            )
                        validation_errors = validate_record(
                            cleaned,
                            line_number=parsed.line_number,
                        )
                        if validation_errors:
                            first = validation_errors[0]
                            raise DataCleaningError(
                                first.message,
                                code=first.code,
                                detail=first.path,
                            )
                        record_format = detect_output_format(cleaned)
                        if record_format is None:
                            raise DataCleaningError(
                                "mapper returned an unsupported output format",
                                code=ERROR_INVALID_FORMAT,
                            )
                        if target_format is not None and record_format != target_format:
                            raise DataCleaningError(
                                "one cleaning run cannot mix messages and DPO "
                                "preference records",
                                code=ERROR_INVALID_FORMAT,
                            )
                        target_format = record_format
                        if deduper.is_duplicate(cleaned):
                            stats.record_drop(
                                "duplicate",
                                sample=parsed.data,
                                line_number=parsed.line_number,
                                is_duplicate=True,
                            )
                        else:
                            _write_json_line(output_handle, cleaned)
                            stats.record_keep()
                    except DropRecord as exc:
                        stats.record_drop(
                            exc.code,
                            sample=parsed.data,
                            line_number=parsed.line_number,
                        )
                    except (DataCleaningError, KeyError, TypeError, ValueError) as exc:
                        code = getattr(exc, "code", ERROR_INVALID_FORMAT)
                        stats.record_drop(
                            code,
                            sample=parsed.data,
                            line_number=parsed.line_number,
                            source_index=source_index,
                            source_path=parsed.source_path,
                            error=str(exc),
                            is_error=True,
                        )

                _flush_durable(output_handle)
                committed = source_index
                committed_source_path = parsed.source_path
                committed_source_line = parsed.line_number
                checkpoint = _checkpoint_payload(
                    source_records_committed=committed,
                    output_offset=output_handle.tell(),
                    stats=stats,
                    source_path=committed_source_path,
                    source_line=committed_source_line,
                )
                atomic_write_json(report_path, _report_payload(stats, checkpoint))

            stats.status = "completed"
            _flush_durable(output_handle)
            checkpoint = _checkpoint_payload(
                source_records_committed=committed,
                output_offset=output_handle.tell(),
                stats=stats,
                source_path=committed_source_path,
                source_line=committed_source_line,
            )
            atomic_write_json(report_path, _report_payload(stats, checkpoint))
        return stats
    except Exception as exc:
        stats.status = "failed"
        stats.failure = f"{type(exc).__name__}: {exc}"
        atomic_write_json(report_path, _report_payload(stats, checkpoint))
        raise


def first_validation_reason(errors: list[ValidationError]) -> str:
    """Return the first validation error code or an empty string."""
    return errors[0].code if errors else ""
