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
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    """Collect row counts, drop reasons, and representative error samples."""

    total: int = 0
    kept: int = 0
    dropped: int = 0
    drop_reasons: Counter[str] = field(default_factory=Counter)
    error_samples: list[dict[str, Any]] = field(default_factory=list)
    max_error_samples: int = 20

    def record_input(self) -> None:
        """Increment the number of rows observed by a cleaning script."""
        self.total += 1

    def record_keep(self) -> None:
        """Increment the number of rows retained by a cleaning script."""
        self.kept += 1

    def record_drop(
        self,
        reason: str,
        *,
        sample: Any | None = None,
        line_number: int | None = None,
        error: str | None = None,
    ) -> None:
        """Increment a drop reason and keep a bounded sample for debugging."""
        self.dropped += 1
        self.drop_reasons[reason] += 1
        if len(self.error_samples) >= self.max_error_samples:
            return
        item: dict[str, Any] = {"reason": reason}
        if line_number is not None:
            item["line_number"] = line_number
        if error:
            item["error"] = error
        if sample is not None:
            item["sample"] = truncate_for_stats(sample)
        self.error_samples.append(item)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable stats payload."""
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped": self.dropped,
            "drop_reasons": dict(self.drop_reasons),
            "error_samples": self.error_samples,
        }

    def write_json(self, path: str | Path) -> None:
        """Write the stats payload to a JSON file."""
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


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


def iter_jsonl(path: str | Path) -> Iterator[ParsedRecord]:
    """Yield parsed JSONL records without stopping on bad lines."""
    with Path(path).open("rb") as handle:
        for line_number, blob in enumerate(handle, start=1):
            raw = decode_text(blob).strip()
            if not raw:
                continue
            try:
                yield ParsedRecord(parse_json_loose(raw), line_number, raw)
            except DataCleaningError as exc:
                yield ParsedRecord(
                    None,
                    line_number,
                    raw,
                    str(exc),
                    exc.code,
                )


def iter_json(path: str | Path) -> Iterator[ParsedRecord]:
    """Yield records from a JSON array, object, or JSONL-looking file."""
    raw = Path(path).read_bytes()
    text = decode_text(raw).strip()
    if not text:
        return
    first = text[:1]
    if first not in ("[", "{"):
        yield from iter_jsonl(path)
        return
    try:
        parsed = parse_json_loose(text)
    except DataCleaningError:
        yield from iter_jsonl(path)
        return
    if isinstance(parsed, list):
        for index, item in enumerate(parsed, start=1):
            yield ParsedRecord(item, index)
    else:
        yield ParsedRecord(parsed, 1)


def iter_csv(path: str | Path) -> Iterator[ParsedRecord]:
    """Yield CSV rows as dictionaries with tolerant dialect detection."""
    with Path(path).open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            return
        for line_number, row in enumerate(reader, start=2):
            cleaned: dict[str, Any] = {}
            for key, value in row.items():
                target_key = "_extra" if key is None else normalize_text(key)
                cleaned[target_key] = value
            yield ParsedRecord(cleaned, line_number)


def iter_records(path: str | Path) -> Iterator[ParsedRecord]:
    """Dispatch to a tolerant reader based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        yield from iter_csv(path)
        return
    if suffix == ".json":
        yield from iter_json(path)
        return
    yield from iter_jsonl(path)


def iter_huggingface_rows(
    path: str | Path,
    *,
    reject_truncated: bool = True,
) -> Iterator[ParsedRecord]:
    """Yield row payloads from a Hugging Face rows/first-rows API response.

    The dataset viewer wraps each payload as ``{"row": ..., "row_idx": ...}``.
    ``truncated_cells`` means the preview is not valid source data; reject such
    rows by default instead of misclassifying the truncated value as dirty JSON.
    """
    parsed_items = list(iter_json(path))
    if len(parsed_items) != 1 or not parsed_items[0].ok:
        yield from parsed_items
        return
    envelope = parsed_items[0].data
    if not isinstance(envelope, dict) or not isinstance(envelope.get("rows"), list):
        yield ParsedRecord(
            None,
            1,
            error="missing Hugging Face rows envelope",
            error_type=ERROR_INVALID_FORMAT,
        )
        return
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


def truncate_for_stats(value: Any, *, max_chars: int = 500) -> Any:
    """Return a compact sample suitable for stats.json."""
    text = value if isinstance(value, str) else stable_json_dumps(value)
    if len(text) <= max_chars:
        return value
    return text[: max_chars - 3] + "..."


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
    has_preference_pair = bool(record.get("chosen")) and bool(record.get("rejected"))
    if has_preference_pair:
        if preference not in {"chosen", "rejected"}:
            raise DataCleaningError(
                "preference row requires preference='chosen' or 'rejected'",
                code=ERROR_INVALID_FORMAT,
            )
        return normalize_messages(record[preference])
    for key in ("messages", "conversations", "conversation", "chosen"):
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
    keep_metadata: bool = True,
) -> dict[str, Any]:
    """Convert a raw row to messages with explicit preference semantics."""
    output: dict[str, Any] = {
        "messages": messages_from_record(record, preference=preference)
    }
    if not keep_metadata:
        return output
    metadata = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "messages",
            "conversations",
            "conversation",
            "chosen",
            "rejected",
            "events",
            "trace",
            "trajectory",
            "steps",
            "event_msg",
            "event_type",
            "source",
            "sender",
            "thought",
            "action",
            "observation",
            "prompt",
            "instruction",
            "question",
            "response",
            "responses",
            "output",
            "answer",
            "answer_text",
            "completion",
            "choices",
            "ground_truth",
            "input",
            "system",
            "system_prompt",
        }
    }
    if metadata:
        output["metadata"] = metadata
    return output


def to_alpaca_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw row or messages row into an alpaca-like JSON object."""
    messages = messages_from_record(record)
    system = ""
    user_parts: list[str] = []
    assistant_parts: list[str] = []
    for message in messages:
        role = message.get("role")
        text = content_to_text(message.get("content", ""))
        if role == "system" and not system:
            system = text
        elif role == "user":
            user_parts.append(text)
        elif role == "assistant":
            assistant_parts.append(text)
    return {
        "system": system,
        "input": "\n\n".join(part for part in user_parts if part),
        "output": "\n\n".join(part for part in assistant_parts if part),
    }


def to_sharegpt_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw row or messages row into ShareGPT conversations."""
    role_map = {"system": "system", "user": "human", "assistant": "gpt", "tool": "tool"}
    conversations = []
    for message in messages_from_record(record):
        conversations.append(
            {
                "from": role_map.get(
                    str(message.get("role")),
                    str(message.get("role")),
                ),
                "value": content_to_text(message.get("content", "")),
            }
        )
    return {"conversations": conversations}


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
        if "id" in call and not isinstance(call["id"], str):
            errors.append(
                ValidationError(
                    ERROR_TYPE,
                    "tool call id must be a string",
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
    return errors


def validate_alpaca_record(
    record: Any,
    *,
    line_number: int | None = None,
) -> list[ValidationError]:
    """Validate a system/input/output alpaca-like record."""
    errors: list[ValidationError] = []
    if not isinstance(record, dict):
        return [
            ValidationError(ERROR_TYPE, "record must be an object", "$", line_number)
        ]
    for key in ("input", "output"):
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
        if not isinstance(record[key], str):
            errors.append(
                ValidationError(
                    ERROR_TYPE,
                    f"{key} must be a string",
                    f"$.{key}",
                    line_number,
                )
            )
        elif not normalize_text(record[key]):
            errors.append(
                ValidationError(
                    ERROR_EMPTY_STRING,
                    f"{key} must not be empty",
                    f"$.{key}",
                    line_number,
                )
            )
    if "system" in record and not isinstance(record["system"], str):
        errors.append(
            ValidationError(
                ERROR_TYPE,
                "system must be a string",
                "$.system",
                line_number,
            )
        )
    return errors


def validate_sharegpt_record(
    record: Any,
    *,
    line_number: int | None = None,
) -> list[ValidationError]:
    """Validate a ShareGPT-style conversations record."""
    if not isinstance(record, dict):
        return [
            ValidationError(ERROR_TYPE, "record must be an object", "$", line_number)
        ]
    conversations = record.get("conversations")
    if conversations is None:
        return [
            ValidationError(
                ERROR_MISSING_FIELD,
                "missing conversations",
                "$.conversations",
                line_number,
            )
        ]
    if not isinstance(conversations, list):
        return [
            ValidationError(
                ERROR_TYPE,
                "conversations must be a list",
                "$.conversations",
                line_number,
            )
        ]
    mapped = {
        "messages": [
            {
                "role": (
                    normalize_role(item.get("from")) if isinstance(item, dict) else ""
                ),
                "content": item.get("value", "") if isinstance(item, dict) else "",
            }
            for item in conversations
        ]
    }
    return validate_messages_record(mapped, line_number=line_number)


def validate_record(
    record: Any,
    target_format: str,
    *,
    line_number: int | None = None,
) -> list[ValidationError]:
    """Validate a record against alpaca, sharegpt, or messages."""
    if target_format == "messages":
        return validate_messages_record(record, line_number=line_number)
    if target_format == "alpaca":
        return validate_alpaca_record(record, line_number=line_number)
    if target_format == "sharegpt":
        return validate_sharegpt_record(record, line_number=line_number)
    return [
        ValidationError(
            ERROR_INVALID_FORMAT,
            f"unsupported format: {target_format}",
            "$",
            line_number,
        )
    ]


def first_validation_reason(errors: list[ValidationError]) -> str:
    """Return the first validation error code or an empty string."""
    return errors[0].code if errors else ""
