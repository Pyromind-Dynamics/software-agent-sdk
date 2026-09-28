from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Literal


logger = logging.getLogger(__name__)

PLATFORM_MAX_INLINE_TEXT_BYTES = 50 * 1024
DEFAULT_MAX_VISIBLE_CHARS: int | None = None
DEFAULT_MAX_INLINE_BYTES: int | None = None
DEFAULT_TAIL_RATIO = 0.25

_OUTPUT_ID_LENGTH = 64
_LOWERCASE_HEX_DIGITS = frozenset("0123456789abcdef")
_FIXED_OUTPUT_ID = "0" * _OUTPUT_ID_LENGTH
_BUSINESS_TOOL_OUTPUT_PREFIX = "business-tool-"
_BUSINESS_TOOL_OUTPUT_SUFFIX = ".txt"


def _normalize_utf8_text(text: str) -> tuple[str, int]:
    normalized: list[str] = []
    replacements = 0
    index = 0
    while index < len(text):
        value = ord(text[index])
        if 0xD800 <= value <= 0xDBFF and index + 1 < len(text):
            low = ord(text[index + 1])
            if 0xDC00 <= low <= 0xDFFF:
                normalized.append(
                    chr(0x10000 + ((value - 0xD800) << 10) + low - 0xDC00)
                )
                index += 2
                continue
        if 0xD800 <= value <= 0xDFFF:
            normalized.append("\N{REPLACEMENT CHARACTER}")
            replacements += 1
        else:
            normalized.append(text[index])
        index += 1
    return "".join(normalized), replacements


def normalize_utf8_text(text: str) -> str:
    return _normalize_utf8_text(text)[0]


def business_tool_output_id(text: str) -> str:
    normalized = normalize_utf8_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_output_id(output_id: str) -> None:
    if len(output_id) != _OUTPUT_ID_LENGTH or any(
        character not in _LOWERCASE_HEX_DIGITS for character in output_id
    ):
        raise ValueError(
            "output_id must be a 64-character lowercase SHA-256 hex digest"
        )


def build_business_tool_output_filename(output_id: str) -> str:
    _validate_output_id(output_id)
    return f"{_BUSINESS_TOOL_OUTPUT_PREFIX}{output_id}{_BUSINESS_TOOL_OUTPUT_SUFFIX}"


def parse_business_tool_output_filename(filename: str) -> str | None:
    if not filename.startswith(_BUSINESS_TOOL_OUTPUT_PREFIX) or not filename.endswith(
        _BUSINESS_TOOL_OUTPUT_SUFFIX
    ):
        return None
    output_id = filename[
        len(_BUSINESS_TOOL_OUTPUT_PREFIX) : -len(_BUSINESS_TOOL_OUTPUT_SUFFIX)
    ]
    try:
        _validate_output_id(output_id)
    except ValueError:
        return None
    return output_id


def build_business_tool_output_relative_path(output_id: str) -> str:
    filename = build_business_tool_output_filename(output_id)
    return f"pi/terminal-output/{filename}"


def build_compact_tool_output_marker(output_id: str) -> str:
    path = build_business_tool_output_relative_path(output_id)
    return f"[tool output truncated; full output: {path}]"


def build_compact_unavailable_marker() -> str:
    return "[tool output truncated; full output unavailable]"


def build_tool_output_marker(
    *,
    tool_name: str,
    original_chars: int,
    original_bytes: int,
    original_lines: int,
    output_id: str,
    full_output_available: bool = True,
) -> str:
    path = (
        build_business_tool_output_relative_path(output_id)
        if full_output_available
        else None
    )
    normalized_tool_name = normalize_utf8_text(tool_name)
    location = (
        f"Full output: {path}\nUse terminal commands such as sed to inspect the file.\n"
        if path is not None
        else "Full output could not be persisted.\n"
    )
    return (
        "<tool-output-truncated>\n"
        f"Tool: {normalized_tool_name}\n"
        f"Original: {original_chars:,} characters, {original_bytes:,} UTF-8 bytes, "
        f"{original_lines:,} lines\n"
        f"SHA-256: {output_id}\n"
        f"{location}"
        "</tool-output-truncated>"
    )


_COMPACT_MARKER_TEMPLATE = build_compact_tool_output_marker(_FIXED_OUTPUT_ID)
MIN_POLICY_VISIBLE_CHARS = len(_COMPACT_MARKER_TEMPLATE)
MIN_POLICY_INLINE_BYTES = len(_COMPACT_MARKER_TEMPLATE.encode("utf-8"))

OutputStrategy = Literal["head_tail", "head", "tail"]
TruncationTrigger = Literal["chars", "bytes", "both"]

_OUTPUT_STRATEGIES: tuple[OutputStrategy, ...] = ("head_tail", "head", "tail")
_OUTPUT_POLICY_FIELDS = {
    "max_visible_chars",
    "max_inline_bytes",
    "strategy",
    "tail_ratio",
}


@dataclass(frozen=True, slots=True)
class ToolOutputPolicy:
    max_visible_chars: int | None = DEFAULT_MAX_VISIBLE_CHARS
    max_inline_bytes: int | None = DEFAULT_MAX_INLINE_BYTES
    strategy: OutputStrategy = "head_tail"
    tail_ratio: float = DEFAULT_TAIL_RATIO

    def __post_init__(self) -> None:
        _validate_optional_positive_int(
            self.max_visible_chars,
            field_name="max_visible_chars",
        )
        _validate_optional_positive_int(
            self.max_inline_bytes,
            field_name="max_inline_bytes",
        )
        if (
            self.max_visible_chars is not None
            and self.max_visible_chars < MIN_POLICY_VISIBLE_CHARS
        ):
            raise ValueError(
                "max_visible_chars is too small to contain the compact "
                "truncation marker"
            )
        if (
            self.max_inline_bytes is not None
            and self.max_inline_bytes < MIN_POLICY_INLINE_BYTES
        ):
            raise ValueError(
                "max_inline_bytes is too small to contain the compact truncation marker"
            )
        if self.strategy not in _OUTPUT_STRATEGIES:
            raise ValueError(f"invalid output strategy: {self.strategy!r}")
        if isinstance(self.tail_ratio, bool) or not isinstance(
            self.tail_ratio, (int, float)
        ):
            raise ValueError("tail_ratio must be a number between 0 and 1")
        if not 0 < self.tail_ratio < 1:
            raise ValueError("tail_ratio must be between 0 and 1")

    @property
    def effective_max_inline_text_bytes(self) -> int:
        requested_bytes = self.max_inline_bytes
        return min(
            requested_bytes
            if requested_bytes is not None
            else PLATFORM_MAX_INLINE_TEXT_BYTES,
            PLATFORM_MAX_INLINE_TEXT_BYTES,
        )


@dataclass(frozen=True, slots=True)
class ToolOutputTruncation:
    content: list[dict[str, Any]]
    truncated: bool
    original_chars: int
    original_bytes: int
    retained_chars: int
    retained_bytes: int
    omitted_chars: int
    omitted_bytes: int
    original_lines: int
    triggered_by: TruncationTrigger | None
    invalid_unicode_replacements: int


@dataclass(frozen=True, slots=True)
class _RetainedRange:
    chars: int
    bytes: int


def resolve_tool_output_policy(meta: dict[str, Any] | None) -> ToolOutputPolicy:
    if meta is None or "output_policy" not in meta:
        return ToolOutputPolicy()

    raw = meta["output_policy"]
    if not isinstance(raw, dict):
        raise ValueError("output_policy must be an object")

    unknown_fields = set(raw) - _OUTPUT_POLICY_FIELDS
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"unknown output_policy fields: {names}")

    max_visible_chars = _optional_positive_int(
        raw.get("max_visible_chars"),
        field_name="max_visible_chars",
    )
    max_inline_bytes = _optional_positive_int(
        raw.get("max_inline_bytes"),
        field_name="max_inline_bytes",
    )
    if (
        max_inline_bytes is not None
        and max_inline_bytes > PLATFORM_MAX_INLINE_TEXT_BYTES
    ):
        logger.warning(
            "Tool output max_inline_bytes=%d exceeds the platform text limit; using %d",
            max_inline_bytes,
            PLATFORM_MAX_INLINE_TEXT_BYTES,
        )
        max_inline_bytes = PLATFORM_MAX_INLINE_TEXT_BYTES

    return ToolOutputPolicy(
        max_visible_chars=max_visible_chars,
        max_inline_bytes=max_inline_bytes,
        strategy=_choice(
            raw.get("strategy", "head_tail"),
            _OUTPUT_STRATEGIES,
            field_name="strategy",
        ),
        tail_ratio=_tail_ratio(raw.get("tail_ratio", DEFAULT_TAIL_RATIO)),
    )


def truncate_tool_content(
    content: list[dict[str, Any]],
    *,
    tool_name: str,
    output_id: str,
    policy: ToolOutputPolicy,
    full_output_available: bool = True,
) -> ToolOutputTruncation:
    _validate_output_id(output_id)
    normalized_content: list[dict[str, Any]] = []
    invalid_unicode_replacements = 0
    for block in content:
        text = block.get("text")
        if block.get("type") != "text" or not isinstance(text, str):
            normalized_content.append(dict(block))
            continue
        normalized_text, replacements = _normalize_utf8_text(text)
        invalid_unicode_replacements += replacements
        normalized_content.append({**block, "text": normalized_text})
    full_text = "".join(
        block["text"]
        for block in normalized_content
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )
    original_chars = len(full_text)
    original_bytes = _utf8_size(full_text)
    original_lines = full_text.count("\n") + 1 if full_text else 0
    effective_max_bytes = policy.effective_max_inline_text_bytes
    chars_exceeded = (
        policy.max_visible_chars is not None
        and original_chars > policy.max_visible_chars
    )
    bytes_exceeded = original_bytes > effective_max_bytes
    triggered_by = _trigger(chars_exceeded, bytes_exceeded)

    if triggered_by is None:
        return ToolOutputTruncation(
            content=normalized_content,
            truncated=False,
            original_chars=original_chars,
            original_bytes=original_bytes,
            retained_chars=original_chars,
            retained_bytes=original_bytes,
            omitted_chars=0,
            omitted_bytes=0,
            original_lines=original_lines,
            triggered_by=None,
            invalid_unicode_replacements=invalid_unicode_replacements,
        )

    if output_id != business_tool_output_id(full_text):
        raise ValueError("output_id does not match the normalized tool output")
    marker = build_tool_output_marker(
        tool_name=tool_name,
        original_chars=original_chars,
        original_bytes=original_bytes,
        original_lines=original_lines,
        output_id=output_id,
        full_output_available=full_output_available,
    )
    selected_marker = _select_marker(
        marker=marker,
        compact_marker=(
            build_compact_tool_output_marker(output_id)
            if full_output_available
            else build_compact_unavailable_marker()
        ),
        max_chars=policy.max_visible_chars,
        max_bytes=effective_max_bytes,
    )
    marker_chars = len(selected_marker)
    marker_bytes = _utf8_size(selected_marker)
    remaining_chars = (
        policy.max_visible_chars - marker_chars
        if policy.max_visible_chars is not None
        else None
    )
    remaining_bytes = effective_max_bytes - marker_bytes
    head, tail = _retained_ranges(
        full_text,
        policy=policy,
        max_chars=remaining_chars,
        max_bytes=remaining_bytes,
    )
    output = _replace_removed_text(
        normalized_content,
        marker=selected_marker,
        head_chars=head.chars,
        tail_chars=tail.chars,
        original_chars=original_chars,
    )

    retained_chars = head.chars + marker_chars + tail.chars
    retained_bytes = head.bytes + marker_bytes + tail.bytes
    if (
        policy.max_visible_chars is not None
        and retained_chars > policy.max_visible_chars
    ):
        raise AssertionError("truncated content exceeds max_visible_chars")
    if retained_bytes > effective_max_bytes:
        raise AssertionError("truncated content exceeds max_inline_bytes")
    visible_text = "".join(
        block["text"]
        for block in output
        if block.get("type") == "text" and isinstance(block.get("text"), str)
    )
    if len(visible_text) != retained_chars:
        raise AssertionError("retained_chars does not match the truncated content")
    if _utf8_size(visible_text) != retained_bytes:
        raise AssertionError("retained_bytes does not match the truncated content")

    return ToolOutputTruncation(
        content=output,
        truncated=True,
        original_chars=original_chars,
        original_bytes=original_bytes,
        retained_chars=retained_chars,
        retained_bytes=retained_bytes,
        omitted_chars=original_chars - head.chars - tail.chars,
        omitted_bytes=original_bytes - head.bytes - tail.bytes,
        original_lines=original_lines,
        triggered_by=triggered_by,
        invalid_unicode_replacements=invalid_unicode_replacements,
    )


def _retained_ranges(
    text: str,
    *,
    policy: ToolOutputPolicy,
    max_chars: int | None,
    max_bytes: int,
) -> tuple[_RetainedRange, _RetainedRange]:
    if policy.strategy == "head":
        return _scan_prefix(text, max_chars=max_chars, max_bytes=max_bytes), (
            _RetainedRange(0, 0)
        )
    if policy.strategy == "tail":
        return _RetainedRange(0, 0), _scan_suffix(
            text,
            max_chars=max_chars,
            max_bytes=max_bytes,
        )

    tail_char_target = (
        int(max_chars * policy.tail_ratio) if max_chars is not None else None
    )
    tail_byte_target = int(max_bytes * policy.tail_ratio)
    tail = _scan_suffix(
        text,
        max_chars=tail_char_target,
        max_bytes=tail_byte_target,
    )
    head_char_budget = max_chars - tail.chars if max_chars is not None else None
    head_byte_budget = max_bytes - tail.bytes
    head = _scan_prefix(
        text[: len(text) - tail.chars],
        max_chars=head_char_budget,
        max_bytes=head_byte_budget,
    )
    return head, tail


def _scan_prefix(
    text: str,
    *,
    max_chars: int | None,
    max_bytes: int,
) -> _RetainedRange:
    retained_chars = 0
    retained_bytes = 0
    for point in text:
        point_bytes = _utf8_size(point)
        if max_chars is not None and retained_chars + 1 > max_chars:
            break
        if retained_bytes + point_bytes > max_bytes:
            break
        retained_chars += 1
        retained_bytes += point_bytes
    return _RetainedRange(retained_chars, retained_bytes)


def _scan_suffix(
    text: str,
    *,
    max_chars: int | None,
    max_bytes: int,
) -> _RetainedRange:
    retained_chars = 0
    retained_bytes = 0
    for point in reversed(text):
        point_bytes = _utf8_size(point)
        if max_chars is not None and retained_chars + 1 > max_chars:
            break
        if retained_bytes + point_bytes > max_bytes:
            break
        retained_chars += 1
        retained_bytes += point_bytes
    return _RetainedRange(retained_chars, retained_bytes)


def _replace_removed_text(
    content: list[dict[str, Any]],
    *,
    marker: str,
    head_chars: int,
    tail_chars: int,
    original_chars: int,
) -> list[dict[str, Any]]:
    tail_start = original_chars - tail_chars
    output: list[dict[str, Any]] = []
    text_offset = 0
    marker_inserted = False

    for block in content:
        text = block.get("text")
        if block.get("type") != "text" or not isinstance(text, str):
            output.append(dict(block))
            continue
        if not text:
            output.append(dict(block))
            continue

        block_start = text_offset
        block_end = block_start + len(text)
        text_offset = block_end
        pieces: list[str] = []

        head_end = min(block_end, head_chars)
        if head_end > block_start:
            pieces.append(text[: head_end - block_start])

        removed_start = max(block_start, head_chars)
        removed_end = min(block_end, tail_start)
        if removed_start < removed_end and not marker_inserted:
            pieces.append(marker)
            marker_inserted = True

        tail_overlap_start = max(block_start, tail_start)
        if tail_overlap_start < block_end:
            pieces.append(text[tail_overlap_start - block_start :])

        if pieces:
            output.append({**block, "text": "".join(pieces)})

    if not marker_inserted:
        raise AssertionError("truncated content did not contain a removed range")
    return output


def _select_marker(
    *,
    marker: str,
    compact_marker: str,
    max_chars: int | None,
    max_bytes: int,
) -> str:
    if not marker or not compact_marker:
        raise ValueError("markers must not be empty")
    if _fits(marker, max_chars=max_chars, max_bytes=max_bytes):
        return marker
    if _fits(compact_marker, max_chars=max_chars, max_bytes=max_bytes):
        return compact_marker
    raise ValueError("compact marker exceeds the configured text budget")


def _fits(text: str, *, max_chars: int | None, max_bytes: int) -> bool:
    return (max_chars is None or len(text) <= max_chars) and (
        _utf8_size(text) <= max_bytes
    )


def _trigger(
    chars_exceeded: bool,
    bytes_exceeded: bool,
) -> TruncationTrigger | None:
    if chars_exceeded and bytes_exceeded:
        return "both"
    if chars_exceeded:
        return "chars"
    if bytes_exceeded:
        return "bytes"
    return None


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    _validate_optional_positive_int(value, field_name=field_name)
    return value


def _validate_optional_positive_int(value: Any, *, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer or null")


def _tail_ratio(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("tail_ratio must be a number between 0 and 1")
    ratio = float(value)
    if not 0 < ratio < 1:
        raise ValueError("tail_ratio must be between 0 and 1")
    return ratio


def _choice[T: str](value: Any, choices: tuple[T, ...], *, field_name: str) -> T:
    if value not in choices:
        raise ValueError(f"invalid {field_name}: {value!r}")
    return value
