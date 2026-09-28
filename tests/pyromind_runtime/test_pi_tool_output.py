from __future__ import annotations

import stat
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from harness_adapter.pi_adapter import persistence
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.tool_output import (
    MIN_POLICY_INLINE_BYTES,
    MIN_POLICY_VISIBLE_CHARS,
    PLATFORM_MAX_INLINE_TEXT_BYTES,
    OutputStrategy,
    ToolOutputPolicy,
    build_business_tool_output_relative_path,
    build_compact_tool_output_marker,
    build_tool_output_marker,
    business_tool_output_id,
    normalize_utf8_text,
    resolve_tool_output_policy,
    truncate_tool_content,
)


OUTPUT_ID = "a" * 64

TOOL_OUTPUT_RETENTION_CASES = [
    {
        "name": "ascii-character-limit",
        "tool_name": "golden_tool",
        "content": [{"type": "text", "text": "abcdefghijklmnopqrstuvwxyz" * 7}],
        "policy": {
            "max_visible_chars": 147,
            "max_inline_bytes": 1000,
            "tail_ratio": 0.25,
        },
        "expected": {
            "output_id": (
                "68ab20d9e0f0a4076dfcf2c0328cf72368213956f420e8afa19b4b9053830f56"
            ),
            "content": [
                {
                    "type": "text",
                    "text": (
                        "abcdef"
                        "[tool output truncated; full output: "
                        "pi/terminal-output/"
                        "business-tool-68ab20d9e0f0a4076dfcf2c0328cf72368213956f420e8afa19b4b9053830f56.txt]"
                        "yz"
                    ),
                }
            ],
            "original_chars": 182,
            "original_bytes": 182,
            "retained_chars": 147,
            "retained_bytes": 147,
            "omitted_chars": 174,
            "omitted_bytes": 174,
            "original_lines": 1,
            "triggered_by": "chars",
            "invalid_unicode_replacements": 0,
        },
    },
    {
        "name": "unicode-byte-limit",
        "tool_name": "golden_tool",
        "content": [{"type": "text", "text": "你" * 60}],
        "policy": {
            "max_visible_chars": None,
            "max_inline_bytes": 151,
            "tail_ratio": 0.25,
        },
        "expected": {
            "output_id": (
                "bf6841c51e5f0c76f038fd90addd02a450331ee2578a4578653ad89fbbd4406d"
            ),
            "content": [
                {
                    "type": "text",
                    "text": (
                        "你你你"
                        "[tool output truncated; full output: "
                        "pi/terminal-output/"
                        "business-tool-bf6841c51e5f0c76f038fd90addd02a450331ee2578a4578653ad89fbbd4406d.txt]"
                        "你"
                    ),
                }
            ],
            "original_chars": 60,
            "original_bytes": 180,
            "retained_chars": 143,
            "retained_bytes": 151,
            "omitted_chars": 56,
            "omitted_bytes": 168,
            "original_lines": 1,
            "triggered_by": "bytes",
            "invalid_unicode_replacements": 0,
        },
    },
    {
        "name": "dual-limit-across-blocks",
        "tool_name": "golden_tool",
        "content": [
            {"type": "text", "text": "你" * 10},
            {"type": "image", "data": "unchanged"},
            {"type": "text", "text": "abcdefghijklmnopqrstuvwxyz" * 6},
        ],
        "policy": {
            "max_visible_chars": 147,
            "max_inline_bytes": 151,
            "tail_ratio": 0.25,
        },
        "expected": {
            "output_id": (
                "919b31f84045284dd2fd6bc616b8b97c317392b7d5e49223864b1ba9726122a2"
            ),
            "content": [
                {
                    "type": "text",
                    "text": (
                        "你你你"
                        "[tool output truncated; full output: "
                        "pi/terminal-output/"
                        "business-tool-919b31f84045284dd2fd6bc616b8b97c317392b7d5e49223864b1ba9726122a2.txt]"
                    ),
                },
                {"type": "image", "data": "unchanged"},
                {"type": "text", "text": "yz"},
            ],
            "original_chars": 166,
            "original_bytes": 186,
            "retained_chars": 144,
            "retained_bytes": 150,
            "omitted_chars": 161,
            "omitted_bytes": 175,
            "original_lines": 1,
            "triggered_by": "both",
            "invalid_unicode_replacements": 0,
        },
    },
    {
        "name": "compact-marker-only",
        "tool_name": "golden_tool",
        "content": [{"type": "text", "text": "a" * 140}],
        "policy": {
            "max_visible_chars": None,
            "max_inline_bytes": 139,
            "tail_ratio": 0.25,
        },
        "expected": {
            "output_id": (
                "c094ed2f614ab7a02e7557f8eba6b03b457ce7beacf1d8031088f97a1770e5e6"
            ),
            "content": [
                {
                    "type": "text",
                    "text": (
                        "[tool output truncated; full output: "
                        "pi/terminal-output/"
                        "business-tool-c094ed2f614ab7a02e7557f8eba6b03b457ce7beacf1d8031088f97a1770e5e6.txt]"
                    ),
                }
            ],
            "original_chars": 140,
            "original_bytes": 140,
            "retained_chars": 139,
            "retained_bytes": 139,
            "omitted_chars": 140,
            "omitted_bytes": 140,
            "original_lines": 1,
            "triggered_by": "bytes",
            "invalid_unicode_replacements": 0,
        },
    },
    {
        "name": "surrogate-normalization",
        "tool_name": "golden_tool",
        "content": [{"type": "text", "text": "a\ud800b😀\udc80"}],
        "policy": {},
        "expected": {
            "output_id": (
                "332828b722e8236b8e1e72c12148047f7ec60e552cb9a9bdb7f75a98525d067a"
            ),
            "content": [{"type": "text", "text": "a�b😀�"}],
            "original_chars": 5,
            "original_bytes": 12,
            "retained_chars": 5,
            "retained_bytes": 12,
            "omitted_chars": 0,
            "omitted_bytes": 0,
            "original_lines": 1,
            "triggered_by": None,
            "invalid_unicode_replacements": 2,
        },
    },
]


def _visible_text(content: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        text
        for block in content
        if block.get("type") == "text" and isinstance((text := block.get("text")), str)
    )


def _truncate(
    content: Sequence[Mapping[str, object]],
    policy: ToolOutputPolicy,
    *,
    strategy_name: str = "test_tool",
):
    full_text = _visible_text(content)
    return truncate_tool_content(
        [dict(block) for block in content],
        tool_name=strategy_name,
        output_id=business_tool_output_id(full_text),
        policy=policy,
    )


def test_resolve_tool_output_policy_uses_defaults_and_caps_bytes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert resolve_tool_output_policy(None) == ToolOutputPolicy()

    policy = resolve_tool_output_policy(
        {
            "output_policy": {
                "max_visible_chars": MIN_POLICY_VISIBLE_CHARS + 1,
                "max_inline_bytes": PLATFORM_MAX_INLINE_TEXT_BYTES + 1,
                "strategy": "tail",
                "tail_ratio": 0.4,
            }
        }
    )

    assert policy.max_visible_chars == MIN_POLICY_VISIBLE_CHARS + 1
    assert policy.max_inline_bytes == PLATFORM_MAX_INLINE_TEXT_BYTES
    assert policy.effective_max_inline_text_bytes == PLATFORM_MAX_INLINE_TEXT_BYTES
    assert policy.strategy == "tail"
    assert policy.tail_ratio == 0.4
    assert "exceeds the platform text limit" in caplog.text


def test_tool_output_policy_direct_constructor_enforces_marker_minimum() -> None:
    with pytest.raises(ValueError, match="too small"):
        ToolOutputPolicy(max_visible_chars=MIN_POLICY_VISIBLE_CHARS - 1)
    with pytest.raises(ValueError, match="too small"):
        ToolOutputPolicy(max_inline_bytes=MIN_POLICY_INLINE_BYTES - 1)


@pytest.mark.parametrize(
    "output_policy",
    [
        "invalid",
        {"unknown": 1},
        {"max_visible_chars": 0},
        {"max_visible_chars": MIN_POLICY_VISIBLE_CHARS - 1},
        {"max_inline_bytes": MIN_POLICY_INLINE_BYTES - 1},
        {"strategy": "invalid"},
        {"tail_ratio": 1},
        {"history_mode": "summary_preferred"},
    ],
)
def test_resolve_tool_output_policy_rejects_invalid_config(
    output_policy: object,
) -> None:
    with pytest.raises(ValueError):
        resolve_tool_output_policy({"output_policy": output_policy})


def test_marker_path_and_policy_minimum_share_one_fixed_format() -> None:
    path = build_business_tool_output_relative_path(OUTPUT_ID)
    marker = build_compact_tool_output_marker(OUTPUT_ID)

    assert path == f"pi/terminal-output/business-tool-{OUTPUT_ID}.txt"
    assert marker == f"[tool output truncated; full output: {path}]"
    assert len(marker) == MIN_POLICY_VISIBLE_CHARS
    assert len(marker.encode("utf-8")) == MIN_POLICY_INLINE_BYTES


@pytest.mark.parametrize(
    "output_id",
    ["a" * 63, "a" * 65, "A" * 64, "g" * 64],
)
def test_marker_builder_rejects_noncanonical_output_ids(output_id: str) -> None:
    with pytest.raises(ValueError, match="64-character lowercase"):
        build_compact_tool_output_marker(output_id)


def test_full_marker_has_no_retained_size_cycle() -> None:
    marker = build_tool_output_marker(
        tool_name="training_analysis",
        original_chars=10_000,
        original_bytes=20_000,
        original_lines=300,
        output_id=OUTPUT_ID,
    )

    assert "Original: 10,000 characters, 20,000 UTF-8 bytes, 300 lines" in marker
    assert "Retained:" not in marker
    assert f"SHA-256: {OUTPUT_ID}" in marker


def test_normalize_utf8_text_matches_text_encoder_surrogate_semantics() -> None:
    text = "a" + "\ud83d\ude00" + "b" + "\ud800" + "c" + "\udc80"

    assert normalize_utf8_text(text) == "a😀b�c�"


def test_content_at_exact_character_and_byte_limits_is_unchanged() -> None:
    content = [{"type": "text", "text": "a" * MIN_POLICY_VISIBLE_CHARS}]
    policy = ToolOutputPolicy(
        max_visible_chars=MIN_POLICY_VISIBLE_CHARS,
        max_inline_bytes=MIN_POLICY_INLINE_BYTES,
    )

    result = _truncate(content, policy)

    assert result.content == content
    assert result.truncated is False
    assert result.original_chars == MIN_POLICY_VISIBLE_CHARS
    assert result.original_bytes == MIN_POLICY_INLINE_BYTES
    assert result.triggered_by is None


def test_unpaired_surrogates_are_normalized_even_without_truncation() -> None:
    content = [
        {"type": "text", "text": "a\ud800b"},
        {"type": "image", "data": "unchanged"},
        {"type": "text", "text": "\ud83d\ude00\udc80"},
    ]

    result = _truncate(content, ToolOutputPolicy())

    assert result.truncated is False
    assert result.invalid_unicode_replacements == 2
    assert result.content == [
        {"type": "text", "text": "a�b"},
        {"type": "image", "data": "unchanged"},
        {"type": "text", "text": "😀�"},
    ]
    assert result.original_chars == 5
    assert result.original_bytes == 12
    assert _visible_text(result.content).encode("utf-8")


def test_head_tail_uses_dual_budgets_across_text_blocks() -> None:
    compact_marker = build_compact_tool_output_marker(
        business_tool_output_id("你" * 40 + "abcdefghijklmnopqrstuvwxyz" * 5)
    )
    content = [
        {"type": "text", "text": "你" * 40},
        {"type": "image", "data": "unchanged"},
        {"type": "text", "text": "abcdefghijklmnopqrstuvwxyz" * 5},
    ]

    result = _truncate(
        content,
        ToolOutputPolicy(
            max_visible_chars=MIN_POLICY_VISIBLE_CHARS + 8,
            max_inline_bytes=MIN_POLICY_INLINE_BYTES + 12,
            tail_ratio=0.25,
        ),
    )

    assert result.triggered_by == "both"
    assert result.content == [
        {"type": "text", "text": "你你你" + compact_marker},
        {"type": "image", "data": "unchanged"},
        {"type": "text", "text": "yz"},
    ]
    assert result.retained_chars == 144
    assert result.retained_bytes == 150
    assert result.omitted_chars == 165
    assert result.omitted_bytes == 239


@pytest.mark.parametrize(
    ("strategy", "expected_payload"),
    [
        ("head", "abcdefgh"),
        ("tail", "stuvwxyz"),
    ],
)
def test_head_or_tail_strategy(
    strategy: OutputStrategy,
    expected_payload: str,
) -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 8
    output_id = business_tool_output_id(text)
    marker = build_compact_tool_output_marker(output_id)
    result = truncate_tool_content(
        [{"type": "text", "text": text}],
        tool_name="test_tool",
        output_id=output_id,
        policy=ToolOutputPolicy(
            max_visible_chars=MIN_POLICY_VISIBLE_CHARS + 8,
            max_inline_bytes=MIN_POLICY_INLINE_BYTES + 8,
            strategy=strategy,
        ),
    )

    expected = (
        expected_payload + marker if strategy == "head" else marker + expected_payload
    )
    assert result.content == [{"type": "text", "text": expected}]


@pytest.mark.parametrize(
    ("text", "policy", "triggered_by"),
    [
        (
            "a" * 160,
            ToolOutputPolicy(max_visible_chars=147, max_inline_bytes=1_000),
            "chars",
        ),
        (
            "你" * 60,
            ToolOutputPolicy(max_visible_chars=1_000, max_inline_bytes=151),
            "bytes",
        ),
        (
            "😀" * 150,
            ToolOutputPolicy(max_visible_chars=145, max_inline_bytes=160),
            "both",
        ),
    ],
)
def test_reports_the_budget_that_triggered_truncation(
    text: str,
    policy: ToolOutputPolicy,
    triggered_by: str,
) -> None:
    result = _truncate([{"type": "text", "text": text}], policy)

    visible_text = _visible_text(result.content)
    assert result.triggered_by == triggered_by
    assert policy.max_visible_chars is None or len(visible_text) <= (
        policy.max_visible_chars
    )
    assert len(visible_text.encode("utf-8")) <= (policy.effective_max_inline_text_bytes)


def test_none_limits_use_bytes_only_and_the_platform_default() -> None:
    bytes_only = ToolOutputPolicy(
        max_visible_chars=None,
        max_inline_bytes=MIN_POLICY_INLINE_BYTES + 1,
    )
    platform_default = ToolOutputPolicy(
        max_visible_chars=MIN_POLICY_VISIBLE_CHARS,
        max_inline_bytes=None,
    )

    result = _truncate([{"type": "text", "text": "你" * 50}], bytes_only)

    assert result.triggered_by == "bytes"
    assert len(_visible_text(result.content).encode("utf-8")) <= (
        MIN_POLICY_INLINE_BYTES + 1
    )
    assert (
        platform_default.effective_max_inline_text_bytes
        == PLATFORM_MAX_INLINE_TEXT_BYTES
    )


def test_full_marker_falls_back_to_compact_marker_at_exact_budget() -> None:
    text = "a" * (MIN_POLICY_INLINE_BYTES + 1)
    output_id = business_tool_output_id(text)
    compact_marker = build_compact_tool_output_marker(output_id)

    result = truncate_tool_content(
        [{"type": "text", "text": text}],
        tool_name="test_tool",
        output_id=output_id,
        policy=ToolOutputPolicy(max_inline_bytes=MIN_POLICY_INLINE_BYTES),
    )

    assert result.content == [{"type": "text", "text": compact_marker}]
    assert result.retained_bytes == MIN_POLICY_INLINE_BYTES
    assert result.omitted_bytes == len(text)


def test_truncation_rejects_an_output_id_for_different_content() -> None:
    with pytest.raises(ValueError, match="does not match"):
        truncate_tool_content(
            [{"type": "text", "text": "a" * (MIN_POLICY_INLINE_BYTES + 1)}],
            tool_name="test_tool",
            output_id=OUTPUT_ID,
            policy=ToolOutputPolicy(max_inline_bytes=MIN_POLICY_INLINE_BYTES),
        )


def test_untruncated_content_still_rejects_an_invalid_output_id() -> None:
    with pytest.raises(ValueError, match="64-character lowercase"):
        truncate_tool_content(
            [{"type": "text", "text": "short"}],
            tool_name="test_tool",
            output_id="not-a-hash",
            policy=ToolOutputPolicy(),
        )


def test_python_retainer_matches_golden_vectors() -> None:
    for case in TOOL_OUTPUT_RETENTION_CASES:
        full_text = _visible_text(case["content"])
        output_id = business_tool_output_id(full_text)
        assert output_id == case["expected"]["output_id"], case["name"]
        result = truncate_tool_content(
            case["content"],
            tool_name=case["tool_name"],
            output_id=output_id,
            policy=ToolOutputPolicy(**case["policy"]),
        )

        assert result.content == case["expected"]["content"], case["name"]
        for field in (
            "original_chars",
            "original_bytes",
            "retained_chars",
            "retained_bytes",
            "omitted_chars",
            "omitted_bytes",
            "original_lines",
            "triggered_by",
            "invalid_unicode_replacements",
        ):
            assert getattr(result, field) == case["expected"][field], case["name"]


def test_save_business_tool_output_uses_hash_path_and_private_permissions(
    tmp_path: Path,
) -> None:
    conversation_dir = tmp_path / "conversation"
    conversation_dir.mkdir()
    files = PiSessionFiles(conversation_dir)

    saved = files.save_business_tool_output("完整输出\n😀")

    expected_hash = business_tool_output_id("完整输出\n😀")
    assert saved.sha256 == expected_hash
    assert saved.relative_path == (
        f"pi/terminal-output/business-tool-{expected_hash}.txt"
    )
    assert saved.path == conversation_dir / saved.relative_path
    assert saved.path.read_text(encoding="utf-8") == "完整输出\n😀"
    assert stat.S_IMODE(files.directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(files.terminal_output_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(saved.path.stat().st_mode) == 0o600


def test_save_business_tool_output_normalizes_unpaired_surrogates(
    tmp_path: Path,
) -> None:
    conversation_dir = tmp_path / "conversation"
    conversation_dir.mkdir()

    saved = PiSessionFiles(conversation_dir).save_business_tool_output("a\ud800b")

    assert saved.path.read_text(encoding="utf-8") == "a�b"
    assert saved.sha256 == business_tool_output_id("a�b")


def test_save_business_tool_output_reuses_matching_content(tmp_path: Path) -> None:
    conversation_dir = tmp_path / "conversation"
    conversation_dir.mkdir()
    files = PiSessionFiles(conversation_dir)

    first = files.save_business_tool_output("same content")
    first_inode = first.path.stat().st_ino
    first.path.chmod(0o644)
    second = files.save_business_tool_output("same content")

    assert second == first
    assert second.path.stat().st_ino == first_inode
    assert stat.S_IMODE(second.path.stat().st_mode) == 0o600


def test_save_business_tool_output_is_safe_for_concurrent_same_content(
    tmp_path: Path,
) -> None:
    conversation_dir = tmp_path / "conversation"
    conversation_dir.mkdir()
    files = PiSessionFiles(conversation_dir)

    with ThreadPoolExecutor(max_workers=8) as executor:
        saved = list(
            executor.map(
                lambda _: files.save_business_tool_output("same content"),
                range(16),
            )
        )

    assert len({item.path for item in saved}) == 1
    assert saved[0].path.read_text(encoding="utf-8") == "same content"
    assert not any(
        path.name.startswith(".") for path in files.terminal_output_directory.iterdir()
    )


def test_save_business_tool_output_rejects_symlinked_directory(
    tmp_path: Path,
) -> None:
    conversation_dir = tmp_path / "conversation"
    pi_directory = conversation_dir / "pi"
    outside = tmp_path / "outside"
    pi_directory.mkdir(parents=True)
    outside.mkdir()
    (pi_directory / "terminal-output").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="must not be a symbolic link"):
        PiSessionFiles(conversation_dir).save_business_tool_output("tool output")

    assert list(outside.iterdir()) == []


def test_save_business_tool_output_rejects_symlinked_destination(
    tmp_path: Path,
) -> None:
    conversation_dir = tmp_path / "conversation"
    terminal_output = conversation_dir / "pi" / "terminal-output"
    terminal_output.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("do not replace", encoding="utf-8")
    output_id = business_tool_output_id("tool output")
    output_path = terminal_output / f"business-tool-{output_id}.txt"
    output_path.symlink_to(outside)

    with pytest.raises(RuntimeError, match="must not be a symbolic link"):
        PiSessionFiles(conversation_dir).save_business_tool_output("tool output")

    assert outside.read_text(encoding="utf-8") == "do not replace"


def test_save_business_tool_output_rejects_non_regular_destination(
    tmp_path: Path,
) -> None:
    conversation_dir = tmp_path / "conversation"
    terminal_output = conversation_dir / "pi" / "terminal-output"
    terminal_output.mkdir(parents=True)
    output_id = business_tool_output_id("tool output")
    (terminal_output / f"business-tool-{output_id}.txt").mkdir()

    with pytest.raises(RuntimeError, match="must be a regular file"):
        PiSessionFiles(conversation_dir).save_business_tool_output("tool output")


def test_save_business_tool_output_rejects_hash_mismatch(tmp_path: Path) -> None:
    conversation_dir = tmp_path / "conversation"
    terminal_output = conversation_dir / "pi" / "terminal-output"
    terminal_output.mkdir(parents=True)
    output_id = business_tool_output_id("expected")
    output_path = terminal_output / f"business-tool-{output_id}.txt"
    output_path.write_text("different", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match its content hash"):
        PiSessionFiles(conversation_dir).save_business_tool_output("expected")


def test_save_business_tool_output_cleans_temporary_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation_dir = tmp_path / "conversation"
    conversation_dir.mkdir()
    files = PiSessionFiles(conversation_dir)

    def fail_replace(source: str, destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(persistence.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        files.save_business_tool_output("tool output")

    assert list(files.terminal_output_directory.iterdir()) == []
