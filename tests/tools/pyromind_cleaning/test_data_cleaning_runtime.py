from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


_ROOT = Path(__file__).parents[3]
_SCRIPTS = _ROOT / ".agents" / "skills" / "data-cleaning" / "scripts"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime_modules() -> tuple[ModuleType, ModuleType]:
    utils = _load_module("cleaning_utils", _SCRIPTS / "cleaning_utils.py")
    validator = _load_module(
        "data_cleaning_validate_format",
        _SCRIPTS / "validate_format.py",
    )
    return utils, validator


def _message_record(value: Any) -> dict[str, Any]:
    text = str(value["text"])
    return {
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"answer:{text}"},
        ]
    }


def test_iter_records_supports_stable_directories_and_input_formats(
    runtime_modules,
    tmp_path,
):
    utils, _ = runtime_modules
    source = tmp_path / "source"
    source.mkdir()
    (source / "b.tsv").write_text("text\nsecond\n", encoding="utf-8")
    (source / "a.csv").write_text("text\nfirst\n", encoding="utf-8")
    (source / "c.txt").write_text("third\n", encoding="utf-8")

    records = list(utils.iter_records(source))

    assert [record.data["text"] for record in records] == [
        "first",
        "second",
        "third",
    ]
    assert [Path(record.source_path).name for record in records] == [
        "a.csv",
        "b.tsv",
        "c.txt",
    ]


def test_iter_json_streams_array_across_read_boundaries(runtime_modules, tmp_path):
    utils, _ = runtime_modules
    source = tmp_path / "large.json"
    source.write_text(
        json.dumps([{"text": "x" * 70000}, {"text": "done"}]),
        encoding="utf-8",
    )

    records = list(utils.iter_records(source))

    assert len(records) == 2
    assert records[0].data["text"] == "x" * 70000
    assert records[1].data == {"text": "done"}


def test_iter_records_sniffs_json_in_text_and_unwraps_huggingface_rows(
    runtime_modules,
    tmp_path,
):
    utils, _ = runtime_modules
    source = tmp_path / "data.txt"
    source.write_text(
        json.dumps(
            {
                "rows": [
                    {"row_idx": 4, "row": {"text": "one"}, "truncated_cells": []},
                    {"row_idx": 8, "row": {"text": "two"}, "truncated_cells": []},
                ]
            }
        ),
        encoding="utf-8",
    )

    records = list(utils.iter_records(source))

    assert [record.data for record in records] == [{"text": "one"}, {"text": "two"}]
    assert [record.line_number for record in records] == [5, 9]

    run_dir = tmp_path / "run"
    stats = utils.run_cleaning(
        input_path=source,
        output_path=run_dir / "output.jsonl",
        state_dir=run_dir,
        mapper=_message_record,
        limit=1,
    )
    assert stats.read == 1
    assert stats.written == 1


def test_iter_records_sniffs_jsonl_in_text_but_preserves_plain_text(
    runtime_modules,
    tmp_path,
):
    utils, _ = runtime_modules
    json_text = tmp_path / "rows.txt"
    json_text.write_text('{"text":"one"}\n{"text":"two"}\n', encoding="utf-8")
    plain_text = tmp_path / "plain.txt"
    plain_text.write_text("first\nsecond\n", encoding="utf-8")
    brace_text = tmp_path / "brace.txt"
    brace_text.write_text("{not JSON, just text}\nsecond\n", encoding="utf-8")

    assert [item.data for item in utils.iter_records(json_text)] == [
        {"text": "one"},
        {"text": "two"},
    ]
    assert [item.data for item in utils.iter_records(plain_text)] == [
        {"text": "first"},
        {"text": "second"},
    ]
    assert [item.data for item in utils.iter_records(brace_text)] == [
        {"text": "{not JSON, just text}"},
        {"text": "second"},
    ]


def test_runner_limit_counts_source_rows_and_isolates_errors(
    runtime_modules,
    tmp_path,
):
    utils, _ = runtime_modules
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"text": "one"}),
                "{bad json}",
                json.dumps({"text": "two"}),
                json.dumps({"text": "three"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "run" / "output.jsonl"

    stats = utils.run_cleaning(
        input_path=source,
        output_path=output,
        state_dir=output.parent,
        mapper=_message_record,
        limit=3,
    )

    assert stats.to_dict() | {} == {
        "read": 3,
        "written": 2,
        "dropped": 1,
        "errors": 1,
        "duplicates": 0,
        "drop_reasons": {"json_decode_error": 1},
        "error_samples": [
            {
                "reason": "json_decode_error",
                "line_number": 2,
                "error": "could not parse JSON",
                "sample": "{bad json}",
            }
        ],
        "status": "completed",
    }
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    errors = (output.parent / "errors.jsonl").read_text(encoding="utf-8")
    assert '"source_index":2' in errors


def test_runner_redacts_sensitive_error_samples(runtime_modules, tmp_path):
    utils, _ = runtime_modules
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"text": "one", "api_key": "private-value"}) + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    def invalid_mapper(value):
        raise ValueError(f"bad row with Bearer {value['api_key']}")

    stats = utils.run_cleaning(
        input_path=source,
        output_path=run_dir / "output.jsonl",
        state_dir=run_dir,
        mapper=invalid_mapper,
    )

    errors = (run_dir / "errors.jsonl").read_text(encoding="utf-8")
    assert stats.errors == 1
    assert "private-value" not in errors
    assert "***REDACTED***" in errors


def test_runner_writes_failed_stats_and_checkpoint_for_structural_input_error(
    runtime_modules,
    tmp_path,
):
    utils, _ = runtime_modules
    run_dir = tmp_path / "run"

    with pytest.raises(FileNotFoundError):
        utils.run_cleaning(
            input_path=tmp_path / "missing.jsonl",
            output_path=run_dir / "output.jsonl",
            state_dir=run_dir,
            mapper=_message_record,
        )

    stats = json.loads((run_dir / "stats.json").read_text())
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
    assert stats["status"] == "failed"
    assert stats["read"] == 0
    assert checkpoint["source_records_committed"] == 0
    assert (run_dir / "errors.jsonl").is_file()


def test_runner_resume_truncates_uncommitted_tails_without_duplicates(
    runtime_modules,
    tmp_path,
):
    utils, _ = runtime_modules
    source = tmp_path / "source.jsonl"
    rows = ["a", "b", "b", "c", "d"]
    source.write_text(
        "".join(json.dumps({"text": value}) + "\n" for value in rows),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    output = run_dir / "output.jsonl"

    def interrupted_mapper(value):
        if value["text"] == "c":
            raise RuntimeError("simulated process interruption")
        return _message_record(value)

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        utils.run_cleaning(
            input_path=source,
            output_path=output,
            state_dir=run_dir,
            mapper=interrupted_mapper,
        )

    failed_stats = json.loads((run_dir / "stats.json").read_text())
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
    assert failed_stats["status"] == "failed"
    assert checkpoint["source"] == {
        "index": 3,
        "path": str(source),
        "line": 3,
    }

    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_message_record({"text": "uncommitted"})) + "\n")
    with (run_dir / "errors.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"error":"uncommitted"}\n')

    stats = utils.run_cleaning(
        input_path=source,
        output_path=output,
        state_dir=run_dir,
        mapper=_message_record,
        resume=True,
    )

    output_rows = [json.loads(line) for line in output.read_text().splitlines()]
    user_values = [row["messages"][0]["content"] for row in output_rows]
    assert user_values == ["a", "b", "c", "d"]
    assert stats.read == 5
    assert stats.written == 4
    assert stats.duplicates == 1
    assert "uncommitted" not in (run_dir / "errors.jsonl").read_text()


def test_messages_validator_enforces_top_level_and_tool_links(runtime_modules):
    utils, _ = runtime_modules
    valid = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"q": "go"}},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "lookup",
                "tool_call_id": "call_1",
                "content": "result",
            },
            {"role": "assistant", "content": "done"},
        ]
    }
    assert utils.validate_messages_record(valid) == []

    invalid = {**valid, "metadata": {"source": "private"}}
    invalid["messages"] = [
        *valid["messages"][:2],
        {
            "role": "tool",
            "name": "lookup",
            "tool_call_id": "missing",
            "content": "result",
        },
    ]
    errors = utils.validate_messages_record(invalid)
    assert {error.code for error in errors} == {"invalid_format"}
    assert any("top-level fields" in error.message for error in errors)
    assert any("does not match" in error.message for error in errors)


def test_messages_validator_requires_user_but_allows_grpo_prompt(runtime_modules):
    utils, _ = runtime_modules

    assert (
        utils.validate_messages_record(
            {"messages": [{"role": "user", "content": "solve this"}]}
        )
        == []
    )
    errors = utils.validate_messages_record(
        {"messages": [{"role": "system", "content": "be concise"}]}
    )
    assert any("at least one user" in error.message for error in errors)


def test_dpo_is_detected_and_preserved_as_preference(runtime_modules):
    utils, _ = runtime_modules
    source = {
        "instruction": "Calculate the result.",
        "input": "2+2?",
        "chosen": "4",
        "rejected": "5",
        "metadata": {"source": "demo"},
    }

    assert utils.detect_training_format(source) == "preference"
    cleaned = utils.to_training_record(source)
    assert cleaned == {
        "prompt": "Calculate the result.\n\n2+2?",
        "chosen": "4",
        "rejected": "5",
    }
    assert utils.validate_preference_record(cleaned) == []
    assert utils.validate_record(cleaned) == []

    with pytest.raises(utils.DataCleaningError, match="must be a string"):
        utils.to_preference_record(
            {"prompt": "question", "chosen": [{"content": "yes"}], "rejected": "no"}
        )


def test_runner_writes_dpo_preference_output(runtime_modules, tmp_path):
    utils, _ = runtime_modules
    source = tmp_path / "dpo.jsonl"
    source.write_text(
        json.dumps({"prompt": "question", "chosen": "yes", "rejected": "no"}) + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    stats = utils.run_cleaning(
        input_path=source,
        output_path=run_dir / "output.jsonl",
        state_dir=run_dir,
        mapper=utils.to_training_record,
    )

    assert stats.written == 1
    assert json.loads((run_dir / "output.jsonl").read_text()) == {
        "prompt": "question",
        "chosen": "yes",
        "rejected": "no",
    }


def test_validator_rejects_mixed_output_formats(runtime_modules, tmp_path):
    _, validator = runtime_modules
    output = tmp_path / "output.jsonl"
    output.write_text(
        json.dumps({"messages": [{"role": "user", "content": "prompt"}]})
        + "\n"
        + json.dumps({"prompt": "p", "chosen": "c", "rejected": "r"})
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "validation.json"

    assert validator.main(["--input", str(output), "--report", str(report_path)]) == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["format"] == "mixed"
    assert report["reasons"] == {"invalid_format": 1}


def test_validator_reports_preference_format(runtime_modules, tmp_path):
    _, validator = runtime_modules
    output = tmp_path / "output.jsonl"
    output.write_text(
        json.dumps({"prompt": "p", "chosen": "c", "rejected": "r"}) + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "validation.json"

    assert validator.main(["--input", str(output), "--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["format"] == "preference"
    assert report["valid"] == 1
    assert report["role_counts"] == {}


@pytest.mark.parametrize(
    ("record", "expected_code"),
    [
        ({"messages": [{"role": "human", "content": "hello"}]}, "invalid_role"),
        ({"messages": [{"role": "user", "content": ""}]}, "empty_string"),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "lookup", "arguments": {}},
                            }
                        ],
                    }
                ]
            },
            "empty_string",
        ),
    ],
)
def test_messages_validator_rejects_invalid_roles_content_and_calls(
    runtime_modules,
    record,
    expected_code,
):
    utils, _ = runtime_modules
    assert expected_code in {
        error.code for error in utils.validate_messages_record(record)
    }


def test_messages_conversion_never_infers_a_preference_branch(runtime_modules):
    utils, _ = runtime_modules
    preference_row = {
        "chosen": [{"role": "assistant", "content": "yes"}],
        "rejected": [{"role": "assistant", "content": "no"}],
    }

    with pytest.raises(utils.DataCleaningError, match="explicitly available"):
        utils.to_messages_record(preference_row)

    assert utils.to_messages_record(preference_row, preference="chosen") == {
        "messages": [{"role": "assistant", "content": "yes"}]
    }


def test_explicit_text_preference_to_sft_preserves_prompt_and_selected_answer(
    runtime_modules,
):
    utils, _ = runtime_modules
    preference_row = {
        "system": "Answer accurately.",
        "instruction": "Calculate the result.",
        "input": "2+2?",
        "chosen": "4",
        "rejected": "5",
    }

    assert utils.to_messages_record(preference_row, preference="chosen") == {
        "messages": [
            {"role": "system", "content": "Answer accurately."},
            {"role": "user", "content": "Calculate the result.\n\n2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }


def test_validator_always_writes_report_and_rejects_empty_output(
    runtime_modules,
    tmp_path,
):
    _, validator = runtime_modules
    output = tmp_path / "output.jsonl"
    output.write_text("", encoding="utf-8")
    report_path = tmp_path / "validation.json"

    exit_code = validator.main(["--input", str(output), "--report", str(report_path)])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["format"] == "unknown"
    assert report["reasons"] == {"empty_dataset": 1}


def test_skill_requires_platform_sample_and_user_confirmation():
    skill = (_ROOT / ".agents" / "skills" / "data-cleaning" / "SKILL.md").read_text(
        encoding="utf-8"
    )

    assert "limit=3" in skill
    assert "等待用户明确确认" in skill
    assert "不得下载数据或本地运行" in skill
    assert "平台产物只能用 `preview_dataset` 查看" in skill
    assert "不要查看、搜索或反复读取 `cleaning_utils.py`" in skill
    assert "prompt/chosen/rejected 自动按 DPO 清洗" in skill
    assert "不透明 `error_log`" in skill
    assert "`read_file`" not in skill
    assert "`execute_bash`" not in skill
    assert "approved_run_id" not in skill

    routing = (
        _ROOT
        / ".agents"
        / "skills"
        / "generate-workflow-dsl"
        / "references"
        / "data-routing.md"
    ).read_text(encoding="utf-8")
    assert "user_prompt_field=prompt" in routing
    assert "assistant_response_field=chosen" in routing
    assert "rejected_field=rejected" in routing


def test_example_cleaner_only_imports_public_utils(runtime_modules):
    utils, _ = runtime_modules
    example = (
        _ROOT
        / ".agents"
        / "skills"
        / "data-cleaning"
        / "references"
        / "example_clean_script.py"
    )
    tree = ast.parse(example.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "cleaning_utils"
        for alias in node.names
    }
    assert imported <= set(utils.__all__)
