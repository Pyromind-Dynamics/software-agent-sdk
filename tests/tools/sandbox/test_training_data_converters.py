"""Tests for the training-data conversion scripts in the skill's scripts/ dir.

The scripts are standalone CLIs (not a package), so they are loaded via
``importlib`` the same way ``sandbox_runner.py`` is in test_sandbox_runner.py.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


_SKILL_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / ".agents"
    / "skills"
    / "data-processing"
    / "scripts"
    / "edp"
)


def _load_script(name: str) -> ModuleType:
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, _SKILL_SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


slime_converter = _load_script("convert_to_slime")
sft_converter = _load_script("convert_to_sft")


# ---------------------------------------------------------------------------
# convert_to_slime
# ---------------------------------------------------------------------------


def _manifest_record(task_id: str, **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "task_id": task_id,
        "image": "img:1",
        "workdir": "/home/user",
        "prompt": f"problem for {task_id}",
        "test_sh": "echo reward > /logs/verifier/reward.txt",
    }
    record.update(overrides)
    return record


def test_convert_to_slime_keeps_only_usable(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    verdicts = tmp_path / "verdicts.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _manifest_record("t-1"),
                _manifest_record("t-2"),
                _manifest_record("t-3"),
            ]
        )
        + "\n"
    )
    verdicts.write_text(
        "\n".join(
            json.dumps(v)
            for v in [
                {"task_id": "t-1", "verdict": "usable", "reward": 1.0},
                {
                    "task_id": "t-2",
                    "verdict": "error",
                    "error_category": "probe_failed",
                },
            ]
        )
        + "\n"
    )

    exit_code = slime_converter.main(
        [
            "--manifest",
            str(manifest),
            "--verdicts",
            str(verdicts),
            "--out",
            str(tmp_path / "slime.jsonl"),
        ]
    )

    assert exit_code == 0
    lines = (tmp_path / "slime.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["label"] == "t-1"
    assert record["prompt"] == [{"role": "user", "content": "problem for t-1"}]
    assert record["metadata"]["protocol"] == "tmax"
    assert record["metadata"]["instance_id"] == "t-1"
    assert record["metadata"]["workdir"] == "/home/user"
    assert record["metadata"]["test_sh"].startswith("echo reward")


def test_convert_to_slime_skips_usable_without_test_sh(tmp_path: Path) -> None:
    records, skipped = slime_converter.convert(
        [_manifest_record("t-1", test_sh="")],
        [{"task_id": "t-1", "verdict": "usable"}],
        "tmax",
    )

    assert records == []
    assert skipped == 1


# ---------------------------------------------------------------------------
# convert_to_sft
# ---------------------------------------------------------------------------


def _cc_trace_events() -> list[dict[str, Any]]:
    return [
        {"type": "user", "message": {"role": "user", "content": "solve the task"}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "internal reasoning"},
                    {"type": "text", "text": "I'll inspect first."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "Read",
                        "input": {"path": "main.py"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "file list",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": [{"type": "text", "text": "file body"}],
                    },
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
            },
        },
        {"type": "result", "result": "Done."},
    ]


def test_convert_trace_maps_blocks_onto_messages() -> None:
    messages = sft_converter.convert_trace(_cc_trace_events())

    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    first_user = messages[0]
    assert first_user["content"] == "solve the task"
    # Consecutive assistant stream events merge; thinking never surfaces.
    assistant = messages[1]
    assert assistant["content"] == "I'll inspect first."
    assert [call["id"] for call in assistant["tool_calls"]] == ["toolu_1", "toolu_2"]
    assert assistant["tool_calls"][0]["function"]["name"] == "Bash"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "command": "ls"
    }
    assert messages[2]["tool_call_id"] == "toolu_1"
    assert messages[2]["content"] == "file list"
    assert messages[3]["content"] == "file body"
    assert messages[-1]["content"] == "Done."


def _pi_trace_events(prompt: str) -> list[dict[str, Any]]:
    return [
        {"type": "message_end", "message": {"role": "user", "content": prompt}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
            },
        },
    ]


def test_convert_to_sft_filters_by_reward_and_missing_trace(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "t-ok.pi_trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _pi_trace_events("solve the task")) + "\n"
    )
    (traces / "t-low.pi_trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _pi_trace_events("solve the task")) + "\n"
    )
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        "\n".join(
            json.dumps(v)
            for v in [
                {"task_id": "t-ok", "verdict": "usable", "reward": 1.0},
                {"task_id": "t-low", "verdict": "usable", "reward": 0.5},
                {"task_id": "t-gone", "verdict": "usable", "reward": 1.0},
            ]
        )
        + "\n"
    )

    exit_code = sft_converter.main(
        [
            "--traces-dir",
            str(traces),
            "--verdicts",
            str(verdicts),
            "--out",
            str(tmp_path / "sft.jsonl"),
        ]
    )

    assert exit_code == 0
    lines = (tmp_path / "sft.jsonl").read_text().splitlines()
    assert len(lines) == 1
    sample = json.loads(lines[0])
    assert list(sample.keys()) == ["messages"]
    assert sample["messages"][0] == {"role": "user", "content": "solve the task"}
    assert sample["messages"][-1]["role"] == "assistant"


def _cc_trace_without_launch_prompt() -> list[dict[str, Any]]:
    """Realistic CC stream: the task prompt is injected at launch, so the
    trace begins with the agent's first turn, not the problem statement."""
    return [
        {"type": "system", "subtype": "init", "cwd": "/home/user"},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll inspect first."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "file list",
                    }
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
            },
        },
    ]


def _write_sft_inputs(
    tmp_path: Path, *, manifest: bool, system_prompt: str | None = None
) -> list[str]:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "t-1.cc_trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _cc_trace_without_launch_prompt()) + "\n"
    )
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        json.dumps({"task_id": "t-1", "verdict": "usable", "reward": 1.0}) + "\n"
    )
    args = [
        "--traces-dir",
        str(traces),
        "--verdicts",
        str(verdicts),
        "--out",
        str(tmp_path / "sft.jsonl"),
    ]
    if manifest:
        manifest_path = tmp_path / "manifest.jsonl"
        manifest_path.write_text(json.dumps(_manifest_record("t-1")) + "\n")
        args += ["--manifest", str(manifest_path)]
    if system_prompt:
        args += ["--system-prompt", system_prompt]
    assert sft_converter.main(args) == 0
    return (tmp_path / "sft.jsonl").read_text().splitlines()


def test_convert_to_sft_cc_trace_without_prompt_is_not_usable(
    tmp_path: Path,
) -> None:
    """CC traces never contain the problem statement: without --manifest
    the sample has no user prompt and is skipped, not emitted prompt-less."""
    lines = _write_sft_inputs(tmp_path, manifest=False)

    assert lines == []


def test_convert_to_sft_manifest_injects_leading_user(tmp_path: Path) -> None:
    lines = _write_sft_inputs(tmp_path, manifest=True)

    assert len(lines) == 1
    messages = json.loads(lines[0])["messages"]
    assert messages[0] == {"role": "user", "content": "problem for t-1"}
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_convert_to_sft_manifest_and_system_prepend_both(tmp_path: Path) -> None:
    lines = _write_sft_inputs(
        tmp_path, manifest=True, system_prompt="You are a coding assistant."
    )

    assert len(lines) == 1
    messages = json.loads(lines[0])["messages"]
    assert messages[0] == {"role": "system", "content": "You are a coding assistant."}
    assert messages[1] == {"role": "user", "content": "problem for t-1"}
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_convert_to_sft_pi_trace_prompt_not_duplicated(tmp_path: Path) -> None:
    """pi traces carry the launch prompt as their first user message; the
    manifest prompt must not be prepended again."""
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "t-1.pi_trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _pi_trace_events("launch prompt t-1")) + "\n"
    )
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        json.dumps({"task_id": "t-1", "verdict": "usable", "reward": 1.0}) + "\n"
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_manifest_record("t-1")) + "\n")

    assert (
        sft_converter.main(
            [
                "--traces-dir",
                str(traces),
                "--verdicts",
                str(verdicts),
                "--manifest",
                str(manifest),
                "--out",
                str(tmp_path / "sft.jsonl"),
            ]
        )
        == 0
    )
    messages = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])[
        "messages"
    ]
    assert messages[0] == {"role": "user", "content": "launch prompt t-1"}
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_build_sft_messages_requires_terminal_assistant() -> None:
    events = [
        {"type": "message_end", "message": {"role": "user", "content": "do it"}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "working"}],
            },
        },
        {"type": "message_end", "message": {"role": "user", "content": "again"}},
    ]
    assert sft_converter.build_sft_messages(events, None, None) is None


def test_convert_to_sft_missing_manifest_prompt_skips_user(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "t-1.cc_trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in _cc_trace_without_launch_prompt()) + "\n"
    )
    verdicts = tmp_path / "verdicts.jsonl"
    verdicts.write_text(
        json.dumps({"task_id": "t-1", "verdict": "usable", "reward": 1.0}) + "\n"
    )
    # Manifest exists but has no entry for t-1: no prompt -> no usable sample.
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_manifest_record("t-other")) + "\n")
    exit_code = sft_converter.main(
        [
            "--traces-dir",
            str(traces),
            "--verdicts",
            str(verdicts),
            "--manifest",
            str(manifest),
            "--out",
            str(tmp_path / "sft.jsonl"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "sft.jsonl").read_text().splitlines() == []
