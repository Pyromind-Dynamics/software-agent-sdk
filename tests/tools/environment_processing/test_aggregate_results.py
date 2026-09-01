"""Tests for the platform-side aggregation script (aggregate_results.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


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


# aggregate_results imports the converters as sibling modules at import time
_load_script("convert_to_slime")
_load_script("convert_to_sft")
aggregate_results = _load_script("aggregate_results")


def _manifest_record(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "image": "img:1",
        "workdir": "/home/user",
        "prompt": f"problem for {task_id}",
        "test_sh": "echo reward > /logs/verifier/reward.txt",
    }


def _verdict(
    task_id: str, verdict: str = "usable", reward: float = 1.0, **extra: Any
) -> dict[str, Any]:
    entry: dict[str, Any] = {"task_id": task_id, "verdict": verdict}
    if verdict == "usable":
        entry["reward"] = reward
    entry.update(extra)
    return entry


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


def _shard_run(
    tmp_path: Path,
    batch: str,
    manifest_records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    traces: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Materialize one edp_submit output layout: batch-XXX/<run>/run/..."""
    run_dir = tmp_path / batch / "run-abc"
    (run_dir / "run" / "traces").mkdir(parents=True)
    (tmp_path / batch / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in manifest_records)
    )
    (run_dir / "run" / "verdicts.jsonl").write_text(
        "".join(json.dumps(v) + "\n" for v in verdicts)
    )
    for task_id, events in (traces or {}).items():
        (run_dir / "run" / "traces" / f"{task_id}.pi_trace.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events)
        )
    return str(run_dir)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _two_shards(tmp_path: Path) -> tuple[str, str]:
    shard1 = _shard_run(
        tmp_path,
        "batch-001",
        [_manifest_record("t-1"), _manifest_record("t-2"), _manifest_record("t-3")],
        [
            _verdict("t-1"),
            _verdict("t-2", reward=0.5),
            _verdict("t-3", "error", error_category="verifier_env_missing"),
        ],
        traces={
            "t-1": _pi_trace_events("problem for t-1"),
            "t-2": _pi_trace_events("problem for t-2"),
        },
    )
    shard2 = _shard_run(
        tmp_path,
        "batch-002",
        [_manifest_record("t-4"), _manifest_record("t-1")],
        [_verdict("t-4"), _verdict("t-1")],
        traces={
            "t-4": _pi_trace_events("problem for t-4"),
            "t-1": _pi_trace_events("problem for t-1"),
        },
    )
    return shard1, shard2


def test_aggregate_merges_shards_and_converts(tmp_path: Path) -> None:
    shard1, shard2 = _two_shards(tmp_path)
    out_dir = tmp_path / "out"

    report = aggregate_results.aggregate([shard1, shard2], str(out_dir))

    assert report["total"] == 5
    assert report["processed"] == 4  # t-1 counted once across shards
    assert report["new_processed"] == 4
    assert report["skipped_duplicate"] == 1
    assert report["verdicts"] == {"usable": 3, "error": 1}
    assert report["error_categories"] == {"verifier_env_missing": 1}
    assert report["verifier_env_missing"] == 1  # broken out separately
    assert report["slime_records"] == 3  # every usable record, any reward
    assert report["skipped_low_reward"] == 1
    assert report["sft_samples"] == 2  # only reward >= min_reward

    verdicts = _jsonl(out_dir / "verdicts.jsonl")
    assert [v["task_id"] for v in verdicts] == ["t-1", "t-2", "t-3", "t-4"]
    slime = _jsonl(out_dir / "slime.jsonl")
    assert {r["label"] for r in slime} == {"t-1", "t-2", "t-4"}
    sft = _jsonl(out_dir / "sft.jsonl")
    assert len(sft) == 2
    # the pi trace prompt is the leading user message; the manifest prompt
    # is not prepended again
    assert sft[0]["messages"][0] == {"role": "user", "content": "problem for t-1"}
    progress = json.loads((out_dir / "progress.json").read_text())
    assert progress["total"] == 5
    assert progress["processed"] == 4


def test_aggregate_resume_skips_checkpointed(tmp_path: Path) -> None:
    shard1, shard2 = _two_shards(tmp_path)
    out_dir = tmp_path / "out"
    aggregate_results.aggregate([shard1, shard2], str(out_dir))

    # simulate a torn trailing write, then rerun the same aggregation
    verdicts_path = out_dir / "verdicts.jsonl"
    verdicts_path.write_text(verdicts_path.read_text()[:-1])

    report = aggregate_results.aggregate([shard1, shard2], str(out_dir))

    assert report["new_processed"] == 0
    assert report["processed"] == 4
    # nothing duplicated and the torn line did not glue to a new record
    assert len(_jsonl(verdicts_path)) == 4
    assert len(_jsonl(out_dir / "slime.jsonl")) == 3
    assert len(_jsonl(out_dir / "sft.jsonl")) == 2


def test_aggregate_missing_verdicts_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "batch-001" / "run-x"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="verdicts.jsonl"):
        aggregate_results.aggregate([str(run_dir)], str(tmp_path / "out"))
