import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from pyromind_agent_server.echomind_consolidation import ConsolidationContext
from pyromind_agent_server.echomind_consolidation_replay import (
    _VERIFIER,
    _finish_live_verification,
    compatible_train_reader,
    verify_dataset_replays,
)
from pyromind_runtime.domain.workflow_execution import WorkflowExecutionRequest


def _jsonl(path: Path, data: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in data))


def _message(text: str, images: list[Path] | None = None) -> dict:
    paths = [str(p) for p in images or []]
    return {
        "images": paths,
        "messages": [
            {"role": "user", "content": "<image>" * len(paths) + text},
            {"role": "assistant", "content": "yes"},
        ],
    }


def _run(source: Path, output: Path, *options: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _VERIFIER.read_text(),
            "--source-batch",
            str(source),
            "--output-dir",
            str(output),
            *options,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_image_contract_decodes_media_and_links_to_approved_source(tmp_path: Path):
    source = tmp_path / "batch"
    source.mkdir()
    approved = tmp_path / "approved"
    approved.mkdir()
    output = tmp_path / "run"
    output.mkdir()
    image = source / "one.png"
    Image.new("RGB", (3, 3), "red").save(image)
    (approved / image.name).write_bytes(image.read_bytes())
    _jsonl(output / "train_messages.jsonl", [_message("describe", [image])])
    _jsonl(output / "val_messages.jsonl", [])
    options = ("--approved-source", str(approved), "--read-only")
    result = _run(source, output, *options)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["approved_source_member"] is True
    Image.new("RGB", (3, 3), "blue").save(approved / image.name)
    assert not json.loads(_run(source, output, *options).stdout)[
        "approved_source_member"
    ]
    row = _message("describe", [image])
    row["images"] = []
    _jsonl(output / "train_messages.jsonl", [row])
    assert "placeholders" in _run(source, output, *options).stderr
    (output / "train_messages.jsonl").unlink()
    (output / "train_messages.jsonl").symlink_to(approved / image.name)
    assert "symlinks" in _run(source, output, *options).stderr


def test_read_only_replay_decodes_bounded_media_without_remote_pillow(tmp_path: Path):
    source = tmp_path / "batch"
    source.mkdir()
    output = tmp_path / "run"
    output.mkdir()
    image = source / "sample.png"
    Image.new("RGB", (2, 2), "red").save(image)
    _jsonl(output / "train_messages.jsonl", [_message("sample", [image])])
    other = source / "other.png"
    Image.new("RGB", (2, 2), "green").save(other)
    _jsonl(output / "val_messages.jsonl", [_message("other", [other])])
    options = (
        "--mount-root",
        str(tmp_path),
        "--read-only",
        "--approved-source",
        str(source),
    )
    local = _run(source, output, *options)
    assert local.returncode == 0
    command = [
        sys.executable,
        "-I",
        "-S",
        str(_VERIFIER),
        "--source-batch",
        str(source),
        "--output-dir",
        str(output),
        *options,
        "--defer-image-decode",
        "2",
        "2048",
        "4096",
    ]
    remote = subprocess.run(command, capture_output=True, text=True, check=True)
    result = json.loads(remote.stdout)
    assert len(result["deferred_media"]) == 2
    assert json.loads(_finish_live_verification(remote.stdout)) == json.loads(
        local.stdout
    )
    entry = result["deferred_media"][hashlib.sha256(image.read_bytes()).hexdigest()]
    assert entry["path"] == "batch/sample.png"
    entry["data"] = base64.b64encode(b"invalid").decode()
    with pytest.raises(ValueError, match="摘要"):
        _finish_live_verification(json.dumps(result))
    result["deferred_media"] = {hashlib.sha256(b"invalid").hexdigest(): entry}
    with pytest.raises(ValueError, match="解码"):
        _finish_live_verification(json.dumps(result))
    limited = subprocess.run(
        command[:-2] + ["8", "4096"], capture_output=True, text=True
    )
    assert limited.returncode != 0 and "file limit" in limited.stderr


@pytest.mark.parametrize("change", ["original", "identical", "unrelated", "tampered"])
def test_text_replay_compiles_only_a_verified_pair(tmp_path: Path, change: str):
    conversation = "a" * 32
    project = "b" * 32
    evidence = dict(
        task_id="100",
        revision=1,
        source="workflow_execution_request",
        definition_sha256="sha256:" + "0" * 64,
    )
    approved = tmp_path / "approved.jsonl"
    _jsonl(approved, [{"text": "first"}, {"text": "fixed val"}])
    context = ConsolidationContext.model_validate(
        {
            "project_id": project,
            "source_graph_sha256": "sha256:" + "a" * 64,
            "evidence_sha256": "sha256:" + "1" * 64,
            "evidence": dict(
                conversation_id=conversation,
                dataset=evidence,
                train=evidence,
                eval=evidence,
            ),
            "source_batch": "/workspace/approved.jsonl",
            "base_model": "/workspace/model",
            "validation_dataset": "/workspace/eval.jsonl",
        }
    )
    receipts = []
    snapshots = {}
    live = {}
    assets = []
    for role, raw in (
        ("pipeline", b"pipeline"),
        ("export", b"export"),
        ("verifier", _VERIFIER.read_bytes()),
    ):
        sha = hashlib.sha256(raw).hexdigest()
        asset_path = (
            f"/workspace/.pyromind-agent/{conversation}/data_preparation/"
            f"recipe-assets/{sha}/{role}.py"
        )
        assets.append((role, asset_path, sha))
        snapshots[asset_path] = raw
    for i, text in enumerate(("first", "first" if change == "identical" else "second")):
        source = tmp_path / f"batch{i}.jsonl"
        _jsonl(source, [{"text": text}, {"text": "fixed val"}])
        output = tmp_path / f"run{i}"
        output.mkdir()
        _jsonl(output / "train_messages.jsonl", [_message(text)])
        _jsonl(output / "val_messages.jsonl", [_message("fixed val")])
        run = _run(source, output)
        assert run.returncode == 0, run.stderr
        uri = f"/workspace/.pyromind-agent/{conversation}/data_preparation/run{i}"
        snapshots[uri + "/verification.json"] = (
            output / "verification.json"
        ).read_bytes()
        result = _run(source, output, "--read-only", "--approved-source", str(approved))
        assert result.returncode == 0, result.stderr
        checked = json.loads(result.stdout)
        if change == "tampered":
            checked["train_rows"] += 1
        if change == "unrelated":
            checked["approved_source_member"] = False
        live[uri] = json.dumps(checked).encode()
        receipts.append(
            WorkflowExecutionRequest.model_validate(
                {
                    "conversation_id": conversation,
                    "task_id": str(i + 100),
                    "request_id": str(i),
                    "fingerprint": "fingerprint",
                    "workflow_path": "workflow.py",
                    "summary": {},
                    "workflow": {
                        "nodes": [
                            {
                                "id": "1",
                                "data": {
                                    "nodeType": "CustomCommandCPUNode",
                                    "config": {"command": "echo sample"},
                                },
                            }
                        ],
                        "edges": [],
                    },
                    "dataset_recipe": {
                        "source_path": "/workspace/" + source.name,
                        "output_directory": uri,
                        "command_template": "prepare {source_batch} {output_uri}",
                        "uses_labeling_gateway": False,
                        "assets": [
                            {"role": role, "path": asset_path, "sha256": sha}
                            for role, asset_path, sha in assets
                        ],
                    },
                }
            )
        )
    replay = verify_dataset_replays(
        context, receipts, snapshots.__getitem__, lambda r: live[r.output_directory]
    )
    if change == "original":
        assert (
            replay.runtime and replay.proof and replay.proof.task_ids == ("100", "101")
        )
        assert "source_batch" in str(replay.runtime)
    else:
        assert replay.proof is None and replay.blocker


def test_train_consumer_checks_the_actual_reader():
    nodes = [
        {"id": k, "data": {"nodeType": k, "config": {"messages_field": "messages"}}}
        for k in (
            "DatasetConfigBuilderMessageNode",
            "DatasetConfigBuilderNode",
            "ModelTrainSFTNode",
        )
    ]
    edges = [
        {"source": a["id"], "target": b["id"], "targetHandle": h, "sourceHandle": h}
        for a, b, h in zip(nodes, nodes[1:], ("dataset_kind_config", "dataset_config"))
    ]
    runtime = {"workflow": {"nodes": nodes, "edges": edges}}
    assert compatible_train_reader(runtime)
    nodes[0]["data"]["config"]["messages_field"] = "other_messages"
    assert not compatible_train_reader(runtime)
    nodes[0]["data"]["config"]["messages_field"] = "messages"
    edges.clear()
    assert not compatible_train_reader(runtime)


def test_tracer_directory_reads_media_and_pins_fixed_validation(tmp_path: Path):
    batch = tmp_path / "batch"
    batch.mkdir()
    source = batch / "group.png"
    Image.new("RGB", (2, 2), "green").save(source)
    _jsonl(batch / "train.jsonl", [{"task_id": "group", "images": ["group.png"]}])
    output = tmp_path / "run"
    output.mkdir()
    fixed = tmp_path / "fixed.jsonl"
    _jsonl(fixed, [_message("fixed text")])
    _jsonl(output / "train_messages.jsonl", [_message("question", [source])])
    (output / "val_messages.jsonl").write_bytes(fixed.read_bytes())

    result = _run(batch, output, "--fixed-validation", str(fixed))
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "verification.json").read_bytes())
    assert summary["source_kind"] == "record_directory"
    assert summary["train_rows"] == summary["val_rows"] == 1

    _jsonl(fixed, [_message("changed")])
    assert "Fixed train-time validation has changed" in (
        _run(batch, output, "--fixed-validation", str(fixed)).stderr
    )
