from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from PIL import Image

from openhands.tools.data_preparation.runner import (
    runtime_public_names,
    validate_managed_image_pipeline,
)


SKILL = Path(__file__).parents[3] / ".agents/skills/data-processing"
TEMPLATE = SKILL / "references/paradigms/llm-pipeline/pcb_inspection_pipeline.py"
SCRIPTS = SKILL / "scripts/preparation"


def _load(name: str, path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("dataflow")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.delenv("DF_OUTPUT_SCHEMA", raising=False)
    monkeypatch.delenv("DF_STATE_DIR", raising=False)
    monkeypatch.delenv("DF_RESUME", raising=False)
    _load("preparation_runtime", SCRIPTS / "preparation_runtime.py", monkeypatch)
    return _load("image_utils", SCRIPTS / "image_utils.py", monkeypatch)


@pytest.fixture
def config(runtime: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    return _load("pcb_inspection_template", TEMPLATE, monkeypatch).CONFIG


def _sample(tmp_path: Path, sample_id: str) -> dict[str, Any]:
    images = [f"{sample_id}.bmp", f"{sample_id}_cam.bmp"]
    for name in images:
        Image.new("RGB", (320, 160), "white").save(tmp_path / name, format="PNG")
    return {
        "id": sample_id,
        "images": images,
        "image_labels": ["待检原图", "正常结构参考"],
        "inspection_rules": "确定的线路断开需报，无法确认时待复核。",
        "annotation": {"label": False},
        "source_metadata": {"user_prompt": "历史任务：仅做 OCR。"},
    }


def _response(
    label: bool | None, *, bbox: Any = None, category: str | None = None
) -> dict[str, Any]:
    return {
        "label": label,
        "category": category,
        "boxes": [] if bbox is None else [bbox],
        "note": "可见局部边缘内凹，需结合当前公差复核。" if bbox else "未见异常区域。",
    }


def _source_images(tmp_path: Path, sample: dict[str, Any]) -> dict[str, str]:
    """The source-image mapping a run over *sample* writes into its row."""
    return {
        label: str((tmp_path / name).resolve())
        for label, name in zip(sample["image_labels"], sample["images"], strict=True)
    }


def test_pcb_template_satisfies_managed_pipeline_contract() -> None:
    validate_managed_image_pipeline(
        TEMPLATE, runtime_public_names(SCRIPTS / "image_utils.py")
    )


def test_pcb_predictions_round_trip_through_dataflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: Any, config: Any
) -> None:
    samples = [_sample(tmp_path, name) for name in ("true", "false", "uncertain")]
    for index, sample in enumerate(samples):
        sample["id"] = f"3-0803A/10/P6K05774A0/4/内　层/{index}"
        sample["inspection_rules"] = "未提供客户规则，依据图像和领域知识给出预标注。"
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in samples))
    original_source = source.read_bytes()
    responses = [
        _response(True, bbox=[200, 100, 600, 500], category="开路"),
        _response(False),
        _response(None, bbox=[100, 200, 300, 400], category="缺口"),
    ]
    responses[0]["boxes"].append([700, 600, 900, 800])
    responses[2]["note"] = "局部缺口可见，但该位置允许公差未知。"
    serving = Mock()
    serving.generate_from_input_multi_images.return_value = [
        json.dumps(value) for value in responses
    ]
    monkeypatch.setattr(runtime, "_create_vlm_serving", lambda _: serving)
    output = tmp_path / "output" / "processed.jsonl"

    runtime.run_image_pipeline(config, str(source), str(output))

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    for row, sample, response in zip(rows, samples, responses, strict=True):
        assert row == {
            "id": sample["id"],
            "source_images": _source_images(tmp_path, sample),
            **response,
        }
    assert list(rows[0]["source_images"]) == ["待检原图", "正常结构参考"]
    frozen = [
        json.loads(line)
        for line in (output.parent / "source_manifest.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in frozen] == [row["id"] for row in samples]
    assert [row["images"] for row in frozen] == [row["images"] for row in samples]
    assert [row["image_labels"] for row in frozen] == [
        row["image_labels"] for row in samples
    ]
    assert all(row["annotation"] == {"label": False} for row in frozen)
    assert source.read_bytes() == original_source
    call = serving.generate_from_input_multi_images.call_args
    assert call.args[1] == [row["image_labels"] for row in samples]
    assert call.kwargs["json_schema"] == config.response_json_schema
    with Image.open(call.args[0][0][0]) as image:
        assert image.size == (320, 160)
    assert not (output.parent / "failures.jsonl").exists()


@pytest.mark.parametrize(
    "problem", ["missing_image", "missing_roles", "role_count", "duplicate_id"]
)
def test_pcb_invalid_input_never_becomes_a_false_positive_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: Any,
    config: Any,
    problem: str,
) -> None:
    sample = _sample(tmp_path, "sample")
    if problem == "missing_image":
        (tmp_path / sample["images"][0]).unlink()
    elif problem == "missing_roles":
        del sample["image_labels"]
    elif problem == "role_count":
        sample["image_labels"] = ["待检原图"]
    source = tmp_path / "input.jsonl"
    source.write_text(
        (json.dumps(sample) + "\n") * (2 if problem == "duplicate_id" else 1)
    )
    serving = Mock()
    monkeypatch.setattr(runtime, "_create_vlm_serving", lambda _: serving)
    output = tmp_path / "output" / "processed.jsonl"

    with pytest.raises(ValueError):
        runtime.run_image_pipeline(config, str(source), str(output))

    serving.generate_from_input_multi_images.assert_not_called()
    assert not output.exists() or not output.read_text().strip()
    assert (output.parent / "failure.json").is_file()


def test_structured_rejects_model_reserved_fields_and_message_wrappers(
    runtime: Any, config: Any
) -> None:
    response = _response(False)
    with pytest.raises(ValueError, match="reserved source id"):
        runtime._validate_response(json.dumps({"id": "wrong", **response}), config)
    with pytest.raises(ValueError, match="reserved source images"):
        runtime._validate_response(
            json.dumps({"source_images": {"待检原图": "/datasets/a.bmp"}, **response}),
            config,
        )
    for wrapped in (
        f"<answer>{json.dumps(response)}</answer>",
        f"```json\n{json.dumps(response)}\n```",
        json.dumps([]),
    ):
        with pytest.raises(ValueError):
            runtime._validate_response(wrapped, config)
    with pytest.raises(ValueError, match="reserved"):
        replace(
            config,
            response_json_schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
            },
        )
    with pytest.raises(ValueError, match="reserved"):
        replace(
            config,
            response_json_schema={
                "type": "object",
                "properties": {"source_images": {"type": "object"}},
            },
        )
    assert not config.training_system_prompt
    with pytest.raises(ValueError, match="training_system_prompt"):
        replace(config, output_format="vision")


def test_structured_retry_skip_and_resume_preserve_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: Any, config: Any
) -> None:
    samples = [_sample(tmp_path, name) for name in ("retry", "failed", "last")]
    for i, sample in enumerate(samples):
        sample["id"] = f"full/path/内　层/{i}"
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in samples))
    good = _response(False, bbox=[383, 339, 635, 711], category="垃圾")
    serving = Mock()
    serving.generate_from_input_multi_images.side_effect = [
        ["bad json", "bad json", json.dumps(_response(False))],
        [json.dumps(good), "bad json"],
    ]
    monkeypatch.setattr(runtime, "_create_vlm_serving", lambda _: serving)
    output = tmp_path / "output" / "processed.jsonl"
    config = replace(config, max_attempts=2)
    with pytest.raises(runtime.PartialPipelineFailure):
        runtime.run_image_pipeline(config, str(source), str(output))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {
            "id": samples[0]["id"],
            "source_images": _source_images(tmp_path, samples[0]),
            **good,
        },
        {
            "id": samples[2]["id"],
            "source_images": _source_images(tmp_path, samples[2]),
            **_response(False),
        },
    ]
    failures = [
        json.loads(line)
        for line in (output.parent / "failures.jsonl").read_text().splitlines()
    ]
    assert failures[0]["source_id"] == samples[1]["id"]
    assert failures[0]["input"]["images"] == samples[1]["images"]
    calls = serving.generate_from_input_multi_images.call_args_list
    assert len(calls) == 2
    assert len(calls[0].args[0]) == 3 and len(calls[1].args[0]) == 2
    # Checkpoint reuse must not reindex the surviving rows or rerun exhausted records.
    monkeypatch.setenv("DF_RESUME", "1")
    serving.generate_from_input_multi_images.reset_mock()
    with pytest.raises(runtime.PartialPipelineFailure):
        runtime.run_image_pipeline(config, str(source), str(output))
    serving.generate_from_input_multi_images.assert_not_called()
    assert [json.loads(line) for line in output.read_text().splitlines()] == rows


@pytest.mark.parametrize("change", ["mode", "schema", "tool_mode"])
def test_structured_contract_cannot_change_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: Any,
    config: Any,
    change: str,
) -> None:
    sample = _sample(tmp_path, "source")
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(sample) + "\n")
    output = tmp_path / "output" / "processed.jsonl"
    serving = Mock()
    serving.generate_from_input_multi_images.return_value = [
        json.dumps(_response(False))
    ]
    monkeypatch.setattr(runtime, "_create_vlm_serving", lambda _: serving)
    runtime.run_image_pipeline(config, str(source), str(output))
    before = output.read_bytes()
    metadata = json.loads((output.parent / "runtime_metadata.json").read_text())
    assert metadata["output_format"] == "structured"
    assert metadata["response_json_schema"] == config.response_json_schema
    if change == "mode":
        config = replace(config, output_format="vision", training_system_prompt="Train")
    elif change == "schema":
        config = replace(
            config, response_json_schema={"type": "object", "required": ["result"]}
        )
    else:
        monkeypatch.setenv("DF_OUTPUT_SCHEMA", "vision")
    monkeypatch.setenv("DF_RESUME", "1")
    serving.generate_from_input_multi_images.reset_mock()
    with pytest.raises(ValueError, match="different|does not match"):
        runtime.run_image_pipeline(config, str(source), str(output))
    serving.generate_from_input_multi_images.assert_not_called()
    assert output.read_bytes() == before


def test_structured_preserves_task_specific_objects_and_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: Any, config: Any
) -> None:
    samples = [_sample(tmp_path, name) for name in ("a", "b", "c")]
    for sample, record_id in zip(
        samples, ("00001", "00002", " /中文　路径/3 "), strict=True
    ):
        sample["id"] = record_id
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in samples))
    # Different nesting/field names, no boxes field, and optional root fields.
    responses = [
        {
            "regions": [
                {"kind": "垃圾", "polygon": [[1, 2], [3, 4], [5, 2]]},
                {"kind": "氧化", "description": "局部颜色变化"},
            ],
            "score": 1,
        },
        {"regions": [], "description": "未见异常"},
        {"status": "review", "description": "无法定位"},
    ]
    serving = Mock()
    serving.generate_from_input_multi_images.side_effect = [
        [json.dumps(row) for row in responses[:2]],
        KeyboardInterrupt,
    ]
    monkeypatch.setattr(runtime, "_create_vlm_serving", lambda _: serving)
    config = replace(config, batch_size=2)
    output = tmp_path / "output" / "processed.jsonl"
    with pytest.raises(KeyboardInterrupt):
        runtime.run_image_pipeline(config, str(source), str(output))
    assert len(output.read_text().splitlines()) == 2
    monkeypatch.setenv("DF_RESUME", "1")
    serving.generate_from_input_multi_images.side_effect = [[json.dumps(responses[2])]]
    runtime.run_image_pipeline(config, str(source), str(output))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [
        {
            "id": sample["id"],
            "source_images": _source_images(tmp_path, sample),
            **response,
        }
        for sample, response in zip(samples, responses, strict=True)
    ]


def test_source_images_keep_repeated_roles_and_strip_the_pod_mount(
    runtime: Any,
) -> None:
    """Two images sharing a role both survive, and pod paths become Storage paths."""
    images = runtime._source_images(
        ["待检原图", "辅助图", "辅助图"],
        [
            "/target-workspace/datasets/pcb/board.bmp",
            "/target-workspace/datasets/pcb/diff.bmp",
            "/target-workspace/datasets/pcb/zoom.bmp",
        ],
    )
    assert images == {
        "待检原图": "/datasets/pcb/board.bmp",
        "辅助图": "/datasets/pcb/diff.bmp",
        "辅助图#2": "/datasets/pcb/zoom.bmp",
    }
    # A local sample run has no mount, so its own path is what it records.
    assert runtime._storage_object_path("/tmp/ws/board.bmp") == "/tmp/ws/board.bmp"
