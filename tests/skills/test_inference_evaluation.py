from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import threading
import zlib
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest


MODULE_PATH = (
    Path(__file__).parents[2]
    / ".agents"
    / "skills"
    / "inference-evaluation"
    / "scripts"
    / "evaluate_inference.py"
)


@pytest.fixture
def evaluation_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_inference", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inference_server() -> Iterator[tuple[str, dict[str, Any]]]:
    state: dict[str, Any] = {"reply": "hello", "replies": [], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            state["requests"].append(payload)
            reply = state["replies"].pop(0) if state["replies"] else state["reply"]
            body = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def evaluation_inputs(tmp_path: Path) -> dict[str, Any]:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps({"id": "a", "prompt": "say hello", "gt": "hello"}) + "\n"
    )
    return {
        "endpoint": "http://unused",
        "model": "default",
        "model_reference": "/models/v1",
        "dataset_path": str(dataset),
        "output_dir": str(tmp_path / "out"),
        "dataset_config_json": json.dumps(
            {"user_prompt_field": "prompt", "reference_field": "gt"}
        ),
        "evaluation_config_json": json.dumps(
            {
                "mode": "agent_rubric",
                "rubrics": [
                    {
                        "name": "correct",
                        "criterion": "Matches GT",
                        "weight": 1,
                        "required": True,
                        "evaluator": {"type": "exact_match"},
                    }
                ],
            }
        ),
        "max_retries": 0,
        "run_id": "run-test-1",
    }


def test_http_report_and_identical_resume(
    evaluation_module, evaluation_inputs, inference_server
):
    endpoint, state = inference_server
    evaluation_inputs["endpoint"] = endpoint
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["metrics"]["pass_rate"] == 1
    assert result["metrics"]["request_count"] == 1
    output = Path(evaluation_inputs["output_dir"])
    # All cases passed, so the badcase-only report renders an empty case list.
    assert (
        "Inference evaluation report" in (output / "evaluation_report.html").read_text()
    )
    assert json.loads((output / "progress.json").read_text())["succeeded"] == 1
    resumed = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert len(state["requests"]) == 1
    assert resumed["metrics"]["current_request_count"] == 0
    assert resumed["metrics"]["reused_predictions"] == 1
    assert len((output / "processed.jsonl").read_text().splitlines()) == 1


def test_zero_business_score_is_successful_execution(
    evaluation_module, evaluation_inputs, inference_server
):
    endpoint, state = inference_server
    state["reply"] = "wrong"
    evaluation_inputs["endpoint"] = endpoint
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["status"] == "succeeded"
    assert result["metrics"]["pass_rate"] == 0
    resumed = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert resumed["metrics"]["reused_predictions"] == 1
    assert len(state["requests"]) == 1


def test_resume_only_retries_missing_prediction(
    evaluation_module, evaluation_inputs, monkeypatch
):
    dataset = Path(evaluation_inputs["dataset_path"])
    dataset.write_text(
        dataset.read_text()
        + json.dumps({"id": "b", "prompt": "fail", "gt": "hello"})
        + "\n"
    )

    def call(endpoint, payload, api_key, timeout):
        if payload["messages"][-1]["content"][-1]["text"] == "fail":
            raise OSError("offline")
        return {"choices": [{"message": {"content": "hello"}}]}, 0.1

    monkeypatch.setattr(evaluation_module, "_call_api", call)
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["metrics"]["evaluated_cases"] == 1
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        lambda *a: ({"choices": [{"message": {"content": "hello"}}]}, 0.1),
    )
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["metrics"]["request_count"] == 3
    assert result["metrics"]["current_request_count"] == 1
    assert result["metrics"]["reused_predictions"] == 1
    assert result["metrics"]["evaluated_cases"] == 2


def test_retries_are_counted(evaluation_module, evaluation_inputs, monkeypatch):
    calls = 0

    def call(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary")
        return {"choices": [{"message": {"content": "hello"}}]}, 0.1

    monkeypatch.setattr(evaluation_module, "_call_api", call)
    monkeypatch.setattr(evaluation_module.time, "sleep", lambda _: None)
    result = evaluation_module.evaluate_inference(
        **{**evaluation_inputs, "max_retries": 1}
    )
    assert result["metrics"]["request_count"] == 2


@pytest.mark.parametrize(
    "change", ["dataset", "generation", "rubric", "media", "run_id"]
)
def test_changed_inputs_cannot_reuse_predictions(
    evaluation_module, evaluation_inputs, monkeypatch, change
):
    dataset = Path(evaluation_inputs["dataset_path"])
    media = dataset.parent / "sample.png"
    media.write_bytes(b"before")
    row = json.loads(dataset.read_text())
    row["image"] = "sample.png"
    dataset.write_text(json.dumps(row) + "\n")
    evaluation_inputs["dataset_config_json"] = json.dumps(
        {"user_prompt_field": "prompt", "reference_field": "gt", "media_field": "image"}
    )
    call = Mock(return_value=({"choices": [{"message": {"content": "hello"}}]}, 0.1))
    monkeypatch.setattr(evaluation_module, "_call_api", call)
    evaluation_module.evaluate_inference(**evaluation_inputs)
    if change == "dataset":
        dataset.write_text(dataset.read_text().replace("say hello", "new prompt"))
    elif change == "generation":
        evaluation_inputs["max_tokens"] = 10
    elif change == "media":
        media.write_bytes(b"after")
    elif change == "run_id":
        evaluation_inputs["run_id"] = "run-other"
    else:
        config = json.loads(evaluation_inputs["evaluation_config_json"])
        config["rubrics"][0]["weight"] = 2
        evaluation_inputs["evaluation_config_json"] = json.dumps(config)
    with pytest.raises(ValueError, match="changed"):
        evaluation_module.evaluate_inference(**evaluation_inputs)
    assert call.call_count == 1


def test_duplicate_ids_are_rejected_before_request(
    evaluation_module, evaluation_inputs, monkeypatch
):
    dataset = Path(evaluation_inputs["dataset_path"])
    dataset.write_text(dataset.read_text() * 2)
    call = Mock()
    monkeypatch.setattr(evaluation_module, "_call_api", call)
    with pytest.raises(ValueError, match="unique"):
        evaluation_module.evaluate_inference(**evaluation_inputs)
    call.assert_not_called()


def test_empty_prediction_cannot_generate_success(
    evaluation_module, evaluation_inputs, monkeypatch
):
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        lambda *a: ({"choices": [{"message": {"content": ""}}]}, 0.1),
    )
    with pytest.raises(RuntimeError, match="no successfully scored"):
        evaluation_module.evaluate_inference(**evaluation_inputs)
    report = json.loads(
        (Path(evaluation_inputs["output_dir"]) / "report.json").read_text()
    )
    assert report["status"] == "failed"
    assert report["metrics"]["evaluated_cases"] == 0


def test_required_rubric_and_bbox(evaluation_module):
    config = {
        "rubrics": [
            {
                "name": "format",
                "criterion": "object",
                "weight": 100,
                "evaluator": {"type": "json_valid"},
            },
            {
                "name": "label",
                "criterion": "correct",
                "weight": 1,
                "required": True,
                "evaluator": {
                    "type": "field_equals",
                    "prediction_path": "label",
                    "reference_path": "label",
                },
            },
        ]
    }
    result = evaluation_module._agent_rubric_evaluation(
        {"label": "A"}, '{"label":"B"}', config
    )
    assert result["overall_score"] > 0.9
    assert result["passed"] is False
    assert evaluation_module._bbox_score(
        [[0, 0, 10, 10], [20, 20, 30, 30]], [[0, 0, 10, 10]], 0.5
    ) == pytest.approx(2 / 3)


def test_scoring_error_reuses_prediction(
    evaluation_module, evaluation_inputs, monkeypatch
):
    call = Mock(return_value=({"choices": [{"message": {"content": "hello"}}]}, 0.1))
    monkeypatch.setattr(evaluation_module, "_call_api", call)
    score = evaluation_module._agent_rubric_evaluation
    monkeypatch.setattr(
        evaluation_module,
        "_agent_rubric_evaluation",
        Mock(side_effect=ValueError("score error")),
    )
    with pytest.raises(RuntimeError):
        evaluation_module.evaluate_inference(**evaluation_inputs)
    monkeypatch.setattr(evaluation_module, "_agent_rubric_evaluation", score)
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["metrics"]["evaluated_cases"] == 1
    assert call.call_count == 1


def test_manifest_binds_run_id_and_evaluation_config(
    evaluation_module, evaluation_inputs, monkeypatch
):
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        Mock(return_value=({"choices": [{"message": {"content": "hello"}}]}, 0.1)),
    )
    evaluation_module.evaluate_inference(**evaluation_inputs)
    manifest = json.loads(
        (Path(evaluation_inputs["output_dir"]) / "run_manifest.json").read_text()
    )
    assert manifest["run_id"] == "run-test-1"
    evaluation = json.loads(evaluation_inputs["evaluation_config_json"])
    digest = hashlib.sha256(
        json.dumps(
            evaluation, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["evaluation_config_sha256"] == digest
    assert manifest["runtime_sha256"] == evaluation_module._runtime_sha256()


def test_report_spec_auto_selects_media_template(
    evaluation_module, evaluation_inputs, monkeypatch
):
    dataset = Path(evaluation_inputs["dataset_path"])
    media = dataset.parent / "sample.png"
    media.write_bytes(_png_bytes(64, 48))
    row = json.loads(dataset.read_text())
    row["image"] = "sample.png"
    dataset.write_text(json.dumps(row) + "\n")
    evaluation_inputs["dataset_config_json"] = json.dumps(
        {
            "user_prompt_field": "prompt",
            "reference_field": "gt",
            "media_field": "image",
        }
    )
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        Mock(return_value=({"choices": [{"message": {"content": "goodbye"}}]}, 0.1)),
    )
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    output = Path(evaluation_inputs["output_dir"])
    assert result["report_spec"]["template_id"] == "evaluation-media-v1"
    assert result["warnings"] == []
    html = (output / "evaluation_report.html").read_text()
    assert "data:image/png;base64," not in html
    report_media = _report_cases(output)[0]["media"][0]
    assert report_media["media_id"] == "00000000-000.png"
    assert (output / "evaluation_media" / report_media["media_id"]).read_bytes() == (
        media.read_bytes()
    )
    spec = json.loads((output / "resolved_report_spec.json").read_text())
    assert "media_compare" in spec["enabled_panels"]


def test_report_spec_basic_template_omits_media_urls(
    evaluation_module, evaluation_inputs, monkeypatch
):
    dataset = Path(evaluation_inputs["dataset_path"])
    media = dataset.parent / "sample.png"
    media.write_bytes(_png_bytes(64, 48))
    row = json.loads(dataset.read_text())
    row["image"] = "sample.png"
    dataset.write_text(json.dumps(row) + "\n")
    evaluation_inputs["dataset_config_json"] = json.dumps(
        {
            "user_prompt_field": "prompt",
            "reference_field": "gt",
            "media_field": "image",
        }
    )
    evaluation_inputs["report_spec_json"] = json.dumps(
        {"template_id": "evaluation-basic-v1"}
    )
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        Mock(return_value=({"choices": [{"message": {"content": "goodbye"}}]}, 0.1)),
    )
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    output = Path(evaluation_inputs["output_dir"])
    html = (output / "evaluation_report.html").read_text()
    assert result["warnings"] == []
    assert "data:image/png;base64," not in html
    assert _report_cases(output)[0]["media"] == [{"caption": "sample.png"}]


def test_report_spec_violations_fall_back_to_basic(
    evaluation_module, evaluation_inputs, monkeypatch
):
    dataset = Path(evaluation_inputs["dataset_path"])
    media = dataset.parent / "sample.png"
    media.write_bytes(_png_bytes(64, 48))
    row = json.loads(dataset.read_text())
    row["image"] = "sample.png"
    dataset.write_text(json.dumps(row) + "\n")
    evaluation_inputs["dataset_config_json"] = json.dumps(
        {
            "user_prompt_field": "prompt",
            "reference_field": "gt",
            "media_field": "image",
        }
    )
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        Mock(return_value=({"choices": [{"message": {"content": "goodbye"}}]}, 0.1)),
    )
    # Media is present, so the capability-derived default would be media-v1.
    # A present-but-invalid spec must still fall back to basic (never media),
    # and never auto-enable the media_compare panel / embed base64 originals.
    invalid_specs = (
        '{"template_id": "custom-html-v1"}',
        '{"custom_html": "<script>"}',
        '{"template_id": "evaluation-basic-v1", "template_version": "9.9.9"}',
        '{"template_id": "evaluation-basic-v1", "enabled_panels": ["chart"]}',
        '{"template_id": "evaluation-basic-v1", "case_order": "dataset"}',
        '{"template_id": "evaluation-basic-v1", "review_scope": "all"}',
        '{"template_id": 123}',
        "{}",
        "not json",
    )
    for spec_json in invalid_specs:
        result = evaluation_module.evaluate_inference(
            **{**evaluation_inputs, "report_spec_json": spec_json}
        )
        assert result["report_spec"]["template_id"] == "evaluation-basic-v1"
        assert result["warnings"]
        html = (
            Path(evaluation_inputs["output_dir"]) / "evaluation_report.html"
        ).read_text()
        assert "data:image/png;base64," not in html
    result = evaluation_module.evaluate_inference(
        **{
            **evaluation_inputs,
            "report_spec_json": json.dumps(
                {
                    "template_id": "evaluation-basic-v1",
                    "title": "PCB 评测",
                    "annotation_schema": {
                        "taxonomy_labels": ["small_object"],
                        "dispositions": ["fix_ground_truth"],
                    },
                }
            ),
        }
    )
    output = Path(evaluation_inputs["output_dir"])
    spec = json.loads((output / "resolved_report_spec.json").read_text())
    assert spec["title"] == "PCB 评测"
    assert spec["annotation_schema"]["taxonomy_labels"] == ["small_object"]
    assert result["warnings"] == []
    assert "PCB 评测" in (output / "evaluation_report.html").read_text()


def test_evaluation_status_validated_when_reference_extractable(
    evaluation_module, evaluation_inputs, monkeypatch
):
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        Mock(return_value=({"choices": [{"message": {"content": "hello"}}]}, 0.1)),
    )
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    output = Path(evaluation_inputs["output_dir"])
    assert result["evaluation_status"] == "validated"
    assert result["validation"]["case_errors"] == {}
    assert result["validation"]["report_errors"] == []
    report = json.loads((output / "report.json").read_text())
    assert report["evaluation_status"] == "validated"


def test_evaluation_status_invalid_on_unresolvable_reference_path(
    evaluation_module, evaluation_inputs, monkeypatch
):
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        Mock(return_value=({"choices": [{"message": {"content": "hello"}}]}, 0.1)),
    )
    evaluation_inputs["evaluation_config_json"] = json.dumps(
        {
            "mode": "agent_rubric",
            "rubrics": [
                {
                    "name": "verdict",
                    "criterion": "verdict equals",
                    "weight": 1,
                    "required": True,
                    "evaluator": {
                        "type": "field_equals",
                        "reference_path": "verdict",
                        "prediction_path": "verdict",
                    },
                }
            ],
        }
    )
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    output = Path(evaluation_inputs["output_dir"])
    # gt is the plain string "hello"; reference_path "verdict" cannot resolve.
    assert result["evaluation_status"] == "invalid"
    assert result["validation"]["invalid_cases"] == ["a"]
    errors = result["validation"]["case_errors"]["a"]
    assert any("reference_path 'verdict'" in error for error in errors)
    # An invalid report is still produced so the defect is diagnosable.
    assert (output / "evaluation_report.html").exists()
    report = json.loads((output / "report.json").read_text())
    assert report["evaluation_status"] == "invalid"


def test_evaluation_status_invalid_on_evaluator_type_mismatch(
    evaluation_module, evaluation_inputs, monkeypatch
):
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        Mock(return_value=({"choices": [{"message": {"content": "hello"}}]}, 0.1)),
    )
    evaluation_inputs["evaluation_config_json"] = json.dumps(
        {
            "mode": "agent_rubric",
            "rubrics": [
                {
                    "name": "count",
                    "criterion": "list count",
                    "weight": 1,
                    "required": False,
                    "evaluator": {"type": "list_count_match"},
                }
            ],
        }
    )
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["evaluation_status"] == "invalid"
    errors = result["validation"]["case_errors"]["a"]
    assert any("list_count_match" in error for error in errors)


def test_evaluation_status_invalid_on_never_resolving_prediction_path(
    evaluation_module, evaluation_inputs, monkeypatch
):
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        Mock(return_value=({"choices": [{"message": {"content": "hello"}}]}, 0.1)),
    )
    evaluation_inputs["evaluation_config_json"] = json.dumps(
        {
            "mode": "agent_rubric",
            "rubrics": [
                {
                    "name": "verdict",
                    "criterion": "verdict equals",
                    "weight": 1,
                    "required": False,
                    "evaluator": {
                        "type": "field_equals",
                        "prediction_path": "verdict",
                    },
                }
            ],
        }
    )
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    # Reference resolves (no per-case error); the prediction path never does,
    # which is a report-level misconfiguration rather than model variance.
    assert result["evaluation_status"] == "invalid"
    assert result["validation"]["case_errors"] == {}
    assert any(
        "prediction_path 'verdict'" in error
        for error in result["validation"]["report_errors"]
    )


def test_evaluation_status_flags_completeness_mismatch(evaluation_module):
    results = [{"id": "a", "error": None, "reference": "hello", "prediction": "hello"}]
    rubrics = [{"name": "r", "evaluator": {"type": "exact_match"}}]
    metrics = {"evaluated_cases": 1, "api_or_evaluation_errors": 0}
    status, case_errors, report_errors = evaluation_module._evaluation_status(
        results, rubrics, metrics, predictions_rows=2
    )
    assert status == "invalid"
    assert case_errors == {}
    assert any("predictions.jsonl has 2 rows" in error for error in report_errors)


def test_resume_recovers_a_truncated_checkpoint_tail(
    evaluation_module,
    evaluation_inputs,
    monkeypatch,
):
    call = Mock(return_value=({"choices": [{"message": {"content": "hello"}}]}, 0.1))
    monkeypatch.setattr(evaluation_module, "_call_api", call)
    evaluation_module.evaluate_inference(**evaluation_inputs)
    checkpoint = Path(evaluation_inputs["output_dir"]) / "predictions.partial.jsonl"
    with checkpoint.open("ab") as stream:
        stream.write(b'{"id": "incomplete", "prediction": "\xe4\xb8')
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["metrics"]["reused_predictions"] == 1
    assert call.call_count == 1
    assert len(checkpoint.read_text().splitlines()) == 1


def _png_bytes(width: int, height: int) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + kind
            + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        )

    raw = b"".join(
        b"\x00" + bytes((x * 7 + y * 13) & 0xFF for x in range(width * 3))
        for y in range(height)
    )
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw)),
            chunk(b"IEND", b""),
        )
    )


def _report_data(html: str) -> dict[str, Any]:
    marker = '<script id="report-data" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def _report_cases(output: Path) -> list[dict[str, Any]]:
    manifest = json.loads((output / "evaluation_cases_manifest.json").read_text())
    cases = []
    for chunk in manifest["chunks"]:
        cases.extend(json.loads((output / chunk["name"]).read_text())["items"])
    return cases


def _stub_api(
    evaluation_module: ModuleType, monkeypatch: Any, content: str = "hello"
) -> None:
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        Mock(return_value=({"choices": [{"message": {"content": content}}]}, 0.1)),
    )


def test_report_data_carries_annotation_binding(
    evaluation_module, evaluation_inputs, monkeypatch
):
    _stub_api(evaluation_module, monkeypatch, content="goodbye")
    evaluation_module.evaluate_inference(**evaluation_inputs)
    html = (
        Path(evaluation_inputs["output_dir"]) / "evaluation_report.html"
    ).read_text()
    data = _report_data(html)
    evaluation = json.loads(evaluation_inputs["evaluation_config_json"])
    digest = hashlib.sha256(
        json.dumps(
            evaluation, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    # The binding key for localStorage staging and import/export validation.
    assert data["run_id"] == "run-test-1"
    assert data["evaluation_config_sha256"] == digest
    assert data["evaluation_status"] == "validated"
    assert data["annotation_schema"] == {}
    assert "annotation" in data["panels"]
    assert data["case_delivery"] == "paged"
    assert data["case_count"] == 1
    assert data["cases"] == []
    assert [
        case["id"] for case in _report_cases(Path(evaluation_inputs["output_dir"]))
    ] == ["a"]


def test_report_renders_only_badcases(
    evaluation_module, evaluation_inputs, monkeypatch
):
    dataset = Path(evaluation_inputs["dataset_path"])
    dataset.write_text(
        json.dumps({"id": "a", "prompt": "say hello", "gt": "hello"})
        + "\n"
        + json.dumps({"id": "b", "prompt": "say bye", "gt": "bye"})
        + "\n"
    )
    _stub_api(evaluation_module, monkeypatch, content="hello")
    evaluation_module.evaluate_inference(**evaluation_inputs)
    html = (
        Path(evaluation_inputs["output_dir"]) / "evaluation_report.html"
    ).read_text()
    data = _report_data(html)
    # Only machine-failed cases are rendered, in predictions.jsonl order;
    # the summary still reports the full-run picture.
    assert [
        case["id"] for case in _report_cases(Path(evaluation_inputs["output_dir"]))
    ] == ["b"]
    assert data["summary"] == {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "pass_rate": 0.5,
    }


def test_report_renders_execution_errors_as_badcases(
    evaluation_module, evaluation_inputs, monkeypatch
):
    dataset = Path(evaluation_inputs["dataset_path"])
    dataset.write_text(
        dataset.read_text()
        + json.dumps({"id": "b", "prompt": "offline", "gt": "bye"})
        + "\n"
    )

    def call(endpoint, payload, api_key, timeout):
        prompt = payload["messages"][-1]["content"][-1]["text"]
        if prompt == "offline":
            raise OSError("unavailable")
        return {"choices": [{"message": {"content": "hello"}}]}, 0.1

    monkeypatch.setattr(evaluation_module, "_call_api", call)
    evaluation_module.evaluate_inference(**evaluation_inputs)

    cases = _report_cases(Path(evaluation_inputs["output_dir"]))
    assert [case["id"] for case in cases] == ["b"]
    assert cases[0]["error"] == "OSError: unavailable"


def test_report_data_uses_annotation_schema_from_spec(
    evaluation_module, evaluation_inputs, monkeypatch
):
    _stub_api(evaluation_module, monkeypatch)
    evaluation_inputs["report_spec_json"] = json.dumps(
        {
            "template_id": "evaluation-basic-v1",
            "annotation_schema": {
                "taxonomy_labels": ["small_object", "low_contrast"],
                "dispositions": ["fix_ground_truth"],
            },
        }
    )
    evaluation_module.evaluate_inference(**evaluation_inputs)
    html = (
        Path(evaluation_inputs["output_dir"]) / "evaluation_report.html"
    ).read_text()
    data = _report_data(html)
    assert data["annotation_schema"] == {
        "taxonomy_labels": ["small_object", "low_contrast"],
        "dispositions": ["fix_ground_truth"],
    }


def test_html_embeds_annotation_frontend(
    evaluation_module, evaluation_inputs, monkeypatch
):
    _stub_api(evaluation_module, monkeypatch, content="incorrect answer")
    evaluation_module.evaluate_inference(**evaluation_inputs)
    assert _report_cases(Path(evaluation_inputs["output_dir"]))[0]["input"] == {
        "prompt": "say hello"
    }
    html = (
        Path(evaluation_inputs["output_dir"]) / "evaluation_report.html"
    ).read_text()
    # Sparse localStorage staging keyed by run_id + evaluation_config_sha256.
    assert "evaluation-annotations:" in html
    # Annotation controls unlock only for a validated report.
    assert 'data.evaluation_status === "validated"' in html
    # Enumerated decision/disposition options from the annotation contract.
    assert "confirmed_badcase" in html
    assert "evaluation_false_alarm" in html
    assert "data_augmentation" in html
    assert "model_miss" in html
    # Full-coverage export targets.
    assert "evaluation_annotations.jsonl" in html
    assert "evaluation_annotations.json" in html
    # In-progress/reviewed distinction keeps the review progress honest;
    # confirmed_badcase records additionally require failure_modes + severity.
    assert "function badgeState" in html
    assert '"in_progress"' in html
    assert "annotationComplete" in html
    # Single-case rendering refreshes the focused workspace after edits.
    assert "renderWorkspace" in html
    assert "global.alert(" not in html
    # Mode determination is the deterministic online marker injected by the
    # stable report entry (never fetch probing); the raw Storage file has no
    # marker and stays read-only with downloads disabled.
    assert "global.__EVAL_REPORT_API__" in html
    assert "window.EvaluationReport.mount" in html
    assert "Autosave unavailable" in html
    # Edited-but-incomplete records save as in_progress with their partial
    # annotation; reviewed stays reserved for conditionally-complete records.
    assert "review_status: annotation" in html
    assert '(isComplete(id) ? "reviewed" : "in_progress")' in html
    # Optional corrected answer is part of the annotation contract.
    assert "corrected_answer" in html
    assert "data_quality_reasons" in html
    assert "training_suggestion" in html
    assert "missing_scenario" in html
    assert "acceptance_reason" in html
    assert "verification_items" in html
    # Autosave: frozen save_id/body PUT with exponential backoff, and
    # downloads that fetch the Storage-side latest via the export API.
    assert "crypto.randomUUID" in html
    assert 'method: "PUT"' in html
    assert "Math.min(backoff * 2, 30000)" in html
    assert "store.exportUrl(entry[0])" in html
    assert "beforeunload" in html
    assert 'event.key === "Escape"' in html
    assert "queue-open" in html
    # The manual round-trip is gone: no import/export file plumbing at all.
    assert "doExport" not in html
    assert "doImport" not in html
    assert '":exported"' not in html
    assert "window.confirm" not in html
    # Report warnings (e.g. media degradation notices) surface as banners.
    assert "(data.warnings || [])" in html


def test_initial_annotation_products_written_on_success(
    evaluation_module, evaluation_inputs, monkeypatch
):
    dataset = Path(evaluation_inputs["dataset_path"])
    dataset.write_text(
        dataset.read_text()
        + json.dumps({"id": "b", "prompt": "fail", "gt": "hello"})
        + "\n"
    )

    def call(endpoint, payload, api_key, timeout):
        text = payload["messages"][-1]["content"][-1]["text"]
        reply = "wrong" if text == "fail" else "hello"
        return {"choices": [{"message": {"content": reply}}]}, 0.1

    monkeypatch.setattr(evaluation_module, "_call_api", call)
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["status"] == "succeeded"
    output = Path(evaluation_inputs["output_dir"])

    # Revision-0 annotation objects cover exactly the machine-failed set.
    annotations_r0 = [
        json.loads(line)
        for line in (output / "evaluation_annotations.r0.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    assert [record["case_id"] for record in annotations_r0] == ["b"]
    record = annotations_r0[0]
    assert record["schema_version"] == 1
    assert record["review_status"] == "unreviewed"
    assert record["human_annotation"] is None
    assert record["machine_result"]["passed"] is False
    assert record["machine_result"]["prediction"] == "wrong"
    assert record["machine_result"]["reference"] == "hello"
    # Deterministic bytes: no timestamps in revision-0 records.
    assert "updated_at" not in json.dumps(record)

    # Merged records cover the full set: passed -> not_in_scope, failed ->
    # unreviewed; machine results are rebuilt from predictions.jsonl.
    labeled = [
        json.loads(line)
        for line in (output / "labeled_evaluation_results.r0.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    assert [row["id"] for row in labeled] == ["a", "b"]
    by_id = {row["id"]: row for row in labeled}
    # Merged records are the complete original prediction rows (inputs,
    # rubric results, errors, timings) plus run binding and human_review —
    # every machine field must match predictions.jsonl verbatim.
    prediction_rows = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (output / "predictions.jsonl").read_text().splitlines()
            if line
        )
    }
    for row_id, merged in by_id.items():
        for key, value in prediction_rows[row_id].items():
            assert merged[key] == value, f"merged.{key} diverges for {row_id}"
        assert merged["run_id"]
        assert merged["evaluation_config_sha256"]
    assert by_id["a"]["human_review"] == {
        "review_status": "not_in_scope",
        "annotation": None,
    }
    assert by_id["b"]["human_review"]["review_status"] == "unreviewed"
    assert by_id["b"]["passed"] is False

    # Canonical-name views mirror revision 0 byte-for-byte (jsonl) and carry
    # the snapshot wrapper with revision=0.
    assert (output / "evaluation_annotations.jsonl").read_bytes() == (
        output / "evaluation_annotations.r0.jsonl"
    ).read_bytes()
    assert (output / "labeled_evaluation_results.jsonl").read_bytes() == (
        output / "labeled_evaluation_results.r0.jsonl"
    ).read_bytes()
    snapshot = json.loads((output / "evaluation_annotations.json").read_text())
    assert snapshot["revision"] == 0
    assert snapshot["review_scope"] == "machine_failed"
    assert snapshot["total"] == 1
    assert snapshot["reviewed"] == 0
    assert snapshot["in_progress"] == 0
    assert snapshot["unreviewed"] == 1

    # report.json does not register revision-suffixed annotation objects.
    report = json.loads((output / "report.json").read_text())
    assert not any(
        "annotation" in name or "labeled" in name for name in report["artifacts"]
    )

    # Re-running (resume) reproduces byte-identical revision-0 objects.
    first = (output / "evaluation_annotations.r0.jsonl").read_bytes()
    evaluation_module.evaluate_inference(**evaluation_inputs)
    assert (output / "evaluation_annotations.r0.jsonl").read_bytes() == first


def test_invalid_report_writes_no_annotation_products(
    evaluation_module, evaluation_inputs, monkeypatch
):
    # A rubric whose reference_path cannot resolve keeps inference successful
    # (valid=True) but forces evaluation_status="invalid"; annotation
    # products must only initialize for validated reports.
    monkeypatch.setattr(
        evaluation_module,
        "_call_api",
        Mock(return_value=({"choices": [{"message": {"content": "hello"}}]}, 0.1)),
    )
    evaluation_inputs["evaluation_config_json"] = json.dumps(
        {
            "mode": "agent_rubric",
            "rubrics": [
                {
                    "name": "verdict",
                    "criterion": "verdict equals",
                    "weight": 1,
                    "required": True,
                    "evaluator": {
                        "type": "field_equals",
                        "reference_path": "verdict",
                        "prediction_path": "verdict",
                    },
                }
            ],
        }
    )
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["status"] == "succeeded"
    assert result["evaluation_status"] == "invalid"
    output = Path(evaluation_inputs["output_dir"])
    assert (output / "evaluation_report.html").exists()
    assert not (output / "evaluation_annotations.r0.jsonl").exists()
    assert not (output / "labeled_evaluation_results.r0.jsonl").exists()
    assert not (output / "evaluation_annotations.json").exists()


def test_resume_preserves_later_annotation_state(
    evaluation_module, evaluation_inputs, monkeypatch, tmp_path
):
    # The eval pod cannot read the Product Store commit state, so the guard
    # is intentionally file-based and conservative: any r{N>0} object
    # (committed or a failed-save leftover) disables re-initialization.
    # Phase 2 adds the commit-state-aware end-to-end coverage: a real
    # committed rN followed by resume, and an orphaned uncommitted rN.
    _stub_api(evaluation_module, monkeypatch)
    output = Path(evaluation_inputs["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_annotations.r3.jsonl").write_text(
        '{"case_id": "a", "review_status": "reviewed"}\n'
    )
    (output / "labeled_evaluation_results.r3.jsonl").write_text("{}\n")
    (output / "evaluation_annotations.jsonl").write_text("SENTINEL-VIEW\n")
    (output / "evaluation_annotations.json").write_text("SENTINEL-SNAPSHOT\n")
    (output / "labeled_evaluation_results.jsonl").write_text("SENTINEL-LABELED\n")
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["status"] == "succeeded"
    assert (
        output / "evaluation_annotations.r3.jsonl"
    ).read_text() == '{"case_id": "a", "review_status": "reviewed"}\n'
    assert (output / "evaluation_annotations.jsonl").read_text() == "SENTINEL-VIEW\n"
    assert (output / "evaluation_annotations.json").read_text() == "SENTINEL-SNAPSHOT\n"
    assert (
        output / "labeled_evaluation_results.jsonl"
    ).read_text() == "SENTINEL-LABELED\n"
    assert not (output / "evaluation_annotations.r0.jsonl").exists()


def test_report_scope_review_scope_reaches_payload(
    evaluation_module, evaluation_inputs, monkeypatch
):
    _stub_api(evaluation_module, monkeypatch)
    evaluation_inputs["report_spec_json"] = json.dumps(
        {
            "template_id": "evaluation-basic-v1",
            "title": "Scoped review",
        }
    )
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["warnings"] == []
    output = Path(evaluation_inputs["output_dir"])
    spec = json.loads((output / "resolved_report_spec.json").read_text())
    assert spec["review_scope"] == "machine_failed"
    payload = _report_data((output / "evaluation_report.html").read_text())
    assert payload["review_scope"] == "machine_failed"


def test_report_data_marks_invalid_status_for_locked_annotation(
    evaluation_module, evaluation_inputs, monkeypatch
):
    _stub_api(evaluation_module, monkeypatch)
    evaluation_inputs["evaluation_config_json"] = json.dumps(
        {
            "mode": "agent_rubric",
            "rubrics": [
                {
                    "name": "verdict",
                    "criterion": "verdict equals",
                    "weight": 1,
                    "required": True,
                    "evaluator": {
                        "type": "field_equals",
                        "reference_path": "verdict",
                        "prediction_path": "verdict",
                    },
                }
            ],
        }
    )
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    assert result["evaluation_status"] == "invalid"
    html = (
        Path(evaluation_inputs["output_dir"]) / "evaluation_report.html"
    ).read_text()
    assert _report_data(html)["evaluation_status"] == "invalid"


def test_report_media_is_externalized_from_html(
    evaluation_module, evaluation_inputs, monkeypatch
):
    _media_eval_inputs(evaluation_inputs, b"any readable bytes")
    _stub_api(evaluation_module, monkeypatch, content="goodbye")
    result = evaluation_module.evaluate_inference(**evaluation_inputs)
    output = Path(evaluation_inputs["output_dir"])
    html = (output / "evaluation_report.html").read_text()
    media = _report_cases(output)[0]["media"]
    assert "base64" not in html
    assert media[0]["media_id"] == "00000000-000.png"
    assert media[0]["caption"] == "sample.png"
    assert (output / "evaluation_media" / media[0]["media_id"]).read_bytes() == (
        b"any readable bytes"
    )
    assert _report_data(html)["warnings"] == []
    assert result["warnings"] == []


def test_report_case_products_are_chunked_and_externalize_media(
    evaluation_module, tmp_path
):
    encoded = base64.b64encode(b"image bytes").decode("ascii")
    cases = [
        {
            "id": f"case-{index}",
            "media": (
                [
                    {
                        "url": f"data:image/png;base64,{encoded}",
                        "caption": "sample.png",
                    }
                ]
                if index == 0
                else []
            ),
        }
        for index in range(21)
    ]
    cases[1]["media"] = [
        {"url": "https://example.test/image.png", "caption": "remote.png"}
    ]

    evaluation_module._write_report_case_products(cases, tmp_path)

    manifest = json.loads((tmp_path / "evaluation_cases_manifest.json").read_text())
    assert manifest == {
        "schema_version": 1,
        "total": 21,
        "chunk_size": 20,
        "chunks": [
            {
                "name": "evaluation_cases/chunk-00000.json",
                "start": 0,
                "count": 20,
            },
            {
                "name": "evaluation_cases/chunk-00001.json",
                "start": 20,
                "count": 1,
            },
        ],
    }
    index = json.loads((tmp_path / "evaluation_cases_index.json").read_text())
    assert index["cases"][0]["media"] == [
        {"media_id": "00000000-000.png", "mime_type": "image/png"}
    ]
    assert index["cases"][20] == {
        "id": "case-20",
        "chunk": 1,
        "offset": 0,
        "media": [],
    }
    first_chunk = json.loads(
        (tmp_path / "evaluation_cases" / "chunk-00000.json").read_text()
    )
    assert first_chunk["items"][0]["media"] == [
        {
            "caption": "sample.png",
            "media_id": "00000000-000.png",
            "mime_type": "image/png",
        }
    ]
    assert "base64" not in json.dumps(first_chunk)
    assert first_chunk["items"][1]["media"] == [
        {"url": "https://example.test/image.png", "caption": "remote.png"}
    ]
    assert (
        tmp_path / "evaluation_media" / "00000000-000.png"
    ).read_bytes() == b"image bytes"


def test_report_media_missing_file_degrades_to_caption_placeholder(
    evaluation_module, evaluation_inputs, tmp_path
):
    # A media file that vanishes between inference input staging and report
    # rendering degrades to a caption placeholder instead of failing the run.
    row = {
        "id": "a",
        "passed": False,
        "reference": "hello",
        "prediction": "goodbye",
        "metrics": {},
        "details": {},
        "latency_seconds": 0.1,
        "error": None,
        "overall_score": 0.0,
        "rubric_results": [],
        "media": ["missing.png"],
    }
    output = tmp_path / "out"
    output.mkdir()
    warnings = evaluation_module._build_html(
        [row],
        {
            "dataset_path": evaluation_inputs["dataset_path"],
            "total": 1,
            "passed": 0,
            "failed": 1,
            "pass_rate": 0.0,
            "aggregate_metrics": {},
            "rubric_metrics": [],
        },
        Path(evaluation_inputs["dataset_path"]),
        {
            "user_prompt_field": "prompt",
            "reference_field": "gt",
            "media_field": "image",
        },
        output,
        {
            "template_id": "evaluation-media-v1",
            "template_version": "1.0.0",
            "title": "Inference evaluation report",
            "enabled_panels": ["rubric", "annotation", "raw_output", "media_compare"],
        },
        "validated",
        {},
        "run-test-1",
        "sha",
    )
    data = _report_data((output / "evaluation_report.html").read_text())
    assert _report_cases(output)[0]["media"] == [{"caption": "missing.png"}]
    assert any("could not be read" in warning for warning in warnings)
    assert data["warnings"] == warnings


def _media_eval_inputs(evaluation_inputs: dict, media_bytes: bytes) -> None:
    dataset = Path(evaluation_inputs["dataset_path"])
    (dataset.parent / "sample.png").write_bytes(media_bytes)
    row = json.loads(dataset.read_text())
    row["image"] = "sample.png"
    dataset.write_text(json.dumps(row) + "\n")
    evaluation_inputs["dataset_config_json"] = json.dumps(
        {
            "user_prompt_field": "prompt",
            "reference_field": "gt",
            "media_field": "image",
        }
    )


def test_report_generation_rejects_non_finite_numbers(evaluation_module, tmp_path):
    """NaN/Infinity must fail generation, not produce a blank report page.

    The browser's JSON.parse rejects NaN/Infinity literals, so embedding
    them would kill the page's only script and render an empty document.
    """
    output = tmp_path / "out"
    output.mkdir()
    spec = {
        "template_id": "evaluation-basic-v1",
        "template_version": "1.2.0",
        "title": "Inference evaluation report",
        "enabled_panels": ["rubric", "annotation"],
    }
    metrics = {
        "dataset_path": str(tmp_path / "cases.jsonl"),
        "total": 1,
        "passed": 0,
        "failed": 1,
        "pass_rate": 0.0,
        "aggregate_metrics": {},
        "rubric_metrics": {},
    }

    def build(score: float) -> None:
        evaluation_module._build_html(
            [
                {
                    "id": "b",
                    "passed": False,
                    "reference": "ref",
                    "prediction": "pred",
                    "overall_score": score,
                    "rubric_results": [
                        {
                            "name": "correct",
                            "passed": False,
                            "score": score,
                            "weight": 1,
                            "criterion": "c",
                            "evidence": [],
                            "failure_reason": "f",
                        }
                    ],
                    "metrics": {},
                    "details": "",
                    "latency_seconds": 0.1,
                    "error": None,
                }
            ],
            metrics,
            tmp_path / "cases.jsonl",
            {},
            output,
            spec,
            "validated",
            {},
            "run-test-1",
            "sha",
        )

    with pytest.raises(ValueError):
        build(float("nan"))
    with pytest.raises(ValueError):
        build(float("inf"))
    build(0.0)
    assert _report_cases(output)[0]["overall_score"] == 0.0


def test_report_payload_escapes_script_closing_sequence(
    evaluation_module, evaluation_inputs, monkeypatch
):
    _stub_api(
        evaluation_module, monkeypatch, content="</script><script>alert(1)</script>"
    )
    evaluation_module.evaluate_inference(**evaluation_inputs)
    html = (
        Path(evaluation_inputs["output_dir"]) / "evaluation_report.html"
    ).read_text()
    marker = '<script id="report-data" type="application/json">'
    start = html.index(marker)
    end = html.index("</script>", start)
    # No literal closing sequence may appear inside the data block, so the
    # payload can never terminate the JSON script element early.
    assert "</script>" not in html[start + len(marker) : end]
    # The hostile content survives the round-trip intact for the annotator.
    assert _report_cases(Path(evaluation_inputs["output_dir"]))[0]["prediction"] == (
        "</script><script>alert(1)</script>"
    )


def test_report_js_avoids_dangerous_dom_sinks(evaluation_module):
    js = evaluation_module._read_report_asset("evaluation_report.js")
    assert "global.EvaluationReport" in js
    assert "createHttpStorage" in js
    assert "function mount(container, options)" in js
    assert 'version: "1.0.0"' in js
    for sink in (
        "innerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function(",
    ):
        assert sink not in js


def test_report_css_is_scoped_to_library_container(evaluation_module):
    css = evaluation_module._read_report_asset("evaluation_report.css")
    assert ".evaluation-report button" in css
    assert "\nbody {" not in css
    assert "\nbutton {" not in css
    assert "\nimg {" not in css


def test_report_ui_source_modules_keep_their_responsibilities():
    source = MODULE_PATH.parent / "report_ui"
    annotation_store = (source / "annotation-store.js").read_text()
    renderer = (source / "renderer.js").read_text()
    public_api = (source / "index.js").read_text()

    assert "saveAnnotations" in annotation_store
    assert "localStorage" in annotation_store
    assert "document.createElement" not in annotation_store
    assert "document.createElement" in renderer
    assert "localStorage" not in renderer
    assert "global.fetch" not in renderer
    assert "function mount(container, options)" in public_api
    assert "beforeUnload" in public_api


_AUTOSAVE_JS_DRIVER = r"""
"use strict";
// Behavior test for the report UI library autosave pipeline. It runs the real
// library in a node vm with a minimal DOM/fetch/timer surface and
// drives it the way a user would: click, dispatch events, flush timers.
const fs = require("fs");
const vm = require("vm");
const assert = require("assert/strict");

const reportJs = fs.readFileSync(process.argv[2], "utf8");

function makePayload() {
  return {
    title: "t",
    template_id: "evaluation-basic-v1",
    template_version: "1.2.0",
    panels: ["rubric", "annotation"],
    evaluation_status: "validated",
    run_id: "run-1",
    evaluation_config_sha256: "sha",
    review_scope: "machine_failed",
    annotation_schema: {},
    dataset_path: "/d",
    summary: { total: 1, passed: 0, failed: 1, pass_rate: 0 },
    aggregate_metrics: {},
    rubric_metrics: {},
    warnings: [],
    cases: [
      {
        id: "c1",
        status: "failed",
        media: [],
        rubric_items: [],
        overall_score: 0,
        reference: "ref",
        prediction: "pred",
        detail: {},
        latency_seconds: 0.1,
        error: null,
        validation_errors: [],
      },
    ],
  };
}

async function boot(payload, initialResponders, storage, apiConfig) {
  const allElements = [];
  function makeElement(tag) {
    const listeners = {};
    const el = {
      tagName: String(tag).toUpperCase(),
      children: [],
      className: "",
      textContent: "",
      value: "",
      checked: false,
      parentNode: null,
      style: {},
      dataset: {},
      classList: {
        set: new Set(),
        add(c) { this.set.add(c); },
        remove(c) { this.set.delete(c); },
        toggle(c, force) {
          const want = force === undefined ? !this.set.has(c) : Boolean(force);
          if (want) this.set.add(c); else this.set.delete(c);
          return want;
        },
        contains(c) { return this.set.has(c); },
      },
      appendChild(child) {
        el.children.push(child);
        child.parentNode = el;
        return child;
      },
      setAttribute(name, value) { el[name] = String(value); },
      focus() { document.activeElement = el; },
      replaceChild(next, old) {
        const index = el.children.indexOf(old);
        if (index >= 0) el.children[index] = next;
        next.parentNode = el;
        old.parentNode = null;
        return old;
      },
      addEventListener(type, fn) {
        (listeners[type] = listeners[type] || []).push(fn);
      },
      dispatch(type, event) {
        (listeners[type] || []).slice().forEach((fn) => fn(event || {}));
      },
      scrollIntoView() {},
    };
    allElements.push(el);
    return el;
  }

  const byId = {
    "report-data": makeElement("script"),
    report: makeElement("main"),
  };
  byId["report-data"].textContent = JSON.stringify(payload);
  const document = {
    getElementById(id) { return byId[id] || makeElement("div"); },
    createElement(tag) { return makeElement(tag); },
    createTextNode(text) { return { textContent: String(text) }; },
  };

  const timers = new Map();
  let timerSeq = 1;
  const setTimeout = (fn, ms) => {
    const id = timerSeq;
    timerSeq += 1;
    timers.set(id, { id, fn });
    return id;
  };
  const clearTimeout = (id) => { timers.delete(id); };
  async function drain() {
    for (let i = 0; i < 50; i += 1) await Promise.resolve();
  }
  async function flushTimers() {
    for (let guard = 0; guard < 100 && timers.size > 0; guard += 1) {
      const entry = timers.values().next().value;
      timers.delete(entry.id);
      entry.fn();
      await drain();
    }
  }

  const fetchCalls = [];
  const responders = [...initialResponders];
  const respond = (fn) => { responders.push(fn); };
  const fetch = (url, opts) => {
    const call = {
      url,
      method: (opts && opts.method) || "GET",
      body: opts && opts.body ? JSON.parse(opts.body) : null,
    };
    fetchCalls.push(call);
    const fn = responders.shift();
    assert.ok(fn, "unexpected fetch: " + call.method + " " + url);
    return Promise.resolve().then(() => fn(call));
  };

  const kv = new Map();
  const localStorage = {
    getItem: (k) => (kv.has(k) ? kv.get(k) : null),
    setItem: (k, v) => { kv.set(k, String(v)); },
    removeItem: (k) => { kv.delete(k); },
  };

  const sandbox = {
    document,
    fetch,
    setTimeout,
    clearTimeout,
    localStorage,
    alert: () => {},
    console,
  };
  sandbox.window = sandbox;
  sandbox.addEventListener = () => {};
  sandbox.__EVAL_REPORT_API__ = apiConfig || {
    annotations: "annotations", exports: "exports/",
  };
  vm.createContext(sandbox);
  vm.runInContext(reportJs, sandbox, { filename: "report.js" });
  assert.equal(typeof sandbox.EvaluationReport.mount, "function");
  const mountOptions = {
    dataElement: byId["report-data"],
  };
  if (storage) mountOptions.storage = storage;
  sandbox.EvaluationReport.mount(byId.report, mountOptions);
  await drain();
  return { allElements, fetchCalls, respond, timers, kv, drain, flushTimers };
}

function findRadio(elements, name, value) {
  const radio = elements.find(
    (el) => el.type === "radio" && el.name === name && el.value === value);
  assert.ok(radio, "radio " + name + "=" + value + " rendered");
  return radio;
}

function findSaveState(elements) {
  return elements.find((el) => String(el.className).startsWith("save-state"));
}

async function scenarioInjectedStorage() {
  const storageCalls = [];
  const storage = {
    loadAnnotations: () => {
      storageCalls.push("load");
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          revision: 0, records: [], reviewed: 0, in_progress: 0,
          unreviewed: 1, overall_status: "in_progress",
        }),
      });
    },
    saveAnnotations: () => {
      throw new Error("unexpected save");
    },
    exportUrl: (name) => "exports/" + name,
  };
  const ctx = await boot(makePayload(), [], storage);
  assert.deepEqual(storageCalls, ["load"]);
  assert.equal(ctx.fetchCalls.length, 0,
    "an injected storage adapter replaces the default HTTP adapter");
}

async function scenarioPagedSummaryFirstLoad() {
  const payload = makePayload();
  payload.cases = [];
  payload.case_delivery = "paged";
  payload.case_count = 1000;
  const pageItems = Array.from({ length: 20 }, (_, index) => ({
    id: "case-" + index,
    status: "failed",
    media: [],
    rubric_items: [],
    overall_score: 0,
    reference: "ref-" + index,
    prediction: "pred-" + index,
    detail: {},
    latency_seconds: 0.1,
    error: null,
    validation_errors: [],
    review_status: "unreviewed",
    human_annotation: null,
  }));
  const ctx = await boot(payload, [
    (call) => {
      assert.ok(call.url.startsWith("cases?limit=20"));
      return {
        ok: true, status: 200,
        json: async () => ({
          items: pageItems, next_cursor: "cursor-20",
          total: 1000, filtered_total: 1000,
        }),
      };
    },
    (call) => {
      assert.equal(call.url, "summary");
      return {
        ok: true, status: 200,
        json: async () => ({
          revision: 0, reviewed: 0, in_progress: 0,
          unreviewed: 1000, overall_status: "in_progress",
        }),
      };
    },
  ], null, {
    annotations: "annotations",
    annotation_summary: "summary",
    cases: "cases",
    exports: "exports/",
  });
  await ctx.drain();
  assert.equal(ctx.fetchCalls.length, 2);
  assert.ok(!ctx.fetchCalls.some((call) => call.url === "annotations"),
    "paged reports must not download full annotations on first load");
}

// D5 regression: a failed initial load must defer saving, never emit a PATCH
// with a null base_revision. Edits stay buffered; a successful retry
// submits them on top of the loaded revision.
async function scenarioLoadFailureThenRetry() {
  const ctx = await boot(makePayload(), [
    () => ({ ok: false, status: 500, json: async () => ({}) }),
  ]);
  const { allElements, fetchCalls, timers, kv } = ctx;
  assert.equal(fetchCalls.length, 1);
  assert.equal(fetchCalls[0].method, "GET");
  const retry = allElements.find(
    (el) => el.tagName === "BUTTON" && el.textContent === "Retry");
  assert.ok(retry, "retry button rendered after load failure");

  findRadio(allElements, "resolution-c1", "uncertain")
    .dispatch("change", {});
  await ctx.drain();
  assert.equal(fetchCalls.length, 1,
    "no request may leave the page while no baseline is loaded");
  assert.equal(timers.size, 0, "no save timer without a baseline");
  const buffered = JSON.parse(kv.get("evaluation-annotations:run-1:sha"));
  assert.equal(buffered.c1.annotation.decision, "needs_discussion",
    "the edit stays in the unconfirmed crash buffer");

  ctx.respond(() => ({
    ok: true,
    status: 200,
    json: async () => ({
      revision: 3, records: [], reviewed: 0, in_progress: 0,
      unreviewed: 1, overall_status: "in_progress",
    }),
  }));
  retry.dispatch("click", {});
  await ctx.drain();
  assert.equal(fetchCalls.length, 2);
  assert.equal(fetchCalls[1].method, "GET");
  assert.equal(timers.size, 1, "the buffered edit is queued once a baseline exists");

  ctx.respond(() => ({
    ok: true,
    status: 200,
    json: async () => ({
      revision: 4, reviewed: 0, in_progress: 1, unreviewed: 0,
      overall_status: "in_progress",
    }),
  }));
  await ctx.flushTimers();
  const patches = fetchCalls.filter((call) => call.method === "PATCH");
  assert.equal(patches.length, 1);
  assert.equal(patches[0].body.base_revision, 3);
  assert.ok(
    Number.isInteger(patches[0].body.base_revision) &&
      patches[0].body.base_revision >= 0,
    "base_revision must be a non-negative integer");
  assert.ok(patches[0].body.save_id.length > 0);
  assert.equal(patches[0].body.changes.length, 1);
  assert.equal(patches[0].body.changes[0].case_id, "c1");
  assert.equal(
    patches[0].body.changes[0].human_annotation.decision,
    "needs_discussion");
  assert.equal(
    patches[0].body.changes[0].human_annotation.resolution_code,
    "uncertain");
  const afterAck = JSON.parse(kv.get("evaluation-annotations:run-1:sha") || "{}");
  assert.ok(!afterAck.c1, "acknowledged edits leave the crash buffer");
}

// A genuine 422 stays a deterministic rejection: the body is not retried.
async function scenarioValidationRejection() {
  const ctx = await boot(makePayload(), [
    () => ({
      ok: true,
      status: 200,
      json: async () => ({
        revision: 0, records: [], reviewed: 0, in_progress: 0,
        unreviewed: 1, overall_status: "in_progress",
      }),
    }),
  ]);
  const { allElements, fetchCalls, timers } = ctx;
  await ctx.drain();
  assert.equal(fetchCalls.length, 1);

  allElements.filter((el) => el.className === 'primary').at(-1).dispatch('click');
  const missingResolution = allElements.filter(
    (el) => el.className === 'validation-summary').at(-1);
  assert.equal(missingResolution.hidden, false);
  assert.ok(missingResolution.textContent.includes('处理结论'));

  findRadio(allElements, "resolution-c1", "uncertain").dispatch("change", {});
  await ctx.drain();
  const verification = allElements.filter(
    (el) => el.tagName === 'TEXTAREA' && el['aria-invalid'] === 'true').at(-1);
  assert.ok(verification, 'missing verification field is marked inline');
  verification.value = '核查 GT';
  verification.dispatch('input');
  const summary = allElements.filter(
    (el) => el.className === 'validation-summary').at(-1);
  assert.equal(summary.hidden, true, 'filling the field clears the hint');
  assert.equal(verification['aria-invalid'], 'false');
  ctx.respond(() => ({
    ok: false,
    status: 422,
    json: async () => ({ detail: { message: "m", errors: ["bad record"] } }),
  }));
  await ctx.flushTimers();
  const patches = fetchCalls.filter((call) => call.method === "PATCH");
  assert.equal(patches.length, 1);
  assert.equal(patches[0].body.base_revision, 0);
  const saveState = findSaveState(allElements);
  assert.ok(saveState, "save-state element rendered");
  assert.ok(String(saveState.className).includes("rejected"),
    "a 422 marks the save rejected, got: " + saveState.className);
  assert.ok(String(saveState.textContent).includes("服务端拒绝"));
  assert.equal(timers.size, 0, "a rejected save never schedules a retry");
  await ctx.drain();
  assert.equal(fetchCalls.length, 2, "a rejected save never retries");
}

(async () => {
  await scenarioInjectedStorage();
  await scenarioPagedSummaryFirstLoad();
  await scenarioLoadFailureThenRetry();
  await scenarioValidationRejection();
  console.log("autosave behavior OK");
})().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_report_js_autosave_pipeline_behavior(evaluation_module, tmp_path):
    """Executable check of the autosave pipeline under node.

    Covers the load-failure path (no PATCH without a loaded baseline, edits
    survive the retry) and the 422 rejection path, driving the real
    embedded library instead of asserting on its source text.
    """
    driver = tmp_path / "autosave_driver.cjs"
    driver.write_text(_AUTOSAVE_JS_DRIVER, encoding="utf-8")
    report_js = tmp_path / "report.js"
    report_js.write_text(
        evaluation_module._read_report_asset("evaluation_report.js"),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["node", str(driver), str(report_js)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"node driver failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "autosave behavior OK" in result.stdout


_CASE_SOURCE_JS_DRIVER = r"""
"use strict";
const fs = require("fs");
const vm = require("vm");
const assert = require("assert/strict");

const scheduled = [];
const idle = [];
let timerId = 0;
const sandbox = {
  setTimeout(fn, ms) {
    const entry = { id: ++timerId, fn, ms, cancelled: false };
    scheduled.push(entry);
    return entry.id;
  },
  clearTimeout(id) {
    const entry = scheduled.find((candidate) => candidate.id === id);
    if (entry) entry.cancelled = true;
  },
  requestIdleCallback(fn) {
    idle.push(fn);
    return idle.length;
  },
  cancelIdleCallback() {},
  console,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), sandbox);
const createCaseSource =
  sandbox.__EVALUATION_REPORT_MODULES__.caseSource.createCaseSource;

function page(start, total) {
  const items = Array.from(
    { length: Math.min(20, total - start) },
    (_, offset) => ({
      id: "case-" + (start + offset),
      review_status: "unreviewed",
      human_annotation: null,
    }),
  );
  return {
    items,
    next_cursor: start + items.length < total ? String(start + items.length) : null,
    total,
    filtered_total: total,
  };
}

async function drain() {
  for (let index = 0; index < 30; index += 1) await Promise.resolve();
}

async function cacheScenario() {
  const calls = [];
  const storage = {
    loadCases(options) {
      const start = options.cursor ? Number(options.cursor) : 0;
      calls.push(start);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => page(start, 1000),
      });
    },
  };
  const store = { registerCases() {}, releaseCases() {}, badgeState() {} };
  const source = createCaseSource({
    data: { cases: [] }, global: sandbox, storage, store,
  });
  await drain();
  assert.equal(source.items.length, 20);
  let current = "case-0";
  for (let index = 1; index < 1000; index += 1) {
    current = await source.move(current, 1);
  }
  assert.equal(current, "case-999");
  assert.equal(source.items.length, 60, "only three pages remain cached");
  assert.equal(source.startOffset, 940);

  while (current !== "case-940") current = await source.move(current, -1);
  current = await source.move(current, -1);
  assert.equal(current, "case-939", "an evicted page reloads by cursor");
  assert.equal(source.items.length, 60);
  assert.equal(source.startOffset, 920);

  source.ensureAhead(source.items.length - 1);
  assert.equal(idle.length, 1, "prefetch is scheduled while idle");
  idle.shift()();
  await drain();
  assert.ok(calls.includes(980));
  source.destroy();
}

async function retryScenario() {
  let attempts = 0;
  const storage = {
    loadCases() {
      attempts += 1;
      if (attempts === 1) return Promise.resolve({ ok: false, status: 503 });
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => page(0, 20),
      });
    },
  };
  const store = { registerCases() {}, releaseCases() {}, badgeState() {} };
  const source = createCaseSource({
    data: { cases: [] }, global: sandbox, storage, store,
  });
  await drain();
  const retry = scheduled.find(
    (entry) => !entry.cancelled && entry.ms === 1000,
  );
  assert.ok(retry, "a failed page schedules an exponential retry");
  retry.fn();
  await drain();
  assert.equal(attempts, 2);
  assert.equal(source.items.length, 20);
  source.destroy();
}

async function filterCacheScenario() {
  const calls = [];
  let finishOld;
  sandbox.AbortController = AbortController;
  const store = {
    revision: 1, registerCases() {}, releaseCases() {}, badgeState() {},
  };
  const storage = {
    loadCases(options) {
      calls.push(options);
      if (options.reviewStatus === 'in_progress') {
        return new Promise(resolve => { finishOld = resolve; });
      }
      return Promise.resolve({ok: true, status: 200,
        json: async () => ({...page(0, 20), revision: store.revision})});
    },
  };
  const source = createCaseSource({
    data: {cases: []}, global: sandbox, storage, store,
  });
  await drain();
  source.setReviewStatus('reviewed');
  await drain();
  assert.equal(calls.length, 2);
  source.setReviewStatus('all');
  assert.equal(source.items.length, 20, 'cached filter is restored immediately');
  assert.equal(calls.length, 2, 'switching back does not request the same page');
  store.revision = 2;
  source.setReviewStatus('reviewed');
  await drain();
  assert.equal(calls.length, 3, 'a new revision invalidates cached filters');
  source.setReviewStatus('in_progress');
  const oldRequest = calls[calls.length - 1];
  source.setReviewStatus('unreviewed');
  await drain();
  assert.equal(oldRequest.signal.aborted, true);
  finishOld({ok: true, status: 200, json: async () => ({
    ...page(980, 1000), revision: 2,
  })});
  await drain();
  assert.equal(source.items[0].id, 'case-0', 'late obsolete filter is ignored');
  source.destroy();
}

(async () => {
  await cacheScenario();
  await retryScenario();
  await filterCacheScenario();
  console.log("case source scale behavior OK");
})().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_report_case_source_bounds_cache_and_retries(tmp_path):
    driver = tmp_path / "case_source_driver.cjs"
    driver.write_text(_CASE_SOURCE_JS_DRIVER, encoding="utf-8")
    source = MODULE_PATH.parent / "report_ui" / "case-source.js"
    result = subprocess.run(
        ["node", str(driver), str(source)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"node driver failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "case source scale behavior OK" in result.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_report_store_rejects_stale_page_annotations(tmp_path):
    driver = tmp_path / "revision_driver.cjs"
    driver.write_text(
        r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const global = {
  localStorage: { getItem() { return null; }, setItem() {} },
  setTimeout, clearTimeout,
};
global.globalThis = global;
vm.createContext(global);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), global);
const createStore = global.__EVALUATION_REPORT_MODULES__.annotationStore
  .createAnnotationStore;
const store = createStore({
  global,
  data: { panels: ['annotation'], evaluation_status: 'validated', cases: [] },
  storage: {
    loadAnnotationSummary: async () => ({ok: true, json: async () => ({revision: 2})}),
    saveAnnotationChanges() { throw new Error('unexpected save'); },
  },
});
(async () => {
  store.load();
  for (let i = 0; i < 10; i++) await Promise.resolve();
  store.registerCases([{id: 'a', review_status: 'reviewed', human_annotation: {
    decision: 'evaluation_false_alarm', disposition: 'no_action', comment: 'saved',
  }}], 2);
  store.registerCases([
    {id: 'a', review_status: 'unreviewed', human_annotation: null},
  ], 1);
  assert.equal(store.getAnnotation('a').comment, 'saved');
  assert.equal(store.badgeState('a'), 'reviewed');
  store.destroy();
})().catch(error => { console.error(error); process.exitCode = 1; });
""",
        encoding="utf-8",
    )
    source = MODULE_PATH.parent / "report_ui" / "annotation-store.js"
    result = subprocess.run(
        ["node", str(driver), str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is unavailable")
def test_report_prediction_parser_preserves_raw_output(tmp_path):
    driver = tmp_path / "prediction_driver.cjs"
    driver.write_text(
        r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const sandbox = {};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sandbox);
const parse = sandbox.__EVALUATION_REPORT_MODULES__.renderer.parsePrediction;
const raw = '<think>reasoning</think> {"result":"pass","confidence":0.96}';
const parsed = parse(raw);
assert.equal(parsed.raw, raw);
assert.equal(parsed.reasoning, 'reasoning');
assert.equal(parsed.structured.result, 'pass');
assert.equal(parse('```json\n{"result":"fail"}\n```').structured.result, 'fail');
const literal = JSON.stringify({reason: '<think>literal</think>'});
assert.equal(parse(literal).structured.reason, '<think>literal</think>');
const invalid = '<think>reasoning</think> not valid JSON';
assert.equal(parse(invalid).raw, invalid);
assert.equal(parse(invalid).text, 'not valid JSON');
assert.equal(parse(invalid).structured, null);
assert.equal(parse('<think>unfinished').text, '<think>unfinished');
""",
        encoding="utf-8",
    )
    source = MODULE_PATH.parent / "report_ui" / "renderer.js"
    result = subprocess.run(
        ["node", str(driver), str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
