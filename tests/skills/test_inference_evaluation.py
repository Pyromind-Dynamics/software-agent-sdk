from __future__ import annotations

import importlib.util
import json
import threading
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
    assert "hello" in (output / "evaluation_report.html").read_text()
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


@pytest.mark.parametrize("change", ["dataset", "generation", "rubric", "media"])
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
