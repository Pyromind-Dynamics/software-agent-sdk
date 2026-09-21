from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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


def test_messages_exact_match_generates_reports_and_resumes(
    evaluation_module: ModuleType,
    inference_server: tuple[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    endpoint, state = inference_server
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-1",
                "messages": [{"role": "user", "content": "Say hello"}],
                "answer": "HELLO",
            }
        )
        + "\n"
    )
    arguments = {
        "endpoint": endpoint,
        "model": "test-model",
        "model_reference": "/workspace/models/test-v1",
        "dataset_path": str(dataset),
        "output_dir": str(tmp_path / "output"),
        "dataset_config_json": json.dumps(
            {
                "id_field": "id",
                "messages_field": "messages",
                "reference_field": "answer",
            }
        ),
        "evaluation_config_json": json.dumps(
            {"mode": "exact_match", "case_sensitive": False}
        ),
        "workers": 1,
    }

    result = evaluation_module.evaluate_inference(**arguments)
    summary = json.loads(result["summary_json"])

    assert summary == {
        "total": 1,
        "request_count": 1,
        "successful_predictions": 1,
        "evaluated_cases": 1,
        "passed": 1,
        "failed": 0,
        "pass_rate": 1.0,
        "api_or_evaluation_errors": 0,
    }
    assert all(
        Path(result[name]).is_file()
        for name in ("report_path", "metrics_path", "predictions_path")
    )
    assert len(state["requests"]) == 1

    arguments["endpoint"] = "http://127.0.0.1:1"
    evaluation_module.evaluate_inference(**arguments)
    assert len(state["requests"]) == 1


def test_resume_retries_only_checkpoint_rows_without_prediction(
    evaluation_module: ModuleType,
    inference_server: tuple[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    endpoint, state = inference_server
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"id": case_id, "prompt": "Say hello", "answer": "hello"})
            for case_id in ("case-1", "case-2")
        )
        + "\n"
    )
    output_dir = tmp_path / "output"
    arguments = {
        "endpoint": endpoint,
        "model": "test-model",
        "model_reference": "/workspace/models/test-v1",
        "dataset_path": str(dataset),
        "output_dir": str(output_dir),
        "dataset_config_json": json.dumps(
            {
                "id_field": "id",
                "user_prompt_field": "prompt",
                "reference_field": "answer",
            }
        ),
        "evaluation_config_json": json.dumps(
            {"mode": "exact_match", "case_sensitive": False}
        ),
        "workers": 1,
    }

    evaluation_module.evaluate_inference(**arguments)
    assert len(state["requests"]) == 2

    checkpoint = output_dir / "predictions.partial.jsonl"
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    failed_row = next(row for row in rows if row["id"] == "case-2")
    failed_row.update(
        {
            "prediction": None,
            "passed": False,
            "overall_score": None,
            "error": "TimeoutError: request timed out",
        }
    )
    checkpoint.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = evaluation_module.evaluate_inference(**arguments)

    assert len(state["requests"]) == 3
    summary = json.loads(result["summary_json"])
    assert summary["successful_predictions"] == 2
    assert summary["api_or_evaluation_errors"] == 0


def test_resume_reuses_prediction_that_failed_business_score(
    evaluation_module: ModuleType,
    inference_server: tuple[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    endpoint, state = inference_server
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps({"id": "case-1", "prompt": "Say hello", "answer": "goodbye"}) + "\n"
    )
    arguments = {
        "endpoint": endpoint,
        "model": "test-model",
        "model_reference": "/workspace/models/test-v1",
        "dataset_path": str(dataset),
        "output_dir": str(tmp_path / "output"),
        "dataset_config_json": json.dumps(
            {
                "id_field": "id",
                "user_prompt_field": "prompt",
                "reference_field": "answer",
            }
        ),
        "evaluation_config_json": json.dumps(
            {"mode": "exact_match", "case_sensitive": False}
        ),
        "workers": 1,
    }

    first = evaluation_module.evaluate_inference(**arguments)
    assert json.loads(first["summary_json"])["passed"] == 0
    assert len(state["requests"]) == 1

    arguments["endpoint"] = "http://127.0.0.1:1"
    resumed = evaluation_module.evaluate_inference(**arguments)

    assert json.loads(resumed["summary_json"])["passed"] == 0
    assert len(state["requests"]) == 1


def test_field_mapping_builds_multimodal_request_and_matches_json(
    evaluation_module: ModuleType,
    inference_server: tuple[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    endpoint, state = inference_server
    state["reply"] = '{"result":"pass","ignored":1}'
    image = "data:image/png;base64,iVBORw0KGgo="
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "case-2",
                "system": "Inspect the image",
                "prompt": "Return JSON",
                "media": [image],
                "expected": {"result": "pass"},
            }
        )
        + "\n"
    )

    result = evaluation_module.evaluate_inference(
        endpoint=endpoint,
        model="vision-model",
        model_reference="deployment-1",
        dataset_path=str(dataset),
        output_dir=str(tmp_path / "output"),
        dataset_config_json=json.dumps(
            {
                "id_field": "case_id",
                "system_prompt_field": "system",
                "user_prompt_field": "prompt",
                "media_field": "media",
                "reference_field": "expected",
            }
        ),
        evaluation_config_json=json.dumps(
            {"mode": "json_fields", "fields": ["result"]}
        ),
        workers=1,
    )

    assert json.loads(result["summary_json"])["passed"] == 1
    content = state["requests"][0]["messages"][1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == image
    assert content[1] == {"type": "text", "text": "Return JSON"}


def test_agent_rubric_is_predefined_and_does_not_call_runtime_planner(
    evaluation_module: ModuleType,
    inference_server: tuple[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    endpoint, state = inference_server
    state["reply"] = '{"label":"pass","confidence":0.9,"boxes":[[0,0,10,10]]}'
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-agent-rubric",
                "prompt": "Classify this sample.",
                "answer": {"label": "pass", "boxes": [[1, 1, 10, 10]]},
            }
        )
        + "\n"
    )

    result = evaluation_module.evaluate_inference(
        endpoint=endpoint,
        model="answer-model",
        model_reference="checkpoint-1",
        dataset_path=str(dataset),
        output_dir=str(tmp_path / "output"),
        dataset_config_json=json.dumps(
            {
                "id_field": "id",
                "user_prompt_field": "prompt",
                "reference_field": "answer",
            }
        ),
        evaluation_config_json=json.dumps(
            {
                "mode": "agent_rubric",
                "pass_threshold": 0.8,
                "rubrics": [
                    {
                        "name": "label_accuracy",
                        "criterion": "Prediction label matches the reference label.",
                        "weight": 4,
                        "required": True,
                        "evaluator": {
                            "type": "field_equals",
                            "prediction_path": "label",
                            "reference_path": "label",
                        },
                    },
                    {
                        "name": "confidence_range",
                        "criterion": "Confidence is between zero and one.",
                        "weight": 1,
                        "evaluator": {
                            "type": "number_range",
                            "prediction_path": "confidence",
                            "min": 0,
                            "max": 1,
                        },
                    },
                    {
                        "name": "bbox_accuracy",
                        "criterion": "Each reference box has a matching prediction.",
                        "weight": 2,
                        "required": True,
                        "evaluator": {
                            "type": "bbox_iou",
                            "prediction_path": "boxes",
                            "reference_path": "boxes",
                            "iou_threshold": 0.5,
                        },
                    },
                ],
            }
        ),
        workers=1,
    )

    assert len(state["requests"]) == 1
    prediction = json.loads(Path(result["predictions_path"]).read_text().strip())
    assert prediction["overall_score"] == 1.0
    assert prediction["details"]["rubric_source"] == "agent"
    assert [item["name"] for item in prediction["rubric_results"]] == [
        "label_accuracy",
        "confidence_range",
        "bbox_accuracy",
    ]


def test_required_rubric_blocks_an_otherwise_passing_score(
    evaluation_module: ModuleType,
) -> None:
    evaluation = evaluation_module._agent_rubric_evaluation(
        {"label": "defect", "confidence": 0.9},
        '{"label":"pass","confidence":0.9}',
        {
            "mode": "agent_rubric",
            "pass_threshold": 0.4,
            "rubrics": [
                {
                    "name": "label_accuracy",
                    "criterion": "Labels match.",
                    "weight": 1,
                    "required": True,
                    "evaluator": {
                        "type": "field_equals",
                        "prediction_path": "label",
                        "reference_path": "label",
                    },
                },
                {
                    "name": "confidence_range",
                    "criterion": "Confidence is valid.",
                    "weight": 1,
                    "evaluator": {
                        "type": "number_range",
                        "prediction_path": "confidence",
                        "min": 0,
                        "max": 1,
                    },
                },
            ],
        },
    )

    assert evaluation["overall_score"] == 0.5
    assert evaluation["passed"] is False
    assert evaluation["details"]["required_failures"] == ["label_accuracy"]


def test_bbox_score_penalizes_extra_predictions(
    evaluation_module: ModuleType,
) -> None:
    score = evaluation_module._bbox_score(
        [[0, 0, 10, 10], [20, 20, 30, 30]],
        [[0, 0, 10, 10]],
        0.5,
    )

    assert score == pytest.approx(2 / 3)


def test_legacy_image_mapping_remains_runtime_compatible(
    evaluation_module: ModuleType,
) -> None:
    values = evaluation_module._media_values(
        {"images": ["sample/gt.jpg", "sample/defect.jpg"]},
        {
            "image_field": "images",
            "image_order": ["defect.jpg", "gt.jpg"],
        },
    )

    assert values == ["sample/defect.jpg", "sample/gt.jpg"]


def test_runtime_planner_mode_is_rejected_before_inference(
    evaluation_module: ModuleType,
    inference_server: tuple[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    endpoint, state = inference_server
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"prompt": "Say hi", "answer": "hi"}) + "\n")

    with pytest.raises(ValueError, match="runtime planning was removed"):
        evaluation_module.evaluate_inference(
            endpoint=endpoint,
            model="test-model",
            model_reference="checkpoint-1",
            dataset_path=str(dataset),
            output_dir=str(tmp_path / "output"),
            dataset_config_json=json.dumps(
                {"user_prompt_field": "prompt", "reference_field": "answer"}
            ),
            evaluation_config_json=json.dumps(
                {"mode": "dynamic_rubric", "planner_prompt": "Plan rubrics"}
            ),
            workers=1,
        )

    assert state["requests"] == []


def test_evaluation_command_rejects_empty_dataset(
    evaluation_module: ModuleType,
    inference_server: tuple[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    endpoint, _ = inference_server
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("")

    with pytest.raises(ValueError, match="no evaluation cases"):
        evaluation_module.evaluate_inference(
            endpoint=endpoint,
            model="test-model",
            model_reference="checkpoint-1",
            dataset_path=str(dataset),
            output_dir=str(tmp_path / "output"),
            dataset_config_json=json.dumps(
                {
                    "user_prompt_field": "prompt",
                    "reference_field": "answer",
                }
            ),
            evaluation_config_json=json.dumps({"mode": "exact_match"}),
            workers=1,
        )


def test_command_line_calls_existing_endpoint(
    inference_server: tuple[str, dict[str, Any]],
    tmp_path: Path,
) -> None:
    endpoint, state = inference_server
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"prompt": "Say hello", "answer": "hello"}) + "\n")
    dataset_config = tmp_path / "dataset_config.json"
    dataset_config.write_text(
        json.dumps({"user_prompt_field": "prompt", "reference_field": "answer"})
    )
    evaluation_config = tmp_path / "evaluation_config.json"
    evaluation_config.write_text(json.dumps({"mode": "exact_match"}))

    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--endpoint",
            endpoint,
            "--model",
            "test-model",
            "--dataset-path",
            str(dataset),
            "--dataset-config",
            str(dataset_config),
            "--evaluation-config",
            str(evaluation_config),
            "--output-dir",
            str(tmp_path / "output"),
            "--workers",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert Path(result["report_path"]).is_file()
    assert len(state["requests"]) == 1


def test_gpu_command_starts_vllm_and_stops_it(
    evaluation_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = Mock()
    process.wait.return_value = 0
    popen = Mock(return_value=process)
    wait_for_vllm = Mock()
    evaluate = Mock(return_value={"report_path": "/workspace/out/report.html"})
    monkeypatch.setattr(evaluation_module.subprocess, "Popen", popen)
    monkeypatch.setattr(evaluation_module, "_wait_for_vllm", wait_for_vllm)
    monkeypatch.setattr(evaluation_module, "evaluate_inference", evaluate)

    result = evaluation_module.serve_and_evaluate(
        model_path="/workspace/models/checkpoint",
        model="default",
        dataset_path="/workspace/datasets/test.jsonl",
        output_dir="/workspace/eval",
        dataset_config_json="{}",
        evaluation_config_json="{}",
        gpu_count=2,
        max_model_len=8192,
    )

    command = popen.call_args.args[0]
    assert command[:3] == [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--max-model-len") + 1] == "8192"
    wait_for_vllm.assert_called_once()
    process.terminate.assert_called_once()
    assert result["report_path"].endswith("report.html")
