from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


MODULE_PATH = (
    Path(__file__).parents[2]
    / ".agents"
    / "skills"
    / "inference-evaluation"
    / "scripts"
    / "validate_pipeline.py"
)


@pytest.fixture
def validator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_pipeline", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_configs(tmp_path: Path, *, mode: str = "agent_rubric") -> tuple[Path, Path]:
    dataset_config = tmp_path / "inference_dataset_config.json"
    dataset_config.write_text(
        json.dumps(
            {
                "user_prompt_field": "prompt",
                "reference_field": "ground_truth",
            }
        )
    )
    evaluation_config = tmp_path / "inference_evaluation_config.json"
    evaluation_config.write_text(
        json.dumps(
            {
                "mode": mode,
                "rubrics": [
                    {
                        "name": "answer_correctness",
                        "criterion": "Prediction matches the ground truth answer.",
                        "weight": 1,
                        "evaluator": {"type": "exact_match"},
                    }
                ],
            }
        )
    )
    return dataset_config, evaluation_config


def test_accepts_gpu_command_agent_rubric_pipeline(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text(
        """evaluation = CustomCommandNode(
    id=1,
    command=(
        "python3 /workspace/.pyromind-agent/run/evaluate_inference.py "
        "--model-path /workspace/models/test --gpu-count 1 "
        "--dataset-path /workspace/datasets/test.jsonl "
        "--dataset-config /workspace/.pyromind-agent/run/dataset.json "
        "--evaluation-config /workspace/.pyromind-agent/run/evaluation.json "
        "--output-dir /workspace/eval/test --limit 0"
    ),
    cpu=4,
    memory=32,
    gpu_count=1,
    gpu_product="NVIDIA-L40S",
)
"""
    )
    dataset_config, evaluation_config = _write_configs(tmp_path)

    errors = validator_module.validate_pipeline(
        workflow, dataset_config, evaluation_config
    )

    assert errors == []


def test_rejects_benchmark_fallback_and_static_metric(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text(
        "inference = VLLMInference(id=1, model_path='/workspace/model')\n"
        "metric = MetricsConfigBuilderCustomNode(id=2, entry='/metric.py:f')\n"
        "benchmark = ModelEvalApiNode(id=3, endpoint=inference.endpoint)\n"
    )
    dataset_config, evaluation_config = _write_configs(tmp_path, mode="exact_match")

    errors = validator_module.validate_pipeline(
        workflow, dataset_config, evaluation_config
    )

    assert any("forbidden benchmark nodes" in error for error in errors)
    assert any("exactly one CustomCommandNode" in error for error in errors)
    assert "evaluation config mode must be agent_rubric" in errors


def test_accepts_cpu_command_for_existing_endpoint(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text(
        """evaluation = CustomCommandCPUNode(
    id=1,
    command=(
        "python3 /workspace/.pyromind-agent/run/evaluate_inference.py "
        "--endpoint https://inference.example/v1 --model deployed-model "
        "--dataset-path /workspace/datasets/test.jsonl "
        "--dataset-config /workspace/.pyromind-agent/run/dataset.json "
        "--evaluation-config /workspace/.pyromind-agent/run/evaluation.json "
        "--output-dir /workspace/eval/test --limit 0"
    ),
    cpu=4,
    memory=32,
)
"""
    )
    dataset_config, evaluation_config = _write_configs(tmp_path)

    errors = validator_module.validate_pipeline(
        workflow, dataset_config, evaluation_config
    )

    assert errors == []


def test_rejects_gpu_count_mismatch(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text(
        """evaluation = CustomCommandNode(
    id=1,
    command=(
        "python3 /workspace/.pyromind-agent/run/evaluate_inference.py "
        "--model-path /workspace/models/test --gpu-count 2 "
        "--dataset-path /workspace/datasets/test.jsonl "
        "--dataset-config /workspace/.pyromind-agent/run/dataset.json "
        "--evaluation-config /workspace/.pyromind-agent/run/evaluation.json "
        "--output-dir /workspace/eval/test --limit 0"
    ),
    cpu=4,
    memory=32,
    gpu_count=1,
    gpu_product="NVIDIA-L40S",
)
"""
    )
    dataset_config, evaluation_config = _write_configs(tmp_path)

    errors = validator_module.validate_pipeline(
        workflow, dataset_config, evaluation_config
    )

    assert "CustomCommandNode gpu_count must match --gpu-count" in errors


def test_rejects_storage_logical_paths_in_command(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text(
        """evaluation = CustomCommandNode(
    id=1,
    command=(
        "python3 /.pyromind-agent/run/evaluate_inference.py "
        "--model-path /workspace/models/test --gpu-count 1 "
        "--dataset-path /workspace/datasets/test.jsonl "
        "--dataset-config /.pyromind-agent/run/dataset.json "
        "--evaluation-config /.pyromind-agent/run/evaluation.json "
        "--output-dir /workspace/eval/test --limit 0"
    ),
    cpu=4,
    memory=32,
    gpu_count=1,
    gpu_product="NVIDIA-L40S",
)
"""
    )
    dataset_config, evaluation_config = _write_configs(tmp_path)

    errors = validator_module.validate_pipeline(
        workflow, dataset_config, evaluation_config
    )

    assert (
        "command must map /.pyromind-agent/... Storage paths to "
        "/workspace/.pyromind-agent/... container paths"
    ) in errors


def test_config_preflight_rejects_legacy_media_keys_and_invalid_rubric(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    dataset_config, evaluation_config = _write_configs(tmp_path)
    dataset_config.write_text(
        json.dumps(
            {
                "user_prompt_field": "prompt",
                "reference_field": "ground_truth",
                "image_field": "images",
                "image_root": "/workspace/datasets/test",
            }
        )
    )
    evaluation_config.write_text(
        json.dumps(
            {
                "mode": "agent_rubric",
                "rubrics": [
                    {
                        "name": "boxes",
                        "criterion": "Boxes match.",
                        "weight": 1,
                        "required": "yes",
                        "evaluator": {"type": "bbox_iou"},
                    }
                ],
            }
        )
    )

    errors = validator_module.validate_configs(dataset_config, evaluation_config)

    assert "dataset config uses legacy image_field; use media_field instead" in errors
    assert "dataset config uses legacy image_root; use media_base_dir instead" in errors
    assert "rubric 1 required must be boolean" in errors
    assert "rubric 1 evaluator requires prediction_path" in errors
    assert "rubric 1 evaluator requires reference_path" in errors
