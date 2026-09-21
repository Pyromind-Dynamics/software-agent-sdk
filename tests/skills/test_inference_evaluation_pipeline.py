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
SKILL_PATH = MODULE_PATH.parents[1] / "SKILL.md"
EVALUATOR_PATH = MODULE_PATH.parent / "evaluate_inference.py"


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


def test_accepts_vllm_and_cpu_agent_rubric_pipeline(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text(
        """inference = VLLMInference(
    id=1,
    model_path="/workspace/models/test",
    port=3000,
    max_model_len=8192,
    gpu_count=1,
    gpu_product="NVIDIA-L40S",
)

evaluation = CustomCommandCPUNode(
    id=2,
    command=(
        "python3 /workspace/.pyromind-agent/run/evaluate_inference.py "
        "--endpoint '$param' --model default "
        "--model-reference /workspace/models/test "
        "--dataset-path /workspace/datasets/test.jsonl "
        "--dataset-config /workspace/.pyromind-agent/run/dataset.json "
        "--evaluation-config /workspace/.pyromind-agent/run/evaluation.json "
        "--output-dir /workspace/eval/test --limit 0"
    ),
    cpu=4,
    memory=32,
    param=inference.endpoint,
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
    assert "workflow must contain exactly one CustomCommandCPUNode" in errors
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


def test_rejects_unbound_vllm_endpoint(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text(
        """inference = VLLMInference(
    id=1,
    model_path="/workspace/models/test",
    port=3000,
    gpu_count=1,
    gpu_product="NVIDIA-L40S",
)

evaluation = CustomCommandCPUNode(
    id=2,
    command=(
        "python3 /workspace/.pyromind-agent/run/evaluate_inference.py "
        "--endpoint http://127.0.0.1:3000/v1 "
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

    assert "CustomCommandCPUNode param must reference VLLMInference.endpoint" in errors
    assert (
        "CustomCommandCPUNode --endpoint must use the directly bound $param" in errors
    )


def test_rejects_shell_fallback_for_bound_endpoint(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text(
        """inference = VLLMInference(
    id=1,
    model_path="/workspace/models/test",
    port=3000,
    gpu_count=1,
    gpu_product="NVIDIA-L40S",
)

evaluation = CustomCommandCPUNode(
    id=2,
    command=(
        "python3 /workspace/.pyromind-agent/run/evaluate_inference.py "
        "--endpoint '${PARAM:-${param:-}}' "
        "--dataset-path /workspace/datasets/test.jsonl "
        "--dataset-config /workspace/.pyromind-agent/run/dataset.json "
        "--evaluation-config /workspace/.pyromind-agent/run/evaluation.json "
        "--output-dir /workspace/eval/test --limit 0"
    ),
    cpu=4,
    memory=32,
    param=inference.endpoint,
)
"""
    )
    dataset_config, evaluation_config = _write_configs(tmp_path)

    errors = validator_module.validate_pipeline(
        workflow, dataset_config, evaluation_config
    )

    assert (
        "CustomCommandCPUNode --endpoint must use the directly bound $param" in errors
    )


def test_rejects_storage_logical_paths_in_command(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text(
        """evaluation = CustomCommandCPUNode(
    id=1,
    command=(
        "python3 /.pyromind-agent/run/evaluate_inference.py "
        "--endpoint https://inference.example/v1 "
        "--dataset-path /workspace/datasets/test.jsonl "
        "--dataset-config /.pyromind-agent/run/dataset.json "
        "--evaluation-config /.pyromind-agent/run/evaluation.json "
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

    assert (
        "command must map /.pyromind-agent/... Storage paths to "
        "/workspace/.pyromind-agent/... container paths"
    ) in errors


def test_rejects_media_base_dir_that_repeats_dataset_directory(
    validator_module: ModuleType, tmp_path: Path
) -> None:
    workflow = tmp_path / "workflow.py"
    workflow.write_text(
        """evaluation = CustomCommandCPUNode(
    id=1,
    command=(
        "python3 /workspace/.pyromind-agent/run/evaluate_inference.py "
        "--endpoint https://inference.example/v1 "
        "--dataset-path /workspace/datasets/demo/eval.jsonl "
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
    config = json.loads(dataset_config.read_text())
    config.update(
        {
            "media_field": "images",
            "media_base_dir": "datasets/demo",
        }
    )
    dataset_config.write_text(json.dumps(config))

    errors = validator_module.validate_pipeline(
        workflow, dataset_config, evaluation_config
    )

    assert (
        "dataset config media_base_dir repeats the dataset directory; omit it "
        "when media paths are relative to the dataset file"
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


def test_skill_uses_available_python_and_stops_repeated_terminal_probes() -> None:
    skill = SKILL_PATH.read_text()

    assert (
        skill.count('python3 "$PYROMIND_SKILLS_PATH/inference-evaluation/scripts/') == 3
    )
    assert 'python "$PYROMIND_SKILLS_PATH/inference-evaluation/scripts/' not in skill
    assert "若两次终端调用返回相同的" in skill
    assert "立即停止终端探测" in skill
    assert "VLLMInference" in skill
    assert "param=inference.endpoint" in skill
    assert "禁止使用 `CustomCommandNode`" in skill


def test_evaluator_does_not_install_inference_runtime_dependencies() -> None:
    evaluator = EVALUATOR_PATH.read_text()

    assert "pip install" not in evaluator
    assert "_prepare_model_runtime" not in evaluator
