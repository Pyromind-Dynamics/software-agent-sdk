from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from pyromind_sdk.client.workflow import DslConverter

from pyromind_common.platform.evaluation import stage_evaluation
from pyromind_common.platform.evaluation_core import EvaluationOptions
from pyromind_common.platform.preparation import LocalFiles, StagingDirectory


@pytest.mark.parametrize(
    ("architecture", "loader"), [("qwen3_vl", "qwen3vl"), ("qwen3_5", "qwen3.5")]
)
def test_sdk_lora_staging_uses_base_architecture_and_one_merge_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, architecture: str, loader: str
):
    (tmp_path / "dataset.json").write_text(
        json.dumps({"user_prompt_field": "prompt", "reference_field": "answer"})
    )
    (tmp_path / "evaluation.json").write_text(
        json.dumps(
            {
                "mode": "agent_rubric",
                "model_source": {
                    "type": "checkpoint",
                    "kind": "lora",
                    "base_model": "/workspace/models/internal-alias",
                },
                "rubrics": [
                    {
                        "name": "exact",
                        "criterion": "match",
                        "weight": 1,
                        "evaluator": {"type": "exact_match"},
                    }
                ],
            }
        )
    )
    original_read_text = LocalFiles.read_text
    config_reads = []

    def read_text(files: LocalFiles, path: str) -> str:
        if path == "/workspace/models/internal-alias/config.json":
            config_reads.append(path)
            return json.dumps({"model_type": architecture})
        return original_read_text(files, path)

    monkeypatch.setattr(LocalFiles, "read_text", read_text)
    options = EvaluationOptions(
        operation="inference",
        input_path="/workspace/datasets/val.jsonl",
        model_path="/workspace/train/checkpoint-100",
        dataset_config_path="dataset.json",
        evaluation_config_path="evaluation.json",
        gpu_count=2,
    )
    staged = stage_evaluation(
        options,
        files=LocalFiles(tmp_path),
        staging=StagingDirectory(tmp_path / "frozen", "/workspace/jobs/evaluation"),
        runtime_dir=Path(__file__).resolve().parents[3]
        / ".agents/skills/inference-evaluation/scripts",
        run_id=uuid4(),
    )
    graph = DslConverter().from_python(staged.dsl, name=staged.name)
    configs = {
        node["data"]["nodeType"]: node["data"]["config"] for node in graph["nodes"]
    }
    assert config_reads == ["/workspace/models/internal-alias/config.json"]
    assert configs["ModelMergeLoraNode"]["model_type"] == loader
    assert configs["ModelMergeLoraNode"]["gpu_count"] == 1
    assert configs["VLLMInference"]["gpu_count"] == 2
