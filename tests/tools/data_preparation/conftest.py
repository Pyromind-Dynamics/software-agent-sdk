from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from pyromind_sdk.client.workflow import DslConverter

from pyromind_common.platform.evaluation_core import (
    EvaluationOptions,
    ValidationSelection,
)
from pyromind_common.platform.preparation import LocalFiles, StagingDirectory


@pytest.fixture
def sdk_files(tmp_path: Path) -> LocalFiles:
    root = tmp_path / "inputs"
    root.mkdir()
    return LocalFiles(root)


@pytest.fixture
def sdk_staging(tmp_path: Path) -> StagingDirectory:
    return StagingDirectory(tmp_path / "frozen", "/workspace/jobs/one")


@pytest.fixture
def sdk_run_id() -> UUID:
    return UUID("11111111-1111-1111-1111-111111111111")


@pytest.fixture
def sdk_model_profiles(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, str]]:
    profiles = {
        name: {
            "DF_API_KEY": f"test-{name}-model-key",
            "DF_API_URL": f"https://{name}.test/v1/chat/completions",
            "DF_API_BASE_URL": f"https://{name}.test/v1",
            "DF_MODEL_NAME": f"{name}-model",
        }
        for name in ("text", "vision")
    }
    monkeypatch.setenv("PYROMIND_DATAFLOW_PROFILES", json.dumps(profiles))
    return profiles


@pytest.fixture
def processing_scripts() -> Path:
    return (
        Path(__file__).resolve().parents[3] / ".agents/skills/data-processing/scripts"
    )


@pytest.fixture
def sdk_validator() -> Iterator[httpx.Client]:
    def validate(request: httpx.Request) -> httpx.Response:
        result: dict[str, Any] = {"valid": True}
        if request.url.path.endswith("dsl/validate"):
            payload = json.loads(request.content)
            result["workflow"] = DslConverter().from_python(
                payload["dsl"], name=payload["name"]
            )
        return httpx.Response(200, json={"success": True, "data": result})

    with httpx.Client(
        base_url="https://validation.test/workflows/",
        transport=httpx.MockTransport(validate),
    ) as client:
        yield client


@pytest.fixture
def sdk_validation_selection() -> ValidationSelection:
    return ValidationSelection.model_validate(
        {
            "selection_run_id": "selected-validation",
            "train": {"path": "/workspace/datasets/train.jsonl", "sha256": "a" * 64},
            "validation": {"path": "/workspace/datasets/val.jsonl", "sha256": "b" * 64},
            "test": {"path": "/workspace/datasets/test.jsonl", "sha256": "c" * 64},
            "model": {
                "path": "/workspace/models/checkpoint-100",
                "kind": "lora",
                "base_model": "/workspace/base",
            },
            "scoring_script": (
                "def parse_record(raw, context): return raw\n"
                "def score_case(case, spec): return {'passed': True}\n"
            ),
            "scoring_spec": {"business_rule": "frozen confirmed rule"},
        }
    )


@pytest.fixture
def sdk_held_out_options(
    sdk_files: LocalFiles, sdk_validation_selection: ValidationSelection
) -> EvaluationOptions:
    (sdk_files.root / "dataset.json").write_text(
        json.dumps({"user_prompt_field": "prompt", "reference_field": "answer"})
    )
    (sdk_files.root / "evaluation.json").write_text(
        json.dumps(
            {
                "mode": "agent_rubric",
                "model_source": {
                    "type": "checkpoint",
                    "kind": "lora",
                    "base_model": "/workspace/base",
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
    return EvaluationOptions(
        operation="inference",
        input_path="/workspace/datasets/test.jsonl",
        model_path="/workspace/models/checkpoint-100",
        selection_run_id=sdk_validation_selection.selection_run_id,
        dataset_config_path="dataset.json",
        evaluation_config_path="evaluation.json",
    )
