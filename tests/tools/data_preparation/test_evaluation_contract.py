"""The model-facing contract must keep script execution and evaluation separate."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from openhands.tools.data_preparation.evaluation_submit import (
    DfSubmitEvaluationAction,
    DfSubmitEvaluationExecutor,
    DfSubmitEvaluationTool,
)
from openhands.tools.data_preparation.platform_submit import (
    DfSubmitPipelineAction,
    DfSubmitPipelineTool,
)


def test_original_stuck_call_is_rejected_before_evaluation_execution():
    with pytest.raises(ValidationError, match="operation"):
        DfSubmitEvaluationAction.model_validate(
            {
                "input_path": "/shence_canvas_sft/eval_baseline.jsonl",
                "mode": "full",
                "cpu": 8,
                "memory": 48,
            }
        )


@pytest.mark.parametrize(
    "operation,options",
    [
        (
            "inference",
            {
                "model_path": "/models/Qwen/Qwen3-32B",
                "dataset_config_path": "public_data/dataset.json",
                "evaluation_config_path": "public_data/evaluation.json",
            },
        ),
        (
            "score",
            {
                "source_manifest_path": "public_data/source.json",
                "scoring_spec_path": "public_data/spec.json",
                "scoring_script_path": "public_data/score.py",
            },
        ),
        (
            "profile",
            {
                "manifest_path": "public_data/profile.json",
                "extraction_script_path": "public_data/extract.py",
            },
        ),
        ("review", {"manifest_path": "public_data/review.json"}),
        ("report", {"manifest_path": "public_data/report.json"}),
        ("resume", {"resume_run_id": str(uuid4())}),
    ],
)
def test_each_operation_requires_its_own_fields(operation, options):
    payload = {"operation": operation, "input_path": "/data/test.jsonl", **options}
    action = DfSubmitEvaluationAction.model_validate(payload)
    request = action.request()
    assert request.cpu == 4 and request.memory == 32
    assert request.resource_overrides == frozenset()
    for field in options:
        incomplete = {key: value for key, value in payload.items() if key != field}
        with pytest.raises(ValidationError, match=field):
            DfSubmitEvaluationAction.model_validate(incomplete)


@pytest.mark.parametrize("field", ["inference", "evaluation", "products", "mode"])
def test_new_tool_rejects_legacy_groups(field):
    with pytest.raises(ValidationError, match=field):
        DfSubmitEvaluationAction.model_validate(
            {
                "operation": "report",
                "input_path": "/data/test.jsonl",
                "manifest_path": "public_data/report.json",
                field: {},
            }
        )


@pytest.mark.parametrize(
    "operation,options,unexpected",
    [
        ("report", {"manifest_path": "report.json"}, {"model_path": "/model"}),
        ("review", {"manifest_path": "review.json"}, {"gpu_count": 1}),
        (
            "report",
            {"manifest_path": "report.json"},
            {"extraction_script_path": "extract.py"},
        ),
        (
            "resume",
            {"resume_run_id": str(uuid4())},
            {"dataset_config_path": "dataset.json"},
        ),
    ],
)
def test_operations_cannot_silently_accept_other_operation_options(
    operation, options, unexpected
):
    with pytest.raises(ValidationError, match="does not accept"):
        DfSubmitEvaluationAction.model_validate(
            {"operation": operation, "input_path": "/data", **options, **unexpected}
        )


def test_exported_schemas_are_independent_and_operation_is_required():
    script = DfSubmitPipelineTool.create()[0].to_mcp_tool()["inputSchema"]
    evaluation = DfSubmitEvaluationTool.create()[0].to_mcp_tool()["inputSchema"]
    assert {"operation", "input_path"} <= set(evaluation["required"])
    assert "model_path" in evaluation["properties"]
    assert not {"inference", "evaluation", "products", "operation", "model_path"} & (
        script["properties"].keys()
    )
    assert not {"script_path", "model_profile", "output_schema", "mode"} & (
        evaluation["properties"].keys()
    )


def test_script_full_and_resume_have_distinct_required_options():
    for payload, missing in [
        ({"input_path": "/data"}, "script_path"),
        ({"input_path": "/data", "mode": "resume"}, "resume_run_id"),
    ]:
        with pytest.raises(ValidationError, match=missing):
            DfSubmitPipelineAction.model_validate(payload)
    with pytest.raises(ValidationError, match="inference"):
        DfSubmitPipelineAction.model_validate(
            {"input_path": "/data", "script_path": "pipeline.py", "inference": {}}
        )


def test_evaluation_executor_never_needs_a_script_executor():
    executor = DfSubmitEvaluationExecutor()
    action = DfSubmitEvaluationAction(
        operation="report", input_path="/data", manifest_path="report.json"
    )
    assert executor(action).is_error  # No active conversation, no side effects.
    assert not hasattr(executor, "submission_handler")
