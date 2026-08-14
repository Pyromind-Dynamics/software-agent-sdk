import json

import pytest
from pydantic import ValidationError

from openhands.tools.update_plan import (
    PlanStep,
    UpdatePlanAction,
    UpdatePlanExecutor,
    UpdatePlanObservation,
    UpdatePlanTool,
)


def test_update_plan_replaces_and_persists_complete_snapshot(tmp_path) -> None:
    executor = UpdatePlanExecutor(str(tmp_path))
    executor(
        UpdatePlanAction(
            explanation="Start",
            plan=[
                PlanStep(step="Locate modules", status="in_progress"),
                PlanStep(step="Trace flow", status="pending"),
            ],
        )
    )

    observation = executor(
        UpdatePlanAction(
            plan=[
                PlanStep(step="Locate modules", status="completed"),
                PlanStep(step="Trace flow", status="in_progress"),
            ]
        )
    )

    assert [step.status for step in observation.plan] == [
        "completed",
        "in_progress",
    ]
    assert json.loads((tmp_path / "PLAN.json").read_text()) == {
        "explanation": None,
        "plan": [
            {"step": "Locate modules", "status": "completed"},
            {"step": "Trace flow", "status": "in_progress"},
        ],
    }
    assert UpdatePlanExecutor(str(tmp_path)).plan == observation.plan


def test_update_plan_rejects_incremental_patch_and_multiple_active_steps() -> None:
    with pytest.raises(ValidationError):
        UpdatePlanAction.model_validate({"index": 0, "newStatus": "completed"})

    with pytest.raises(ValidationError, match="At most one"):
        UpdatePlanAction(
            plan=[
                PlanStep(step="One", status="in_progress"),
                PlanStep(step="Two", status="in_progress"),
            ]
        )


def test_update_plan_normalizes_single_nested_argument_wrapper() -> None:
    tool = UpdatePlanTool(
        description="test",
        action_type=UpdatePlanAction,
        observation_type=UpdatePlanObservation,
    )
    malformed = {
        "summary": "Prepare the data pipeline",
        "plan": [
            {
                "explanation": "",
                "plan": [
                    {"step": "Prepare df_submit_pipeline", "status": "in_progress"}
                ],
                "status": "in_progress",
            }
        ],
    }

    assert tool.normalize_arguments(malformed) == {
        "summary": "Prepare the data pipeline",
        "explanation": "",
        "plan": [{"step": "Prepare df_submit_pipeline", "status": "in_progress"}],
    }


def test_update_plan_leaves_ambiguous_nested_wrapper_unchanged() -> None:
    tool = UpdatePlanTool(
        description="test",
        action_type=UpdatePlanAction,
        observation_type=UpdatePlanObservation,
    )
    malformed = {
        "plan": [
            {
                "plan": [
                    {"step": "One", "status": "in_progress"},
                    {"step": "Two", "status": "in_progress"},
                ]
            }
        ]
    }

    assert tool.normalize_arguments(malformed) is malformed
