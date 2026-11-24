from uuid import uuid4

import pytest

from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.model_evaluation_plan import (
    model_evaluation_tasks,
    submission_paths,
    validate_evaluation_inputs,
)
from pyromind_runtime.application.plan_execution import (
    plan_executions,
    with_platform_states,
)
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.pipeline import PipelineRun
from pyromind_runtime.domain.preparation import ProjectView
from pyromind_runtime.domain.snapshot import ConversationSnapshot


def test_legacy_model_eval_and_local_analysis_have_distinct_completion_evidence():
    tasks = [
        {
            "id": "train",
            "kind": "train_run",
            "stage_id": "train",
            "name": "Train",
            "outputs": [{"name": "merged_model"}],
        },
        {
            "id": "evaluate",
            "kind": "checkpoint_eval",
            "stage_id": "eval",
            "name": "Evaluate",
            "inputs": [
                {"name": "model", "task_id": "train", "output_name": "merged_model"},
                {"name": "val_dataset", "resource_id": "val"},
            ],
        },
        {"id": "analyze", "kind": "analysis", "stage_id": "eval", "name": "Analyze"},
        {
            "id": "select",
            "kind": "checkpoint_eval",
            "stage_id": "eval",
            "name": "Select",
            "inputs": [{"name": "predictions", "resource_id": "pred"}],
        },
    ]
    project = ProjectView.model_validate(
        {
            "project_id": "project",
            "conversation_id": "chat",
            "plan_sequence": 1,
            "creation_command_id": "create",
            "plan": {
                "tasks": tasks,
                "stages": [{"id": k, "kind": k, "name": k} for k in ("train", "eval")],
                "resources": [
                    {
                        "id": "val",
                        "name": "Validation",
                        "storage_path": "/workspace/data/val.jsonl",
                    },
                    {"id": "pred", "name": "Training predictions"},
                ],
            },
        }
    )
    required = model_evaluation_tasks(
        project, ECHOMIND_PRODUCT_PROFILE.preparation_policy
    )
    assert required == {"evaluate"}
    rows = [
        {"id": t["id"], "stage_id": t["stage_id"], "business_status": "completed"}
        for t in tasks
    ]
    rows[0]["verified_outputs"] = {"merged_model": "/workspace/models/merged"}
    task = project.plan.tasks[1]
    dataset = "/data/val.jsonl"
    validate_evaluation_inputs(project, task, rows, "/models/merged", dataset)
    for model, data, error in [
        ("/models/base", dataset, "merged"),
        ("/models/merged", "/data/train.jsonl", "val.jsonl"),
    ]:
        with pytest.raises(ValueError, match=error):
            validate_evaluation_inputs(project, task, rows, model, data)
    with pytest.raises(ValueError, match="Verify train.merged_model"):
        validate_evaluation_inputs(project, task, [], "/models/merged", dataset)
    selected_task = task.model_copy(
        update={
            "inputs": tuple(
                item.model_copy(update={"output_name": "model_output"})
                if item.task_id
                else item
                for item in task.inputs
            )
        }
    )
    selected = "/workspace/models/sft/checkpoint-6"
    training = {
        **rows[0],
        "verified_outputs": {
            "model_output": "/workspace/models/sft",
            "selected_checkpoint": selected,
        },
    }
    validate_evaluation_inputs(project, selected_task, [training], selected, dataset)
    validate_evaluation_inputs(
        project, selected_task, [training], "/models/sft", dataset
    )
    for wrong in (
        "/workspace/models/sft/checkpoint-5",
        "/workspace/models/other/checkpoint-6",
    ):
        with pytest.raises(ValueError, match="requires"):
            validate_evaluation_inputs(
                project, selected_task, [training], wrong, dataset
            )
    with pytest.raises(ValueError, match="requires"):
        validate_evaluation_inputs(
            project, selected_task, [training], selected, "/data/train.jsonl"
        )
    score = PipelineRun(
        run_id=uuid4(),
        conversation_id="chat",
        task_id="1272",
        definition={"kind": "prediction_evaluation", "input_path": "/training"},
        fingerprint="test",
        output_dir="/result",
    )
    resume: JsonObject = {
        "operation": "resume",
        "resume_run_id": str(score.run_id),
        "input_path": dataset,
    }
    with pytest.raises(ValueError, match="inference"):
        submission_paths("df_submit_evaluation", resume, {str(score.run_id): score})
    runs = tuple(plan_executions(project, rows))
    snapshot = ConversationSnapshot(
        conversation_id="chat",
        capabilities=HarnessCapabilities(),
        experiment_executions=runs,
    )
    unbound = with_platform_states(snapshot, required).experiment_executions
    assert unbound and unbound[1].state == "needs_input"
    linked = runs[1].model_copy(update={"platform_task_id": "1272"})
    snapshot = snapshot.model_copy(
        update={"experiment_executions": (runs[0], linked, *runs[2:])}
    )
    views = with_platform_states(snapshot, required, [score]).experiment_executions
    assert views and [r.state for r in views] == [
        "succeeded",
        "needs_input",
        "succeeded",
        "succeeded",
    ]
    inference = score.model_copy(
        update={
            "definition": {
                "inference": {"model_path": "/models/merged"},
                "input_path": dataset,
            }
        }
    )
    assert submission_paths(
        "df_submit_evaluation", resume, {str(inference.run_id): inference}
    ) == ("/models/merged", dataset)
    completed = with_platform_states(
        snapshot, required, [inference]
    ).experiment_executions
    assert completed and completed[1].state == "succeeded"
