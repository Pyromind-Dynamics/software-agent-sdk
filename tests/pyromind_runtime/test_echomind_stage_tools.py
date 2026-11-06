import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.echomind_stage_tools import (
    ECHOMIND_STAGE_TOOL_SPECS,
    echomind_stage_tool_handler,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_plans import PlanRevision, WorkflowPlan

from openhands.agent_server.agent_profiles.echomind_manager import (
    ECHOMIND_MANAGER_DEVELOPMENT_TOOLS,
)
from openhands.agent_server.echomind_projects import ProjectStore
from openhands.agent_server.echomind_stage_runs import ProjectRunStore


def _plan() -> WorkflowPlan:
    return WorkflowPlan.model_validate(
        {
            "skill": "experiment-planning",
            "name": "AVI experiment",
            "goal": "Validate Dataset stage execution.",
            "resources": [
                {
                    "id": "images",
                    "name": "AVI images",
                    "uri": "/datasets/pcb_avi_eval_3/",
                }
            ],
            "stages": [
                {"id": "dataset", "name": "Dataset"},
                {"id": "inference", "name": "Inference"},
                {"id": "evaluator", "name": "Evaluator"},
                {"id": "train", "name": "Train"},
            ],
            "edges": [
                {"source": "dataset", "target": "inference"},
                {"source": "inference", "target": "evaluator"},
                {"source": "evaluator", "target": "train"},
            ],
        }
    )


def _sample_details(task_run_id: str = "task-dataset-sample") -> dict:
    return {
        "sample_count": 3,
        "image_integrity": {"status": "passed"},
        "bbox_validation": {"status": "not_applicable"},
        "schema_validation": {"status": "passed"},
        "schema_differences": [],
        "pipeline": f"public_data/experiment-runs/{task_run_id}/pipeline.py",
        "sample_output": "public_data/avi/sample.jsonl",
        "submission_spec": {
            "pipeline_path": (f"public_data/experiment-runs/{task_run_id}/pipeline.py"),
            "input_path": "/datasets/pcb_avi_eval_3/",
            "output_schema": "vision",
            "model_profile": "none",
            "convert_format": "none",
        },
        "checks": {
            "formal_validation": {
                "tool": (
                    ".agents/skills/data-processing/scripts/preparation/"
                    "validate_prepared_data.py"
                ),
                "schema": "vision",
                "status": "passed",
                "rows": 3,
                "command": (
                    "python3 .agents/skills/data-processing/scripts/preparation/"
                    "validate_prepared_data.py public_data/avi/sample.jsonl "
                    "--schema vision --image-root "
                    "/target-workspace/datasets/pcb_avi_eval_3 "
                    "--report public_data/avi/validation.json"
                ),
                "report_uri": "public_data/avi/validation.json",
                "output_sha256": "a" * 64,
            }
        },
    }


async def test_stage_tool_handler_records_dataset_sample_for_owning_conversation(
    tmp_path,
):
    conversations = tmp_path / "conversations"
    project_store = ProjectStore.for_conversations_dir(conversations)
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run, _ = ProjectRunStore(project_store.root).start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )

    async def request_stage_task(conversation_id, request, context):
        return {"id": "task-dataset-sample"}

    handle = echomind_stage_tool_handler(
        conversations,
        request_stage_task=request_stage_task,
    )
    context = RequestContext(user_id="42")

    approved = await handle(
        "conversation-1",
        "approve_project_stage_plan",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "approve-1",
            "summary": "Dataset schema and acceptance criteria confirmed.",
            "decisions": {"schema": "vision"},
        },
        context,
    )
    await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "dataset",
            "request_id": "dataset-sample",
            "task": {
                "name": "Validate sample",
                "instruction": "Validate the configured sample.",
                "inputs": {"phase": "sample"},
            },
        },
        context,
    )
    updated = await handle(
        "conversation-1",
        "record_project_stage_result",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "sample-1",
            "outcome": "waiting_user",
            "message": "Three samples completed.",
            "input_refs": ["/datasets/pcb_avi_eval_3/"],
            "task_run_id": "task-dataset-sample",
            "details": _sample_details(),
        },
        context,
    )
    loaded = await handle(
        "conversation-1",
        "get_project_stage_run",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
        },
        context,
    )

    assert run.status == "clarifying"
    assert approved["status"] == "running"
    assert approved["updates"][0]["details"]["decisions"] == {"schema": "vision"}
    assert updated["status"] == "waiting_user"
    assert updated["artifacts"][0]["kind"] == "stage_sample_report"
    assert loaded == updated

    with pytest.raises(PermissionError, match="another conversation"):
        await handle(
            "conversation-2",
            "get_project_stage_run",
            {
                "project_id": project.project.project_id,
                "run_id": run.run_id,
            },
            context,
        )


async def test_stage_tool_rejects_request_id_reused_for_another_status(tmp_path):
    conversations = tmp_path / "conversations"
    project_store = ProjectStore.for_conversations_dir(conversations)
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run, _ = ProjectRunStore(project_store.root).start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )

    async def request_stage_task(conversation_id, request, context):
        del conversation_id, request, context
        return {"id": "task-dataset-sample"}

    handle = echomind_stage_tool_handler(
        conversations,
        request_stage_task=request_stage_task,
    )
    context = RequestContext(user_id="42")
    request_id = f"project-created:{project.project.project_id}"
    await handle(
        "conversation-1",
        "approve_project_stage_plan",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": request_id,
            "summary": "Dataset scope confirmed.",
            "decisions": {},
        },
        context,
    )
    await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "dataset",
            "request_id": "dataset-sample",
            "task": {
                "name": "Validate sample",
                "instruction": "Validate the configured sample.",
                "inputs": {"phase": "sample"},
            },
        },
        context,
    )

    with pytest.raises(ValueError, match="already used for status 'running'"):
        await handle(
            "conversation-1",
            "record_project_stage_result",
            {
                "project_id": project.project.project_id,
                "run_id": run.run_id,
                "request_id": request_id,
                "outcome": "waiting_user",
                "message": "Sample passed.",
                "task_run_id": "task-dataset-sample",
                "details": _sample_details(),
            },
            context,
        )

    unchanged = await handle(
        "conversation-1",
        "get_project_stage_run",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
        },
        context,
    )
    assert unchanged["status"] == "running"
    assert unchanged["artifacts"] == []


async def test_open_project_stage_creates_or_reuses_node_run(tmp_path):
    conversations = tmp_path / "conversations"
    project_store = ProjectStore.for_conversations_dir(conversations)
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    handle = echomind_stage_tool_handler(conversations)
    context = RequestContext(user_id="42")
    arguments = {
        "project_id": project.project.project_id,
        "node_id": "dataset",
        "request_id": "project-created:" + project.project.project_id,
    }

    opened = await handle(
        "conversation-1",
        "open_project_stage",
        arguments,
        context,
    )
    repeated = await handle(
        "conversation-1",
        "open_project_stage",
        arguments,
        context,
    )

    assert opened == repeated
    assert opened["project_id"] == project.project.project_id
    assert opened["node_id"] == "dataset"
    assert opened["request_id"] == arguments["request_id"]
    assert opened["status"] == "clarifying"
    assert len(run_store.list(project.project.project_id)) == 1

    with pytest.raises(PermissionError, match="another conversation"):
        await handle(
            "conversation-2",
            "open_project_stage",
            arguments,
            context,
        )


async def test_stage_task_request_links_task_run_to_stage_run(tmp_path):
    conversations = tmp_path / "conversations"
    project_store = ProjectStore.for_conversations_dir(conversations)
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    captured = []
    events = []

    async def request_stage_task(conversation_id, request, context):
        events.append(("request", request.request_id))
        captured.append((conversation_id, request, context.user_id))
        return {
            "id": f"task-{request.request_id}",
            "project_id": project.project.project_id,
            "created_at": datetime(2026, 11, 4, tzinfo=UTC),
        }

    async def cancel_stage_task(conversation_id, task_run_id, command_id, context):
        events.append(("cancel", task_run_id))
        return {"run_id": task_run_id, "state": "cancel_requested"}

    handle = echomind_stage_tool_handler(
        conversations,
        request_stage_task=request_stage_task,
        cancel_stage_task=cancel_stage_task,
    )
    context = RequestContext(user_id="42")
    await handle(
        "conversation-1",
        "approve_project_stage_plan",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "approve-1",
            "summary": "Dataset scope confirmed.",
            "decisions": {},
        },
        context,
    )

    result = await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "dataset",
            "request_id": "dataset-check",
            "task": {
                "name": "Check dataset",
                "instruction": "Validate the configured dataset.",
                "inputs": {"phase": "sample"},
            },
            "intent": "execute",
        },
        context,
    )
    repeated = await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "dataset",
            "request_id": "dataset-check",
            "task": {
                "name": "Check dataset",
                "instruction": "Validate the configured dataset.",
                "inputs": {"phase": "sample"},
            },
            "intent": "execute",
        },
        context,
    )
    second = await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "dataset",
            "request_id": "dataset-sample-2",
            "task": {
                "name": "Validate another sample",
                "instruction": "Validate another configured sample.",
                "inputs": {"phase": "sample"},
            },
            "intent": "execute",
        },
        context,
    )

    assert len(captured) == 2
    assert events == [
        ("request", "dataset-check"),
        ("cancel", "task-dataset-check"),
        ("request", "dataset-sample-2"),
    ]
    assert captured[0][0] == "conversation-1"
    assert captured[0][1].stage_id == "dataset"
    assert captured[0][1].stage_run_id == run.run_id
    assert captured[0][1].role_id == "dataset"
    assert captured[0][1].intent == "execute"
    assert captured[0][1].task is not None
    assert captured[0][1].task.instruction == "Validate the configured dataset."
    assert captured[0][1].task.inputs["phase"] == "sample"
    assert captured[0][1].task.inputs["path_contract"]["storage_mount"] == (
        "/target-workspace"
    )
    assert captured[0][2] == "42"
    assert result["task_run_id"] == "task-dataset-check"
    assert repeated == {
        "id": "task-dataset-check",
        "project_id": project.project.project_id,
        "stage_run_id": run.run_id,
        "task_run_id": "task-dataset-check",
        "idempotent": True,
    }
    assert second["task_run_id"] == "task-dataset-sample-2"
    json.dumps(result)
    reloaded = run_store.get(
        project_id=project.project.project_id,
        run_id=run.run_id,
    )
    assert reloaded.task_run_id == "task-dataset-sample-2"
    assert [item.task_run_id for item in reloaded.task_runs] == [
        "task-dataset-check",
        "task-dataset-sample-2",
    ]
    assert [item.status for item in reloaded.task_runs] == [
        "cancelled",
        "running",
    ]


async def test_stage_task_redispatch_aborts_when_cancel_fails(tmp_path):
    conversations = tmp_path / "conversations"
    project_store = ProjectStore.for_conversations_dir(conversations)
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    requested = []

    async def request_stage_task(conversation_id, request, context):
        del conversation_id, context
        requested.append(request.request_id)
        return {"id": f"task-{request.request_id}"}

    async def cancel_stage_task(conversation_id, task_run_id, command_id, context):
        del conversation_id, task_run_id, command_id, context
        raise ValueError("submission acceptance is uncertain")

    handle = echomind_stage_tool_handler(
        conversations,
        request_stage_task=request_stage_task,
        cancel_stage_task=cancel_stage_task,
    )
    context = RequestContext(user_id="42")
    await handle(
        "conversation-1",
        "approve_project_stage_plan",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "approve-1",
            "summary": "Dataset scope confirmed.",
            "decisions": {},
        },
        context,
    )
    await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "dataset",
            "request_id": "dataset-sample-initial",
            "task": {
                "name": "Validate sample",
                "instruction": "Validate the configured sample.",
                "inputs": {"phase": "sample"},
            },
        },
        context,
    )

    with pytest.raises(ValueError, match="submission acceptance is uncertain"):
        await handle(
            "conversation-1",
            "request_project_stage_task",
            {
                "project_id": project.project.project_id,
                "run_id": run.run_id,
                "role_id": "dataset",
                "request_id": "dataset-sample-redo",
                "task": {
                    "name": "Redo sample",
                    "instruction": "Validate the sample again.",
                    "inputs": {"phase": "sample"},
                },
            },
            context,
        )

    assert requested == ["dataset-sample-initial"]
    reloaded = run_store.get(
        project_id=project.project.project_id,
        run_id=run.run_id,
    )
    assert [item.status for item in reloaded.task_runs] == ["running"]


async def test_confirm_waiting_stage_run_allows_new_full_task(tmp_path):
    conversations = tmp_path / "conversations"
    project_store = ProjectStore.for_conversations_dir(conversations)
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )

    captured = []

    async def request_stage_task(conversation_id, request, context):
        captured.append(request)
        return {"id": f"task-{request.request_id}"}

    handle = echomind_stage_tool_handler(
        conversations,
        request_stage_task=request_stage_task,
    )
    context = RequestContext(user_id="42")
    await handle(
        "conversation-1",
        "approve_project_stage_plan",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "approve-1",
            "summary": "Dataset scope confirmed.",
            "decisions": {},
        },
        context,
    )
    await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "dataset",
            "request_id": "dataset-sample",
            "task": {
                "name": "Validate sample",
                "instruction": "Validate the configured dataset sample.",
                "inputs": {"phase": "sample"},
            },
        },
        context,
    )
    run, _, _ = run_store.bind_task_run(
        run,
        request_id="dataset-sample",
        task_run_id="task-dataset-sample",
        role_id="dataset",
    )
    await handle(
        "conversation-1",
        "record_project_stage_result",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "sample-report",
            "outcome": "waiting_user",
            "message": "Sample is ready for confirmation.",
            "task_run_id": "task-dataset-sample",
            "details": _sample_details(),
        },
        context,
    )

    with pytest.raises(ValueError, match="running or failed"):
        await handle(
            "conversation-1",
            "request_project_stage_task",
            {
                "project_id": project.project.project_id,
                "run_id": run.run_id,
                "role_id": "dataset",
                "request_id": "dataset-full-too-early",
                "task": {
                    "name": "Process full dataset",
                    "instruction": "Process the full dataset.",
                    "inputs": {"phase": "execute"},
                },
            },
            context,
        )

    confirmed = await handle(
        "conversation-1",
        "confirm_project_stage_run",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "confirm-sample",
        },
        context,
    )
    full = await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "dataset",
            "request_id": "dataset-full",
            "task": {
                "name": "Process full dataset",
                "instruction": "Process the full dataset.",
                "inputs": {"phase": "submit"},
            },
        },
        context,
    )

    assert confirmed["status"] == "running"
    assert full["task_run_id"] == "task-dataset-full"
    assert captured[-1].task is not None
    assert captured[-1].task.name == "Submit the accepted Dataset pipeline"
    assert "Call df_submit_pipeline exactly once" in captured[-1].task.instruction
    assert "Process the full dataset." not in captured[-1].task.instruction
    assert captured[-1].task.inputs["submission_spec"]["pipeline_path"] == (
        "public_data/experiment-runs/task-dataset-sample/pipeline.py"
    )


async def test_dataset_sample_can_be_repaired_within_the_same_stage_run(tmp_path):
    conversations = tmp_path / "conversations"
    project_store = ProjectStore.for_conversations_dir(conversations)
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )

    async def request_stage_task(conversation_id, request, context):
        return {"id": f"task-{request.request_id}"}

    handle = echomind_stage_tool_handler(
        conversations,
        request_stage_task=request_stage_task,
    )
    context = RequestContext(user_id="42")
    await handle(
        "conversation-1",
        "approve_project_stage_plan",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "approve-1",
            "summary": "Dataset scope confirmed.",
            "decisions": {},
        },
        context,
    )
    await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "dataset",
            "request_id": "dataset-sample-initial",
            "task": {
                "name": "Validate sample",
                "instruction": "Validate the configured sample.",
                "inputs": {"phase": "sample"},
            },
        },
        context,
    )
    await handle(
        "conversation-1",
        "record_project_stage_result",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "sample-initial-result",
            "outcome": "waiting_user",
            "message": "Initial sample completed.",
            "task_run_id": "task-dataset-sample-initial",
            "details": _sample_details("task-dataset-sample-initial"),
        },
        context,
    )

    repaired = await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "dataset",
            "request_id": "dataset-sample-repair",
            "task": {
                "name": "Repair sample",
                "instruction": "Produce the missing reusable pipeline contract.",
                "inputs": {"phase": "sample"},
            },
        },
        context,
    )
    reloaded = run_store.get(
        project_id=project.project.project_id,
        run_id=run.run_id,
    )

    assert repaired["task_run_id"] == "task-dataset-sample-repair"
    assert reloaded.status == "running"
    assert [task.task_run_id for task in reloaded.task_runs] == [
        "task-dataset-sample-initial",
        "task-dataset-sample-repair",
    ]


async def test_stage_task_request_resolves_evaluator_alias_and_rejects_other_roles(
    tmp_path,
):
    conversations = tmp_path / "conversations"
    project_store = ProjectStore.for_conversations_dir(conversations)
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    evaluator = next(node for node in project.graph.nodes if node.id == "evaluator")
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=evaluator,
        request_id="start-evaluator",
    )
    captured = []

    async def request_stage_task(conversation_id, request, context):
        captured.append((conversation_id, request, context.user_id))
        return {"id": f"task-{request.request_id}"}

    handle = echomind_stage_tool_handler(
        conversations,
        request_stage_task=request_stage_task,
    )
    context = RequestContext(user_id="42")
    await handle(
        "conversation-1",
        "approve_project_stage_plan",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "approve-evaluator",
            "summary": "Evaluator scope confirmed.",
            "decisions": {},
        },
        context,
    )
    with pytest.raises(ValueError, match="does not match"):
        await handle(
            "conversation-1",
            "request_project_stage_task",
            {
                "project_id": project.project.project_id,
                "run_id": run.run_id,
                "role_id": "inference",
                "request_id": "wrong-role",
                "task": {
                    "name": "Evaluate",
                    "instruction": "Score existing predictions.",
                },
            },
            context,
        )

    result = await handle(
        "conversation-1",
        "request_project_stage_task",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "role_id": "evaluator",
            "request_id": "evaluate",
            "task": {
                "name": "Evaluate",
                "instruction": "Score existing predictions.",
            },
        },
        context,
    )

    assert captured[0][1].role_id == "eval"
    assert result["task_run_id"] == "task-evaluate"
    reloaded = run_store.get(
        project_id=project.project.project_id,
        run_id=run.run_id,
    )
    assert reloaded.task_runs[0].role_id == "eval"


async def test_all_four_development_nodes_delegate_to_their_execution_roles(
    tmp_path,
):
    conversations = tmp_path / "conversations"
    project_store = ProjectStore.for_conversations_dir(conversations)
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    captured = []

    async def request_stage_task(conversation_id, request, context):
        captured.append((conversation_id, request, context.user_id))
        return {"id": f"task-{request.request_id}"}

    handle = echomind_stage_tool_handler(
        conversations,
        request_stage_task=request_stage_task,
    )
    context = RequestContext(user_id="42")

    for node in project.graph.nodes:
        run = await handle(
            "conversation-1",
            "open_project_stage",
            {
                "project_id": project.project.project_id,
                "node_id": node.id,
                "request_id": f"start-{node.id}",
            },
            context,
        )
        await handle(
            "conversation-1",
            "approve_project_stage_plan",
            {
                "project_id": project.project.project_id,
                "run_id": run["run_id"],
                "request_id": f"approve-{node.id}",
                "summary": f"{node.title} scope confirmed.",
                "decisions": {},
            },
            context,
        )
        await handle(
            "conversation-1",
            "request_project_stage_task",
            {
                "project_id": project.project.project_id,
                "run_id": run["run_id"],
                "role_id": node.execution_role,
                "request_id": f"execute-{node.id}",
                "task": {
                    "name": f"Execute {node.title}",
                    "instruction": f"Complete the {node.title} node.",
                    "inputs": {
                        "phase": "sample" if node.id == "dataset" else "execute"
                    },
                },
            },
            context,
        )

    assert [request.role_id for _, request, _ in captured] == [
        "dataset",
        "inference",
        "eval",
        "train",
    ]
    assert [request.stage_id for _, request, _ in captured] == [
        "dataset",
        "inference",
        "evaluator",
        "train",
    ]


async def test_pi_adapter_exposes_extension_tools_only_to_allowed_profile(tmp_path):
    adapter = PiAdapter(tmp_path / "conversations", terminal_backend="os-sandbox")
    adapter.set_tool_extensions(
        ECHOMIND_STAGE_TOOL_SPECS,
        echomind_stage_tool_handler(tmp_path / "conversations"),
    )
    names = {spec.name for spec in ECHOMIND_STAGE_TOOL_SPECS}
    echomind_session = SimpleNamespace(
        config={"tool_allowlist": tuple(names)},
    )
    workbench_session = SimpleNamespace(config={"tool_allowlist": ("read",)})

    assert names == {
        spec["name"] for spec in adapter._tool_specs(cast(Any, echomind_session))
    }
    assert "open_project_stage" in names
    assert not names.intersection(
        spec["name"] for spec in adapter._tool_specs(cast(Any, workbench_session))
    )


async def test_project_stage_tools_do_not_materialize_sandbox_context(
    tmp_path, monkeypatch
):
    adapter = PiAdapter(tmp_path / "conversations", terminal_backend="sandbox")
    captured: list[tuple[str, dict[str, Any]]] = []

    async def handle(
        _conversation_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        _context: RequestContext,
    ) -> dict[str, Any]:
        captured.append((tool_name, arguments))
        return {"status": "clarifying"}

    adapter.set_tool_extensions(ECHOMIND_STAGE_TOOL_SPECS, handle)

    async def fail_tool_context(_session):
        raise AssertionError("local stage tools must not resolve a sandbox workspace")

    monkeypatch.setattr(adapter, "_tool_context", fail_tool_context)
    session = SimpleNamespace(
        session_id="conversation-1",
        config={"tool_allowlist": ("get_project_stage_run",)},
        context=RequestContext(user_id="42"),
    )
    arguments = {"project_id": "project-1", "run_id": "run-1"}

    result = await adapter._runner_request(
        cast(Any, session),
        "tool.execute",
        {"tool_name": "get_project_stage_run", "arguments": arguments},
    )

    assert captured == [("get_project_stage_run", arguments)]
    assert result["details"] == {"status": "clarifying"}


def test_echomind_manager_development_session_exposes_dataset_submission_tools(
    tmp_path,
):
    adapter = PiAdapter(tmp_path / "conversations", terminal_backend="os-sandbox")
    adapter.set_tool_extensions(
        ECHOMIND_STAGE_TOOL_SPECS,
        echomind_stage_tool_handler(tmp_path / "conversations"),
    )
    session = SimpleNamespace(
        config={"tool_allowlist": ECHOMIND_MANAGER_DEVELOPMENT_TOOLS}
    )

    names = {spec["name"] for spec in adapter._tool_specs(cast(Any, session))}

    assert {
        "open_project_stage",
        "df_submit_pipeline",
        "df_check_progress",
        "get_project_stage_run",
        "request_project_stage_task",
        "approve_project_stage_plan",
        "confirm_project_stage_run",
        "record_project_stage_result",
    } <= names


async def test_stage_tool_records_platform_submission_and_version(tmp_path):
    conversations = tmp_path / "conversations"
    project_store = ProjectStore.for_conversations_dir(conversations)
    project = project_store.bind_plan(
        PlanRevision(
            project_id=uuid4().hex,
            revision=1,
            plan=_plan(),
            confirmed_by="42",
        ),
        echomind_conversation_id="conversation-1",
        owner_user_id="42",
    )
    run_store = ProjectRunStore(project_store.root)
    run, _ = run_store.start_agent_development(
        project=project,
        node=project.graph.nodes[0],
        request_id="start-dataset",
    )
    handle = echomind_stage_tool_handler(conversations)
    context = RequestContext(user_id="42")
    await handle(
        "conversation-1",
        "approve_project_stage_plan",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "approve-1",
            "summary": "Dataset schema and acceptance criteria confirmed.",
            "decisions": {"schema": "vision"},
        },
        context,
    )
    run, _, _ = run_store.bind_task_run(
        run,
        request_id="dataset-sample",
        task_run_id="task-dataset-sample",
        role_id="dataset",
    )
    await handle(
        "conversation-1",
        "record_project_stage_result",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "sample-1",
            "outcome": "waiting_user",
            "message": "Sample passed.",
            "input_refs": ["/datasets/pcb_avi_eval_3/"],
            "details": _sample_details(),
            "task_run_id": "task-dataset-sample",
        },
        context,
    )
    confirmed = await handle(
        "conversation-1",
        "confirm_project_stage_run",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "confirm-1",
        },
        context,
    )
    assert confirmed["status"] == "running"
    run = run_store.get(
        project_id=project.project.project_id,
        run_id=run.run_id,
    )
    run_store.bind_task_run(
        run,
        request_id="dataset-full",
        task_run_id="task-dataset-full",
        role_id="dataset",
    )

    with pytest.raises(ValueError, match="waiting_external"):
        await handle(
            "conversation-1",
            "record_project_stage_result",
            {
                "project_id": project.project.project_id,
                "run_id": run.run_id,
                "request_id": "complete-too-early",
                "outcome": "succeeded",
                "message": "Dataset version verified without platform submission.",
                "task_run_id": "task-dataset-full",
                "details": {},
            },
            context,
        )

    submitted = await handle(
        "conversation-1",
        "record_project_stage_result",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "submit-1",
            "outcome": "waiting_external",
            "message": "Platform task submitted.",
            "input_refs": ["/datasets/pcb_avi_eval_3/"],
            "details": {},
            "external_task_id": "task-1",
            "task_run_id": "task-dataset-full",
            "output_uri": "/runs/dataset-1",
        },
        context,
    )
    completed = await handle(
        "conversation-1",
        "record_project_stage_result",
        {
            "project_id": project.project.project_id,
            "run_id": run.run_id,
            "request_id": "complete-1",
            "outcome": "succeeded",
            "message": "Dataset version verified.",
            "input_refs": ["/datasets/pcb_avi_eval_3/"],
            "details": {
                "dataset_version": {
                    "source_fingerprint": "sha256:source",
                    "source_refs": ["/datasets/pcb_avi_eval_3/"],
                    "train_uri": "/runs/dataset-1/train.jsonl",
                    "eval_uri": "/runs/dataset-1/eval.jsonl",
                    "train_records": 3,
                    "eval_records": 3,
                    "report_uri": "/runs/dataset-1/report.json",
                }
            },
            "external_task_id": "task-1",
            "task_run_id": "task-dataset-full",
            "output_uri": "/runs/dataset-1",
        },
        context,
    )

    assert submitted["status"] == "waiting_external"
    assert submitted["external_task_id"] == "task-1"
    assert completed["status"] == "succeeded"
    assert completed["artifacts"][-1]["kind"] == "dataset_version"
