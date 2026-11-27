"""Keep SDK evaluation evidence and its originating chat turn after merging."""

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from pyromind_runtime.application.model_evaluation_plan import (
    check_inference_receipt,
    sdk_inference_paths,
)
from pyromind_runtime.application.plan_execution import (
    completed_callback_task,
    plan_delivery_source,
    plan_executions,
    platform_results_ready,
    with_platform_states,
)
from pyromind_runtime.application.sdk_execution_metadata import (
    verified_sdk_inference_paths,
)
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.preparation import SetPreparationPlan
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.domain.versions import VersionToolRequest

from . import test_preparation
from .test_sdk_task_callbacks import sdk_project_runtime as sdk_project_runtime
from .test_version_integration import version_runtime as version_runtime
from .test_workflow_execution import _runtime


standard_preparation_setup = test_preparation.preparation_setup
standard_ready_request = test_preparation.ready_request


@pytest.fixture
def preparation_setup(standard_preparation_setup):
    preparations, owner, policy = standard_preparation_setup
    return (
        preparations,
        owner,
        policy.model_copy(
            update={
                "workflow_nodes": tuple(
                    node.model_copy(
                        update={
                            "tasks": tuple(
                                task.model_copy(update={"evaluation_target": "model"})
                                for task in node.tasks
                            )
                        }
                    )
                    for node in policy.workflow_nodes
                )
            }
        ),
    )


@pytest.fixture
def dataset_input_name():
    return "dataset"


@pytest.fixture
def ready_request(standard_ready_request, dataset_input_name):
    document = standard_ready_request.model_dump(mode="json")
    document["plan"]["tasks"][0]["inputs"][0]["name"] = dataset_input_name
    document["plan"]["resources"].append(
        {"id": "model", "name": "Candidate", "storage_path": "/workspace/models/merged"}
    )
    document["plan"]["tasks"][0]["inputs"].append(
        {"name": "model", "resource_id": "model"}
    )
    return SetPreparationPlan.model_validate(document)


@pytest.fixture
def inference_graph():
    return {
        "name": "SDK inference",
        "nodes": [
            {
                "id": "1",
                "data": {
                    "nodeType": "VLLMInference",
                    "config": {"model_path": "/workspace/models/merged"},
                },
            },
            {
                "id": "2",
                "data": {
                    "nodeType": "CustomCommandCPUNode",
                    "config": {
                        "command": "python3 /workspace/eval/evaluate_inference.py "
                        '--endpoint "$param" --model-reference /models/merged '
                        "--dataset-path /workspace/val.jsonl"
                    },
                },
            },
        ],
        "edges": [
            {
                "source": "1",
                "sourceHandle": "endpoint",
                "target": "2",
                "targetHandle": "param",
            }
        ],
    }


@pytest.mark.parametrize("source", ["storage", "lora", "clone", "download"])
def test_sdk_inference_paths_follow_the_served_model(inference_graph, source):
    graph = inference_graph
    if source != "storage":
        graph["nodes"][0]["data"]["config"] = {}
        kind, config = {
            "lora": ("ModelMergeLoraNode", {"lora_path": "/workspace/models/merged"}),
            "clone": ("CloneAndCacheModel", {"model": "/models/merged"}),
            "download": ("DownloadAndCacheModel", {"modelname": "/models/merged"}),
        }[source]
        graph["nodes"].append({"id": "3", "data": {"nodeType": kind, "config": config}})
        graph["edges"].append(
            {
                "source": "3",
                "sourceHandle": "merged_model_path"
                if source == "lora"
                else "model_path",
                "target": "1",
                "targetHandle": "model_path",
            }
        )
    assert sdk_inference_paths(graph) == ("/models/merged", "/workspace/val.jsonl")


@pytest.mark.parametrize("change", ["model", "connection", "scoring", "command"])
def test_sdk_inference_paths_reject_unmatched_or_scoring_only_graphs(
    inference_graph, change
):
    graph = inference_graph
    if change == "model":
        graph["nodes"][0]["data"]["config"]["model_path"] = "/workspace/models/base"
    elif change == "connection":
        graph["edges"] = []
    elif change == "scoring":
        graph["nodes"][0]["data"]["nodeType"] = "InputNode"
    else:
        graph["nodes"][1]["data"]["config"]["command"] += " && echo other"
    assert sdk_inference_paths(graph) is None


async def test_sdk_frozen_inference_is_accepted_without_a_legacy_pipeline(
    sdk_project_runtime, version_runtime, inference_graph
):
    runtime, context, project = sdk_project_runtime
    runtime.versions = version_runtime[0].versions
    scope = project.conversation_id
    environment = await runtime.platform_sdk_context(scope, context)
    entry = json.loads(environment["PYROMIND_PROJECT_EXECUTIONS"])[0]
    address = TaskCorrelation.model_validate_json(entry["out_ids"]["evaluation"])
    await runtime.deliver_external_task_status(
        scope,
        task_id="sdk-evaluation",
        status="Succeeded",
        correlation=address,
        trusted_platform=True,
        auto_run=False,
    )
    request = VersionToolRequest(
        action="bind_execution_metadata",
        arguments={"task_id": "sdk-evaluation", "workflow": inference_graph},
    )
    await runtime.version_call(scope, request, "metadata", context)
    store = runtime._store(scope)
    paths = verified_sdk_inference_paths(store)
    assert paths == {"sdk-evaluation": ("/models/merged", "/workspace/val.jsonl")}
    assert store.load_pipeline_runs() == {}
    runs = store.load_snapshot().experiment_executions
    assert runs
    run = runs[0].model_copy(update={"state": "succeeded"})
    assert check_inference_receipt(run, {}).state == "needs_input"
    assert check_inference_receipt(run, {}, sdk_receipts=paths).state == "succeeded"
    metadata = deepcopy(request.arguments)
    changed_graph = deepcopy(inference_graph)
    changed_graph["nodes"][0]["data"]["config"]["model_path"] = "/other"
    metadata["workflow"] = changed_graph
    with pytest.raises(ValueError, match="already frozen"):
        await runtime.version_call(
            scope,
            VersionToolRequest(action="bind_execution_metadata", arguments=metadata),
            "rewrite-metadata",
            context,
        )


async def test_late_sdk_callback_keeps_the_signed_originating_chat_turn(tmp_path):
    runtime, _, _, context = await _runtime(tmp_path)
    scope = "workflow-conversation"
    store = runtime._store(scope)
    try:
        original = None
        for task, run, turn in (
            ("requirement:first", "run-first", "chat-turn-first"),
            ("requirement:second", "run-second", "chat-turn-second"),
        ):
            store.append(
                ProductEvent(
                    conversation_id=scope,
                    type="task.queued",
                    payload={
                        "task_id": task,
                        "kind": "requirement",
                        "instruction": "Review the task",
                        "origin_turn_id": turn,
                    },
                )
            )
            store.append(
                ProductEvent(
                    conversation_id=scope,
                    type="status.changed",
                    run_id=run,
                    task_id=task,
                    payload={"status": "running"},
                )
            )
            environment = await runtime.platform_sdk_context(scope, context)
            correlation = TaskCorrelation.model_validate_json(
                json.loads(environment["PYROMIND_TASK_CONTEXTS"])["workflow_run"]
            )
            assert correlation.origin_run_id == run
            if run == "run-first":
                original = correlation
        assert original is not None
        with pytest.raises(PermissionError, match="signature"):
            await runtime.deliver_external_task_status(
                scope,
                task_id="forged",
                status="Succeeded",
                correlation=original.model_copy(update={"origin_run_id": "run-second"}),
                trusted_platform=True,
                auto_run=False,
            )
        await runtime.deliver_external_task_status(
            scope,
            task_id="late-sdk-task",
            status="Succeeded",
            correlation=original,
            trusted_platform=True,
            auto_run=False,
        )
        snapshot = store.load_snapshot()
        task = next(
            task for task in snapshot.external_tasks if task.task_id == "late-sdk-task"
        )
        assert task.run_id == "run-first"
        assert snapshot.task_runs is not None
        origin = snapshot.task_runs[task.run_id]
        assert snapshot.task_origins[origin.task_id] == "chat-turn-first"
    finally:
        await runtime.close()


@pytest.mark.parametrize("receipt_kind", ["evaluation", "workflow_run"])
@pytest.mark.parametrize("change", [None, "model", "dataset"])
@pytest.mark.parametrize(
    "dataset_input_name", ["dataset", "eval_data", "evaluation_dataset"]
)
async def test_sdk_evaluation_plan_capture_checks_frozen_paths(
    sdk_project_runtime,
    version_runtime,
    inference_graph,
    monkeypatch,
    change,
    receipt_kind,
    dataset_input_name,
):
    runtime, context, project = sdk_project_runtime
    runtime.versions = version_runtime[0].versions
    scope = project.conversation_id
    model = "/models/base" if change == "model" else "/models/merged"
    dataset = (
        "/workspace/train.jsonl" if change == "dataset" else "/workspace/eval.jsonl"
    )
    inference_graph["nodes"][0]["data"]["config"]["model_path"] = model
    inference_graph["nodes"][1]["data"]["config"]["command"] = (
        "python3 /workspace/eval/evaluate_inference.py --endpoint '$param' "
        f"--model-reference {model} --dataset-path {dataset}"
    )
    environment = await runtime.platform_sdk_context(scope, context)
    entry = json.loads(environment["PYROMIND_PROJECT_EXECUTIONS"])[0]
    address = TaskCorrelation.model_validate_json(entry["out_ids"][receipt_kind])
    await runtime.deliver_external_task_status(
        scope,
        task_id="sdk-evaluation",
        status="Succeeded",
        correlation=address,
        trusted_platform=True,
        auto_run=False,
    )
    await runtime.version_call(
        scope,
        VersionToolRequest(
            action="bind_execution_metadata",
            arguments={"task_id": "sdk-evaluation", "workflow": inference_graph},
        ),
        "metadata",
        context,
    )
    rows = [
        {
            "id": project.plan.tasks[0].id,
            "stage_id": project.plan.tasks[0].stage_id,
            "plan_attempt": 1,
            "platform_task_id": "sdk-evaluation",
            "business_status": "completed",
        }
    ]
    active = runtime._active[scope]
    monkeypatch.setattr(
        active.adapter,
        "read_workspace_text",
        AsyncMock(return_value=json.dumps({"tasks": rows})),
    )
    assert await runtime._capture_plan_progress(active, runtime._store(scope)) == rows
    snapshot = await runtime.get_snapshot(scope, context)
    assert snapshot.experiment_executions
    run = snapshot.experiment_executions[0]
    assert run.platform_task_id == "sdk-evaluation"
    if receipt_kind == "workflow_run":
        assert not verified_sdk_inference_paths(runtime._store(scope), project, set())
    if change:
        assert run.state == "needs_input"
        assert run.waiting_reason
        if receipt_kind == "evaluation":
            assert "requires" in run.waiting_reason
        assert not run.artifacts and run.report_uri is None
    else:
        assert run.state == "succeeded"
        assert run.waiting_reason is None


async def test_sdk_second_shard_matches_its_registered_plan_attempt(
    sdk_project_runtime,
):
    runtime, context, project = sdk_project_runtime
    scope = project.conversation_id
    environment = await runtime.platform_sdk_context(scope, context)
    entry = json.loads(environment["PYROMIND_PROJECT_EXECUTIONS"])[0]
    address = TaskCorrelation.model_validate_json(entry["out_ids"]["evaluation"])
    for identity in ("shard-first", "shard-second"):
        await runtime.deliver_external_task_status(
            scope,
            task_id=identity,
            status="Succeeded",
            correlation=address,
            trusted_platform=True,
            auto_run=True,
        )
    store = runtime._store(scope)
    snapshot = store.load_snapshot()
    registered = snapshot.experiment_executions
    assert registered and registered[0].platform_task_ids == (
        "shard-first",
        "shard-second",
    )
    task = project.plan.tasks[0]
    (run,) = tuple(
        plan_executions(
            project,
            [
                {
                    "id": task.id,
                    "stage_id": task.stage_id,
                    "plan_attempt": 1,
                    "platform_task_id": "shard-second",
                    "business_status": "completed",
                }
            ],
            registered=registered,
        )
    )
    assert run.run_id == registered[0].run_id and run.plan_attempt == 1
    assert (
        completed_callback_task(
            store.load_snapshot(),
            "shard-second",
            project.project_id,
            allow_pending_analysis=True,
        )
        == task.id
    )


async def test_selected_eval_submission_requires_its_own_inference_receipt(
    sdk_project_runtime, version_runtime, inference_graph, monkeypatch
):
    runtime, context, project = sdk_project_runtime
    runtime.versions = version_runtime[0].versions
    scope = project.conversation_id
    environment = await runtime.platform_sdk_context(scope, context)
    (entry,) = json.loads(environment["PYROMIND_PROJECT_EXECUTIONS"])
    address = TaskCorrelation.model_validate_json(entry["out_ids"]["workflow_run"])
    task = project.plan.tasks[0]
    store = runtime._store(scope)
    active = runtime._active[scope]
    graph = inference_graph
    graph["nodes"][1]["data"]["config"]["command"] = (
        "python3 evaluate_inference.py --endpoint '$param' "
        "--model-reference /models/merged --dataset-path /workspace/eval.jsonl"
    )
    for identity in ("first-inference", "score-only", "final-inference"):
        await runtime.deliver_external_task_status(
            scope,
            task_id=identity,
            status="Succeeded",
            correlation=address,
            trusted_platform=True,
            auto_run=False,
        )
        await runtime.version_call(
            scope,
            VersionToolRequest(
                action="bind_execution_metadata",
                arguments={
                    "task_id": identity,
                    "workflow": graph
                    if identity != "score-only"
                    else {
                        "name": "score only",
                        "nodes": graph["nodes"][1:],
                        "edges": [],
                    },
                },
            ),
            "metadata-" + identity,
            context,
        )
        rows = [
            {
                "id": task.id,
                "stage_id": task.stage_id,
                "attempt": 1,
                "platform_task_id": identity,
                "business_status": "completed",
            }
        ]
        monkeypatch.setattr(
            active.adapter,
            "read_workspace_text",
            AsyncMock(return_value=json.dumps({"tasks": rows})),
        )
        await runtime._capture_plan_progress(active, store)
        snapshot = await runtime.get_snapshot(scope, context)
        assert snapshot.experiment_executions
        (run,) = snapshot.experiment_executions
        assert run.platform_task_id == identity
        assert run.state == ("needs_input" if identity == "score-only" else "succeeded")

    saved = store.load_snapshot()
    assert saved.experiment_executions
    (selected,) = saved.experiment_executions
    delivered = saved.model_copy(
        update={
            "experiment_executions": (
                selected.model_copy(update={"report_state": "ready"}),
            ),
            "external_tasks": tuple(
                external.model_copy(update={"status": "failed"})
                if external.task_id == "first-inference"
                else external
                for external in saved.external_tasks
            ),
        }
    )
    required = {task.id}
    paths = verified_sdk_inference_paths(store, project, required)
    statuses = {task.task_id: task.status for task in delivered.external_tasks}
    assert not platform_results_ready(selected, statuses)
    source = plan_delivery_source(selected, paths)
    assert source.platform_task_ids == ("final-inference",)
    assert platform_results_ready(source, statuses)
    assert plan_delivery_source(selected, {}) == selected
    evaluated = with_platform_states(delivered, required, sdk_receipts=paths)
    assert (
        evaluated.experiment_executions
        and evaluated.experiment_executions[0].state == "succeeded"
    )
    ordinary = with_platform_states(delivered, set(), sdk_receipts=paths)
    assert (
        ordinary.experiment_executions
        and ordinary.experiment_executions[0].state == "failed"
    )


async def test_blocked_model_eval_projects_a_new_signed_retry(
    sdk_project_runtime, version_runtime, inference_graph, monkeypatch
):
    runtime, context, project = sdk_project_runtime
    runtime.versions = version_runtime[0].versions
    scope = project.conversation_id
    (entry,) = json.loads(
        (await runtime.platform_sdk_context(scope, context))[
            "PYROMIND_PROJECT_EXECUTIONS"
        ]
    )
    evaluation = TaskCorrelation.model_validate_json(entry["out_ids"]["evaluation"])
    preparation = TaskCorrelation.model_validate_json(
        entry["out_ids"]["data_preparation"]
    )
    store = runtime._store(scope)

    async def current_run():
        snapshot = await runtime.get_snapshot(scope, context)
        assert snapshot.experiment_executions
        (run,) = snapshot.experiment_executions
        return run

    await runtime.deliver_external_task_status(
        scope,
        task_id="old-evaluation",
        status="Failed",
        correlation=evaluation,
        trusted_platform=True,
        auto_run=False,
    )
    saved = store.load_snapshot()
    assert saved.experiment_executions
    (old,) = saved.experiment_executions
    runtime._record_execution(
        store, old.model_copy(update={"state": "needs_input", "waiting_reason": "old"})
    )
    for task_id, correlation in (
        ("new-evaluation", evaluation),
        ("auxiliary-preparation", preparation),
    ):
        await runtime.deliver_external_task_status(
            scope,
            task_id=task_id,
            status="Running",
            correlation=correlation,
            trusted_platform=True,
            auto_run=False,
        )

    running = await current_run()
    assert running.state == "waiting_platform"
    assert running.platform_task_id == "new-evaluation"
    assert running.waiting_reason is None
    assert running.platform_task_ids == (
        "old-evaluation",
        "new-evaluation",
        "auxiliary-preparation",
    )

    await runtime.deliver_external_task_status(
        scope,
        task_id="new-evaluation",
        status="Succeeded",
        correlation=evaluation,
        trusted_platform=True,
        auto_run=False,
    )
    blocked = await current_run()
    assert blocked.state == "needs_input"
    assert blocked.platform_task_id == "new-evaluation"
    assert blocked.waiting_reason

    inference_graph["nodes"][1]["data"]["config"]["command"] = (
        "python3 evaluate_inference.py --endpoint '$param' "
        "--model-reference /models/merged --dataset-path /workspace/eval.jsonl"
    )
    await runtime.version_call(
        scope,
        VersionToolRequest(
            action="bind_execution_metadata",
            arguments={"task_id": "new-evaluation", "workflow": inference_graph},
        ),
        "retry-metadata",
        context,
    )
    ready = await current_run()
    assert ready.state == "waiting_analysis"
    assert ready.platform_task_id == "new-evaluation"
    assert not ready.registered_report_ids

    task = project.plan.tasks[0]
    active = runtime._active[scope]
    snapshot = store.load_snapshot()
    assert snapshot.experiment_executions
    (persisted,) = snapshot.experiment_executions
    assert persisted.platform_task_id == "old-evaluation"
    record = {
        "plan_attempt": 1,
        "platform_task_id": "new-evaluation",
        "business_status": "in_progress",
    }
    monkeypatch.setattr(
        active.adapter,
        "read_workspace_text",
        AsyncMock(
            return_value=json.dumps(
                {"tasks": [{"id": task.id, "stage_id": task.stage_id, **record}]}
            )
        ),
    )
    receipts = verified_sdk_inference_paths(store, project, {task.id})
    accepted = await runtime._resolve_evaluation_delivery_retry(
        active, project, task, persisted, snapshot, receipts
    )
    assert accepted.platform_task_id == "new-evaluation"
    assert accepted.state == "waiting_analysis"
    source = plan_delivery_source(accepted, receipts)
    assert source.platform_task_ids == ("new-evaluation",)
    with pytest.raises(ValueError, match="requires model"):
        await runtime._resolve_evaluation_delivery_retry(
            active,
            project,
            task,
            persisted,
            snapshot,
            {"new-evaluation": ("/workspace/wrong-model", "/workspace/eval.jsonl")},
        )
