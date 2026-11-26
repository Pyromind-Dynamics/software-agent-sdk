"""Bind SDK execution evidence without invoking a platform submission path."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from pyromind_common.platform.credentials import materialize_runtime_environment
from pyromind_common.platform.preparation import (
    LocalFiles,
    StagingDirectory,
    prepare_pipeline,
)
from pyromind_common.platform.scene_options import PipelineOptions
from pyromind_runtime.application.consolidation_evidence import (
    ConsolidationEvidence,
    ConsolidationStageEvidence,
    consolidation_definition_sha256,
    successful_dataset_recipe_runs,
)
from pyromind_runtime.application.sdk_workflow_archive import SdkWorkflowArchive
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.domain.versions import VersionToolRequest
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from .test_preparation import (
    preparation_setup as preparation_setup,
    ready_request as ready_request,
)
from .test_sdk_task_callbacks import sdk_project_runtime as sdk_project_runtime
from .test_version_integration import version_runtime as version_runtime


@pytest.fixture
def sdk_recipe_factory(tmp_path, monkeypatch):
    root = tmp_path / "recipe-inputs"
    root.mkdir()
    for name in ("pipeline.py", "export.py"):
        (root / name).write_text("print('recipe')\n")
    monkeypatch.setenv(
        "PYROMIND_DATAFLOW_PROFILES",
        json.dumps(
            {
                "text": {
                    "DF_API_KEY": "sdk-test-model-secret",
                    "DF_API_URL": "https://model.test/v1/chat/completions",
                    "DF_MODEL_NAME": "model",
                }
            }
        ),
    )

    def prepare(task_id="123", parent=None):
        staging = StagingDirectory(tmp_path / task_id, f"/workspace/jobs/{task_id}")
        graph = prepare_pipeline(
            PipelineOptions(
                script_path="pipeline.py",
                sft_export_script_path="export.py",
                input_path=f"/workspace/batch-{task_id}.jsonl",
                replay_of_task_id=parent,
                model_profile="text",
            ),
            files=LocalFiles(root),
            staging=staging,
            runtime_dir=(
                Path(__file__).resolve().parents[2]
                / ".agents/skills/data-processing/scripts/preparation"
            ),
            run_id=uuid4(),
        )
        recipe = json.loads((staging.local / "dataset-recipe.json").read_text())
        assert recipe["uses_labeling_gateway"] is True
        return {"task_id": task_id, "workflow": graph, "dataset_recipe": recipe}

    return prepare


@pytest.fixture
def sdk_recipe_metadata(sdk_recipe_factory):
    return sdk_recipe_factory()


async def _owned_task(runtime, context, kind="data_preparation"):
    environment = await runtime.platform_sdk_context("c", context)
    address = TaskCorrelation.model_validate_json(
        json.loads(environment["PYROMIND_TASK_CONTEXTS"])[kind]
    )
    await runtime.deliver_external_task_status(
        "c",
        task_id="123",
        status="Succeeded",
        correlation=address,
        trusted_platform=True,
        auto_run=False,
    )


async def test_sdk_recipe_binding_preserves_versions_and_consolidation_without_submit(
    version_runtime, sdk_recipe_metadata
):
    runtime, submission, context = version_runtime
    await _owned_task(runtime, context)
    request = VersionToolRequest(
        action="bind_execution_metadata", arguments=sdk_recipe_metadata
    )
    result = await runtime.version_call("c", request, "metadata", context)
    assert result["dataset_recipe_recorded"] is True
    assert result["status"] == "metadata_bound"
    store = runtime._store("c")
    receipt = store.load_workflow_execution_requests()["sdk:123"]
    assert receipt.dataset_recipe is not None
    assert receipt.dataset_recipe.uses_labeling_gateway is True
    assert 'export DF_API_KEY="$param"' in receipt.dataset_recipe.command_template
    stage = ConsolidationStageEvidence(
        task_id="123",
        source="workflow_execution_request",
        definition_sha256=consolidation_definition_sha256(receipt.workflow),
        revision=receipt.revision,
    )
    evidence = ConsolidationEvidence(
        conversation_id="c", dataset=stage, train=stage, eval=stage
    )
    assert successful_dataset_recipe_runs(store, evidence) == [receipt]
    project = runtime.versions.project("c", context)
    before = runtime.versions.store.list_versions(project.id)
    assert await runtime.version_call("c", request, "metadata-again", context) == result
    assert runtime.versions.store.list_versions(project.id) == before
    assert submission.submissions == []
    assert all(
        b"sdk-test-model-secret" not in path.read_bytes()
        for path in store.directory.rglob("*")
        if path.is_file()
    )
    task = store.load_snapshot().external_tasks[0]
    assert task.correlation == receipt.correlation and task.status == "succeeded"
    assert task.output_dir == receipt.dataset_recipe.output_directory


@pytest.mark.parametrize("owned", [False, True])
async def test_sdk_metadata_requires_confirmed_owned_task(
    version_runtime, sdk_recipe_metadata, owned
):
    runtime, submission, context = version_runtime
    if owned:
        await _owned_task(runtime, context)
        context = replace(context, user_id="other")
    with pytest.raises(PermissionError):
        await runtime.version_call(
            "c",
            VersionToolRequest(
                action="bind_execution_metadata", arguments=sdk_recipe_metadata
            ),
            "metadata",
            context,
        )
    assert runtime._store("c").load_workflow_execution_requests() == {}
    assert submission.submissions == []


@pytest.mark.parametrize("change", ["facts", "credentials", "asset", "replay"])
async def test_sdk_metadata_rejects_changes_credentials_and_foreign_recipe_evidence(
    version_runtime, sdk_recipe_metadata, change
):
    runtime, submission, context = version_runtime
    await _owned_task(runtime, context)
    if change == "facts":
        await runtime.version_call(
            "c",
            VersionToolRequest(
                action="bind_execution_metadata", arguments=sdk_recipe_metadata
            ),
            "metadata",
            context,
        )
    changed = deepcopy(sdk_recipe_metadata)
    if change == "facts":
        changed["workflow"]["name"] = "different-execution"
    elif change == "credentials":
        changed["workflow"]["nodes"][0]["data"]["config"]["command"] = (
            "export DF_API_KEY=must-not-persist"
        )
    elif change == "asset":
        changed["dataset_recipe"]["assets"][0]["path"] = (
            "/workspace/foreign/recipe-assets/pipeline.py"
        )
    else:
        changed["dataset_recipe"]["replay_of_task_id"] = "999"
    with pytest.raises(ValueError):
        await runtime.version_call(
            "c",
            VersionToolRequest(action="bind_execution_metadata", arguments=changed),
            "metadata",
            context,
        )
    assert submission.submissions == []


async def test_sdk_metadata_retries_metadata_failure_without_a_second_submit(
    version_runtime, sdk_recipe_metadata, monkeypatch
):
    runtime, submission, context = version_runtime
    await _owned_task(runtime, context)
    save = FileProductStore.save_workflow_execution_request

    def unavailable(*args, **kwargs):
        raise OSError("metadata unavailable")

    request = VersionToolRequest(
        action="bind_execution_metadata", arguments=sdk_recipe_metadata
    )
    monkeypatch.setattr(
        FileProductStore, "save_workflow_execution_request", unavailable
    )
    with pytest.raises(OSError, match="metadata unavailable"):
        await runtime.version_call("c", request, "metadata", context)
    monkeypatch.setattr(FileProductStore, "save_workflow_execution_request", save)
    result = await runtime.version_call("c", request, "metadata", context)
    assert result["status"] == "metadata_bound"
    assert submission.submissions == []


@pytest.mark.parametrize("foreign", [False, True])
async def test_sdk_metadata_checks_signed_project_task_and_attempt(
    sdk_project_runtime, version_runtime, sdk_recipe_metadata, foreign
):
    runtime, context, project = sdk_project_runtime
    runtime.versions = version_runtime[0].versions
    scope = project.conversation_id
    environment = await runtime.platform_sdk_context(scope, context)
    execution = json.loads(environment["PYROMIND_PROJECT_EXECUTIONS"])[0]
    address = TaskCorrelation.model_validate_json(
        execution["out_ids"]["data_preparation"]
    )
    assert address.execution is not None
    if foreign:
        address = runtime._store(scope).issue_task_correlation(
            address.model_copy(
                update={
                    "execution": address.execution.model_copy(
                        update={"project_id": "foreign-project"}
                    )
                }
            )
        )
    await runtime.deliver_external_task_status(
        scope,
        task_id="123",
        status="Succeeded",
        correlation=address,
        trusted_platform=True,
        auto_run=False,
    )
    request = VersionToolRequest(
        action="bind_execution_metadata", arguments=sdk_recipe_metadata
    )
    if foreign:
        with pytest.raises(PermissionError, match="Project/task/attempt"):
            await runtime.version_call(scope, request, "metadata", context)
    else:
        result = await runtime.version_call(scope, request, "metadata", context)
        assert result["dataset_recipe_recorded"] is True


@pytest.mark.parametrize(
    "case", ["normal", "legacy", "prelabel_only", "manual_verified"]
)
async def test_project_dataset_callback_captures_recipe_and_independent_replay(
    sdk_project_runtime, version_runtime, sdk_recipe_factory, case
):
    runtime, context, project = sdk_project_runtime
    runtime.versions = version_runtime[0].versions
    scope = project.conversation_id
    context = replace(context, cookie="session=owner", x_cluster=project.cluster)
    env = await runtime.platform_sdk_context(scope, context)
    signed = json.loads(env["PYROMIND_PROJECT_EXECUTIONS"])[0]["out_ids"]
    unscoped = json.loads(env["PYROMIND_TASK_CONTEXTS"])
    first = sdk_recipe_factory("120")
    replay = sdk_recipe_factory("121", parent="120")
    tasks = {"120": first, "121": replay}
    legacy_parent = case != "normal"
    if legacy_parent:
        first["workflow"]["nodes"][0]["data"]["config"] = {"command": "echo legacy"}

    only_prelabel = {
        "nodes": [
            {
                "id": "1",
                "data": {
                    "nodeType": "CustomCommandCPUNode",
                    "config": {
                        "command": (
                            "set -eu\nexport DF_OUTPUT_SCHEMA=structured\n"
                            "python pipeline.py --output processed.jsonl"
                        )
                    },
                },
            }
        ],
        "edges": [],
    }

    async def read(task_id, _context):
        if task_id == "121" and case == "prelabel_only":
            return only_prelabel
        return materialize_runtime_environment(tasks[task_id]["workflow"])

    archive = SdkWorkflowArchive(read, runtime.versions, runtime.register_external_task)
    runtime._sdk_workflow_archive = archive
    runtime._task_contexts[scope] = context
    for task_id, origin in (("120", signed), ("121", unscoped)):
        if task_id == "121" and case == "manual_verified":
            runtime._task_contexts.pop(scope)
        await runtime.deliver_external_task_status(
            scope,
            task_id=task_id,
            status="Succeeded",
            trusted_platform=True,
            auto_run=False,
            correlation=TaskCorrelation.model_validate_json(origin["data_preparation"]),
        )
        await asyncio.gather(*tuple(archive._workers.values()))
    store = runtime._store(scope)
    if case == "prelabel_only":
        for claimed in (only_prelabel, replay["workflow"]):
            with pytest.raises(ValueError, match="未执行这份完整配方"):
                await runtime.version_call(
                    scope,
                    VersionToolRequest(
                        action="bind_execution_metadata",
                        arguments={**replay, "workflow": claimed},
                    ),
                    "manual-replay",
                    context,
                )
        assert "sdk:121" not in store.load_workflow_execution_requests()
        assert all(
            run.platform_task_id != "121"
            for run in store.load_snapshot().experiment_executions
        )
        return
    if case == "manual_verified":
        assert "sdk:121" not in store.load_workflow_execution_requests()
        await runtime.version_call(
            scope,
            VersionToolRequest(action="bind_execution_metadata", arguments=replay),
            "manual-replay",
            context,
        )
    first_receipt, second_receipt = (
        store.load_workflow_execution_requests()["sdk:" + i] for i in tasks
    )
    assert (first_receipt.dataset_recipe is None) == legacy_parent
    assert (
        second_receipt.dataset_recipe
        and second_receipt.dataset_recipe.replay_of_task_id == "120"
    )
    assert (
        second_receipt.workflow["nodes"][0]["data"]["config"]["param"] == "DF_API_KEY"
    )
    snapshot = store.load_snapshot()
    assert (
        next(t.output_dir for t in snapshot.external_tasks if t.task_id == "121")
        == "/workspace/jobs/121"
    )
    assert all(run.platform_task_id != "121" for run in snapshot.experiment_executions)
    assert (
        "sdk-test-model-secret"
        not in (store.directory / "workflow-execution-requests.json").read_text()
    )
    assert not version_runtime[1].submissions
    foreign = store.issue_task_correlation(
        TaskCorrelation.model_validate_json(unscoped["data_preparation"]).model_copy(
            update={"cluster": "other"}
        )
    )
    runtime._sdk_workflow_archive = None
    await runtime.deliver_external_task_status(
        scope,
        task_id="122",
        status="Succeeded",
        trusted_platform=True,
        auto_run=False,
        correlation=foreign,
    )
    with pytest.raises(PermissionError, match="signed Project Dataset"):
        await runtime.version_call(
            scope,
            VersionToolRequest(
                action="bind_execution_metadata",
                arguments=sdk_recipe_factory("122", parent="120"),
            ),
            "foreign-replay",
            context,
        )
