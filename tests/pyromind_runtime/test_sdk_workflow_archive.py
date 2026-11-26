import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from pyromind_runtime.application.sdk_execution_metadata import (
    bind_sdk_execution_metadata,
    portable_sdk_workflow,
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
def platform_graph():
    return {
        "name": "SFT",
        "extra": {"taskId": "21", "customType": "fromtask"},
        "nodes": [
            {
                "id": "wb",
                "properties": {"dystatus": "Succeeded"},
                "outputs_values": ["injected-private-output"],
                "data": {
                    "nodeType": "PyroMindWandbConfigBuilderNode",
                    "config": {"wandb_api_key": "A" * 40},
                },
            },
            {
                "id": "train",
                "data": {
                    "nodeType": "ModelTrainSFTNode",
                    "config": {"wandb_config": "injected-private-output"},
                },
            },
        ],
        "edges": [
            {
                "source": "wb",
                "sourceHandle": "wandb_config",
                "target": "train",
                "targetHandle": "wandb_config",
            }
        ],
    }


@pytest.mark.parametrize("mode", ["callback", "restore", "renew"])
async def test_scoped_sdk_submission_archives_platform_graph_without_agent_binding(
    sdk_project_runtime, version_runtime, platform_graph, mode
):
    runtime, context, project = sdk_project_runtime
    runtime.versions = version_runtime[0].versions
    calls = []

    async def read(task_id, owner):
        assert owner.user_id == context.user_id
        assert owner.x_cluster == project.cluster
        calls.append(task_id)
        if mode == "renew" and len(calls) == 1:
            raise PermissionError("expired")
        return platform_graph

    archive = SdkWorkflowArchive(read, runtime.versions, runtime.register_external_task)
    runtime._sdk_workflow_archive = archive
    scope = project.conversation_id
    context = replace(context, cookie="session=first")
    env = await runtime.platform_sdk_context(scope, context)
    token = json.loads(env["PYROMIND_PROJECT_EXECUTIONS"])[0]["out_ids"]["workflow_run"]
    correlation = TaskCorrelation.model_validate_json(token)
    runtime._task_contexts[scope] = (
        replace(context, cookie=None) if mode == "restore" else context
    )
    await runtime.deliver_external_task_status(
        scope,
        task_id="21",
        status="Succeeded",
        correlation=correlation,
        trusted_platform=True,
        auto_run=False,
    )
    await asyncio.gather(*archive._workers.values())
    store = runtime._store(scope)
    assert store.load_snapshot().external_tasks[0].status == "succeeded"
    if mode != "callback":
        assert "sdk:21" not in store.load_workflow_execution_requests()
        if mode == "renew":
            # A refresh with unchanged credentials is throttled.
            await runtime.get_snapshot(scope, context)
            assert calls == ["21"]
            context = replace(context, cookie="session=renewed")
        await runtime.get_snapshot(scope, context)
        await asyncio.gather(*archive._workers.values())
    receipt = store.load_workflow_execution_requests()["sdk:21"]
    assert receipt.correlation == correlation
    assert receipt.workflow == portable_sdk_workflow(platform_graph)
    assert receipt.summary["metadata_source"] == "studio_task"
    assert "injected-private-output" not in json.dumps(receipt.workflow)
    assert "A" * 40 not in json.dumps(receipt.workflow)
    assert (
        receipt.workflow["nodes"][0]["data"]["config"]["wandb_api_key"]
        == "WANDB_API_KEY"
    )
    assert receipt.workflow["nodes"][1]["data"]["config"] == {}

    await runtime.get_snapshot(scope, context)
    assert not archive._workers
    assert len(calls) == (2 if mode == "renew" else 1)
    source = deepcopy(receipt.workflow)
    enriched = VersionToolRequest(
        action="bind_execution_metadata",
        arguments={"task_id": "21", "workflow": source, "code": "print('frozen')"},
    )
    await runtime.version_call(scope, enriched, "enrich", context)
    await runtime.version_call(scope, enriched, "repeat", context)
    final = store.load_workflow_execution_requests()["sdk:21"]
    assert final.revision == 2 and final.workflow == receipt.workflow
    version_project = runtime.versions.project(scope, context, create=False)
    assert version_project
    (version_run,) = runtime.versions.store.list_runs(version_project.id)
    assert version_run.source.task_id == "21"
    assert "version://" + version_run.spec.inputs["config"] == receipt.workflow_path
    assert final.workflow_path == "version://" + version_run.id
    restored_archive = SdkWorkflowArchive(
        read, runtime.versions, runtime.register_external_task
    )
    restored_archive.schedule(store, context, store.load_snapshot().external_tasks)
    await asyncio.gather(*restored_archive._workers.values())
    assert len(calls) == (2 if mode == "renew" else 1)
    assert store.load_workflow_execution_requests()["sdk:21"] == final
    assert len(version_runtime[1].submissions) == 0
    source["nodes"][1]["data"]["config"]["epochs"] = 200
    with pytest.raises(ValueError, match="already frozen"):
        await runtime.version_call(
            scope,
            VersionToolRequest(
                action="bind_execution_metadata",
                arguments={"task_id": "21", "workflow": source},
            ),
            "changed",
            context,
        )


async def test_concurrent_platform_archive_rejects_unsafe_graph_and_reuses_config(
    sdk_project_runtime, version_runtime, platform_graph
):
    runtime, context, project = sdk_project_runtime
    versions = version_runtime[0].versions
    scope = project.conversation_id
    env = await runtime.platform_sdk_context(scope, context)
    entry = json.loads(env["PYROMIND_PROJECT_EXECUTIONS"])[0]
    await runtime.deliver_external_task_status(
        scope,
        task_id="21",
        status="Succeeded",
        auto_run=False,
        trusted_platform=True,
        correlation=TaskCorrelation.model_validate_json(
            entry["out_ids"]["workflow_run"]
        ),
    )
    store = runtime._store(scope)

    def bind(graph):
        return bind_sdk_execution_metadata(
            FileProductStore(store.conversation_dir),
            versions,
            {"task_id": "21", "workflow": graph},
            context,
            platform_archive=True,
        )[0]

    malformed = deepcopy(platform_graph)
    malformed["nodes"][0]["data"]["config"]["wandb_api_key"] = {"secret": "unexpected"}
    with pytest.raises(ValueError, match="credential reference"):
        bind(malformed)
    unsafe = deepcopy(platform_graph)
    unsafe["nodes"][1]["data"]["config"]["command"] = (
        'export WANDB_API_KEY="private-value"'
    )
    with pytest.raises(ValueError, match="Worker credentials"):
        bind(unsafe)
    platform_graph["nodes"][0]["data"]["config"]["wandb_api_key"] = (
        "secret-alias-or-unfamiliar-key-format"
    )
    result = await asyncio.gather(
        *[asyncio.to_thread(bind, platform_graph) for _ in range(3)]
    )
    assert result[0] == result[1] == result[2]
    assert len(store.load_workflow_execution_requests()) == 1
    version_project = versions.project(scope, context, create=False)
    assert (
        version_project and len(versions.store.list_versions(version_project.id)) == 1
    )
    assert '"wandb_api_key": "WANDB_API_KEY"' in json.dumps(result[0].workflow)
    serialized = (store.directory / "workflow-execution-requests.json").read_text()
    assert (
        "secret-alias-or-unfamiliar-key-format" not in serialized
        and "private-value" not in serialized
    )
