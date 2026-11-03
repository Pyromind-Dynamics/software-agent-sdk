import asyncio
from pathlib import Path

import pytest
from harness_adapter import artifact_snapshots as platform
from harness_adapter.pi_adapter.business_tool_host import (
    PyromindBusinessToolHost,
    ToolExecutionContext,
)
from harness_adapter.version_tools import (
    ProductVersionsAction,
    ProductVersionsTool,
    submission_bridge,
)
from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.pipeline_runs import PipelineRuns
from pyromind_runtime.application.versions import Versions
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.versions import (
    SaveVersion,
    SnapshotSource,
    VersionToolRequest,
)
from pyromind_runtime.infrastructure.file_artifact_snapshots import (
    FileArtifactSnapshots,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.infrastructure.sqlite_version_store import SQLiteVersionStore
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter
from .test_workflow_execution import FakeSubmission


@pytest.fixture
async def version_runtime(tmp_path):
    service = Versions(
        SQLiteVersionStore(tmp_path / "versions.sqlite3"),
        FileArtifactSnapshots(tmp_path / "storage"),
    )
    submission = FakeSubmission()
    context = RequestContext(user_id="42")
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        FakeAdapter(),
        workflow_submission=submission,
        versions=service,
    )
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="c",
            user_id="42",
            workspace_root=str(tmp_path / "conversations" / "c"),
        ),
        context,
    )
    yield runtime, submission, context
    await runtime.close()


async def test_direct_tool_submission_is_durable_and_idempotent(version_runtime):
    runtime, submission, context = version_runtime
    result = await runtime.submit_versioned_workflow(
        "c", "call-1", submission.workflow, "workflow = 1", context
    )
    assert result["task_id"] == "task-123"
    assert (
        await runtime.submit_versioned_workflow(
            "c", "call-1", submission.workflow, "workflow = 1", context
        )
        == result
    )
    assert len(submission.submissions) == 1
    project = runtime.versions.store.bound_project("c")
    records = runtime.versions.store.list_runs(project.id)
    assert len(records) == 1 and records[0].source.task_id == "task-123"
    with pytest.raises(ValueError, match="different execution"):
        await runtime.submit_versioned_workflow(
            "c", "call-1", {"name": "changed"}, None, context
        )


async def test_graph_submission_registers_the_await_owner_immediately(
    version_runtime,
):
    runtime, submission, context = version_runtime
    await runtime.submit_versioned_workflow(
        "c",
        "graph-call",
        submission.workflow,
        None,
        context,
        "graph-run-1",
    )

    task = next(
        task
        for task in runtime._store("c").load_snapshot().external_tasks
        if task.task_id == "task-123"
    )
    assert task.kind == "graph_await"
    assert task.graph_run_id == "graph-run-1"
    assert task.run_id == "graph-run-1"


async def test_uncertain_submission_never_auto_retries(version_runtime, monkeypatch):
    runtime, _, context = version_runtime
    calls = []

    async def timeout(*args):
        calls.append(args)
        raise TimeoutError("platform may have accepted it")

    monkeypatch.setattr(runtime._workflow_submission, "submit", timeout)
    with pytest.raises(ValueError, match="platform may have accepted it"):
        await runtime.submit_versioned_workflow(
            "c", "same-call", {"name": "train"}, None, context
        )
    with pytest.raises(ValueError, match="uncertain"):
        await runtime.submit_versioned_workflow(
            "c", "same-call", {"name": "train"}, None, context
        )
    assert len(calls) == 1


async def test_callback_snapshot_failure_replay_recovers_once(
    version_runtime, monkeypatch
):
    runtime, submission, context = version_runtime
    await runtime.submit_versioned_workflow(
        "c", "call-1", submission.workflow, None, context
    )
    snapshots = runtime.versions.snapshots
    original = snapshots.save

    def fail(*args):
        raise OSError("temporarily unavailable")

    monkeypatch.setattr(snapshots, "save", fail)
    completion = {
        "artifacts": [
            {
                "kind": "report",
                "logical_key": "report",
                "source": {"text": "result", "filename": "report.txt"},
            }
        ]
    }
    await runtime.deliver_external_task_status(
        "c",
        task_id="task-123",
        status="succeeded",
        completion_result=completion,
        auto_run=False,
    )
    project = runtime.versions.store.bound_project("c")
    outputs = [
        v for v in runtime.versions.store.list_versions(project.id) if v.producer_run_id
    ]
    assert len(outputs) == 1 and outputs[0].state == "failed"
    monkeypatch.setattr(snapshots, "save", original)
    await runtime.deliver_external_task_status(
        "c",
        task_id="task-123",
        status="succeeded",
        completion_result=completion,
        auto_run=False,
    )
    outputs = [
        v for v in runtime.versions.store.list_versions(project.id) if v.producer_run_id
    ]
    assert len(outputs) == 1 and outputs[0].state == "ready"
    assert len(submission.submissions) == 1


async def test_reconcile_links_task_after_metadata_write_failure(
    version_runtime, monkeypatch
):
    runtime, submission, context = version_runtime
    link = runtime.versions.store.link_task

    def fail(*args):
        raise OSError("sqlite unavailable")

    monkeypatch.setattr(runtime.versions.store, "link_task", fail)
    await runtime.submit_versioned_workflow(
        "c", "call-1", submission.workflow, None, context
    )
    project = runtime.versions.store.bound_project("c")
    assert runtime.versions.store.list_runs(project.id)[0].source.task_id is None
    monkeypatch.setattr(runtime.versions.store, "link_task", link)
    await runtime.reconcile_versions("c", context)
    assert runtime.versions.store.list_runs(project.id)[0].source.task_id == "task-123"
    assert len(submission.submissions) == 1


async def test_pi_and_openhands_tools_share_runtime_semantics(
    version_runtime, tmp_path
):
    runtime, _, context = version_runtime
    host = PyromindBusinessToolHost(
        skill_roots=[],
        skills_directory=Path(__file__).resolve().parents[2] / ".agents" / "skills",
    )
    host.version_call = runtime.version_call
    spec = next(item for item in host.specs() if item["name"] == "product_versions")
    assert "input_schema" in spec
    tool_context = ToolExecutionContext(
        conversation_id="c",
        workspace_root=tmp_path,
        request_context=context,
        model_configuration={},
    )
    args = {
        "action": "save",
        "command_id": "save",
        "arguments": {
            "kind": "code",
            "logical_key": "script",
            "source": {"text": "print(1)"},
        },
    }
    saved = await host.execute("product_versions", args, tool_context)
    assert saved["details"]["state"] == "ready"
    tool = ProductVersionsTool.create(
        callback=runtime.version_call,
        scope="c",
        context=context,
        loop=asyncio.get_running_loop(),
    )[0]
    result = await asyncio.to_thread(tool, ProductVersionsAction.model_validate(args))
    assert not result.is_error
    project = runtime.versions.store.bound_project("c")
    assert len(runtime.versions.store.list_versions(project.id)) == 1
    with pytest.raises(PermissionError):
        await runtime.version_call(
            "c",
            VersionToolRequest(action="list"),
            "read",
            RequestContext(user_id="other"),
        )


async def test_native_submission_bridge_returns_platform_response(version_runtime):
    runtime, submission, context = version_runtime
    handler = submission_bridge(
        runtime.submit_versioned_workflow,
        "c",
        lambda: "native-action",
        context,
        asyncio.get_running_loop(),
    )
    result = await asyncio.to_thread(handler, submission.workflow, "sft", "code")
    assert result.task_id == "task-123"


def test_platform_snapshot_upload_and_resume_use_original_bytes(tmp_path, monkeypatch):
    objects = {"/train.py": b"original"}

    def download(**kwargs):
        return objects[kwargs["storage_path"]]

    def upload(**kwargs):
        path = str(Path(kwargs["target_dir"]) / kwargs["local_path"].name)
        objects[path] = kwargs["local_path"].read_bytes()
        return path

    monkeypatch.setattr(platform, "download_file_from_pyromind", download)
    monkeypatch.setattr(platform, "upload_local_file_to_pyromind", upload)
    snapshots = platform.PlatformArtifactSnapshots(tmp_path)
    context = RequestContext(user_id="42")
    result = snapshots.save(
        SnapshotSource(storage_ref="/train.py"),
        "/.pyromind-agent/c/versions/v",
        context,
    )
    objects["/train.py"] = b"changed"
    replay = platform.PlatformArtifactSnapshots(tmp_path).save(
        SnapshotSource(storage_ref="/train.py"),
        "/.pyromind-agent/c/versions/v",
        context,
    )
    assert replay == result
    assert snapshots.read_text(result, context) == "original"
    objects[result.storage_ref] = b"tampered"
    with pytest.raises(ValueError, match="digest changed"):
        snapshots.verify(result, context)


def test_version_code_respects_architecture_boundaries():
    import ast

    root = Path(__file__).resolve().parents[2]
    for relative in (
        "pyromind-runtime/pyromind_runtime/application/versions.py",
        "pyromind-runtime/pyromind_runtime/domain/versions.py",
        "pyromind-runtime/pyromind_runtime/ports/versions.py",
        "pyromind-runtime/pyromind_runtime/infrastructure/sqlite_version_store.py",
    ):
        tree = ast.parse((root / relative).read_text())
        imports = [
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ]
        assert not any(
            name
            and name.split(".")[0]
            in {"openhands", "harness_adapter", "fastapi", "pyromind_agent_server"}
            for name in imports
        )
        if "/application/" in relative:
            assert not any(name and ".infrastructure" in name for name in imports)


async def test_confirmation_checks_snapshot_before_platform_submit(version_runtime):
    from pyromind_runtime.domain.commands import PermissionResponseCommand

    runtime, submission, context = version_runtime
    await runtime.prepare_workflow_execution(
        "c", "storage/project/workflow.py", context, request_id="request"
    )
    project = runtime.versions.store.bound_project("c")
    version = runtime.versions.store.list_versions(project.id)[0]
    runtime.versions.snapshots.path(version.manifest.storage_ref, context).write_text(
        "tampered"
    )
    with pytest.raises(ValueError, match="digest changed"):
        await runtime.submit_command(
            "c",
            PermissionResponseCommand(
                command_id="confirm",
                permission_id="workflow-run:request",
                decision="allow_once",
            ),
            context,
        )
    assert submission.submissions == []


@pytest.mark.parametrize("model_source", ["storage", "clone", "download"])
def test_pipeline_execution_uses_frozen_script_and_config(tmp_path, model_source):
    root = tmp_path / "c"
    root.mkdir()
    product = FileProductStore(root)
    product.create(
        ConversationSnapshot(conversation_id="c", capabilities=HarnessCapabilities()),
        user_id="42",
    )
    service = Versions(
        SQLiteVersionStore(tmp_path / "versions.sqlite3"),
        FileArtifactSnapshots(tmp_path / "storage"),
    )
    context = RequestContext(user_id="42")
    assert isinstance(service.snapshots, FileArtifactSnapshots)
    for reference in ("/data.jsonl", "/model"):
        path = service.snapshots.path(reference, context)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
    runs = PipelineRuns(
        "c", product, WorkflowExternalTaskRegistry(tmp_path), service, context
    )
    project = service.project("c", context)
    assert project is not None
    selected = service.store.reserve_version(
        project,
        "selected-model",
        SaveVersion(
            kind="model",
            logical_key="model",
            source=SnapshotSource(storage_ref="/model"),
        ),
    )
    selected = service.finish(selected, context)
    service.store.select(
        project.id,
        "model",
        selected.id,
        "select-model",
        "previous model",
        project.revision,
    )
    reference = "/model" if model_source == "storage" else "Unlisted/Model"
    definition = {
        "input_path": "/data.jsonl",
        "inference": {"model_path": reference},
        "dataset_config": {"field": "prompt"},
        "evaluation_config": {"rubrics": [], "model_source": {"type": model_source}},
    }
    run = runs.reserve(definition, None)
    directory = run["output_dir"]
    workflow = {
        "command": (
            f"python3 /workspace{directory}/evaluate_inference.py "
            f"--dataset-config /workspace{directory}/dataset_config.json"
        )
    }
    effective, effective_dsl = runs.prepare_execution(
        run["run_id"], workflow, workflow["command"], "print('frozen')"
    )
    assert directory not in effective["command"]
    assert "/versions/" in effective["command"]
    assert effective_dsl == effective["command"]
    assert runs.prepare_execution(
        run["run_id"], workflow, workflow["command"], "print('frozen')"
    ) == (effective, effective_dsl)
    project = service.store.bound_project("c")
    assert project is not None
    record = service.store.list_runs(project.id)[0]
    assert "runtime_script" in record.spec.inputs
    saved = record.spec.parameters["definition"]
    assert isinstance(saved, dict)
    saved_definition = saved["definition"]
    assert isinstance(saved_definition, dict)
    saved_inference = saved_definition["inference"]
    assert isinstance(saved_inference, dict)
    assert saved_inference["model_path"] == (
        selected.manifest.storage_ref if model_source == "storage" else reference
    )
    assert ("input.inference.model_path" in record.spec.inputs) == (
        model_source == "storage"
    )
    runs.finish_submission(run["run_id"], run["revision"], "task", "submitted")
    assert service.store.list_runs(project.id)[0].source.task_id == "task"
