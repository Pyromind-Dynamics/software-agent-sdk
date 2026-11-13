import asyncio
import json
from dataclasses import replace
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


@pytest.fixture
def archived_runtime(version_runtime):
    runtime, _, _ = version_runtime
    runtime.versions.archive_runs = True
    return version_runtime


def archived_records(runtime, context):
    snapshots = runtime.versions.snapshots
    assert isinstance(snapshots, FileArtifactSnapshots)
    root = snapshots.path("/.pyromind-agent/c/lineage", context)
    return [json.loads(path.read_text()) for path in root.rglob("lineage.json")]


async def test_run_archives_are_disabled_by_default(version_runtime, monkeypatch):
    runtime, submission, context = version_runtime

    def unexpected_archive(*args):
        pytest.fail("disabled archives must not perform any work")

    monkeypatch.setattr(runtime.versions, "archive_run", unexpected_archive)
    await runtime.submit_versioned_workflow(
        "c", "call", submission.workflow, "workflow = 1", context
    )
    assert archived_records(runtime, context) == []
    assert submission.submissions == [(submission.workflow, "c")]


async def test_run_archives_capture_lineage_without_changing_execution(
    archived_runtime,
    tmp_path,
    monkeypatch,
):
    from pyromind_runtime.infrastructure.sql_database import create_database_engine
    from pyromind_runtime.infrastructure.sql_experiment_index import (
        SQLExperimentIndex,
        experiment_versions,
        metadata,
    )

    runtime, submission, context = archived_runtime
    engine = create_database_engine(f"sqlite:///{tmp_path / 'index.sqlite3'}")
    metadata.create_all(engine)
    monkeypatch.setattr(runtime, "_experiment_index", SQLExperimentIndex(engine))
    context = replace(context, cookie="private-cookie", authorization="private-auth")
    await runtime.submit_versioned_workflow(
        "c", "call", submission.workflow, "workflow = 1", context
    )
    initial = archived_records(runtime, context)
    assert len(initial) == 1
    assert initial[0]["execution"]["task_id"] == "task-123"
    assert initial[0]["execution"]["submission_status"] == "submitted"
    assert initial[0]["execution"]["task_status"] == "pending"
    assert any(
        document["text"] == "workflow = 1"
        for document in initial[0]["documents"].values()
    )
    await runtime.reconcile_versions("c", context)
    assert archived_records(runtime, context) == initial

    snapshots = runtime.versions.snapshots
    assert isinstance(snapshots, FileArtifactSnapshots)
    model = snapshots.path("/models/checkpoint-10", context)
    model.mkdir(parents=True)
    completion = {
        "artifacts": [
            {
                "kind": "model",
                "logical_key": "checkpoint",
                "source": {"storage_ref": "/models/checkpoint-10"},
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
    await runtime.reconcile_versions("c", context)
    records = archived_records(runtime, context)
    finished = next(
        row for row in records if row["execution"]["task_status"] == "succeeded"
    )
    output = next(v for v in finished["artifacts"] if v["producer_run_id"])
    assert output["state"] == "reference_only"
    assert output["manifest"]["storage_ref"] == "/models/checkpoint-10"
    assert output["manifest"]["parents"]
    assert finished["input_snapshot_guarantee"] == "per_artifact_state_only"
    assert "source" not in output["manifest"]
    text = json.dumps(records)
    assert "private-cookie" not in text and "private-auth" not in text
    await runtime.deliver_external_task_status(
        "c", task_id="task-123", status="succeeded", auto_run=False
    )
    await runtime.reconcile_versions("c", context)
    assert archived_records(runtime, context) == records
    assert submission.submissions == [(submission.workflow, "c")]
    project = runtime.versions.store.bound_project("c")
    assert len(runtime.versions.store.list_runs(project.id)) == 1
    assert len(runtime.versions.store.list_versions(project.id)) == 3
    exports = list(
        snapshots.path(
            "/.pyromind-agent/c/projects/" + project.id + "/experiments/v1",
            context,
        ).rglob("experiment.json")
    )
    completed = [
        json.loads(path.read_text())
        for path in exports
        if json.loads(path.read_text())["execution"]["status"] == "succeeded"
    ]
    assert len(completed) == 1
    assert completed[0]["outputs"][0]["state"] == "reference_only"
    assert completed[0]["execution"]["archive_ref"]
    assert len(completed[0]["outputs"]) == 1
    initial_exports = set(exports)
    await runtime.reconcile_versions("c", context)
    assert set(exports[0].parents[2].rglob("experiment.json")) == initial_exports
    try:
        with engine.connect() as conn:
            indexed = conn.execute(experiment_versions.select()).one()
        assert indexed.status == "succeeded"
        assert indexed.document == completed[0]
        assert indexed.archive_ref.endswith("/content/experiment.json")
    finally:
        engine.dispose()


async def test_experiment_archives_share_iteration_directory_on_retry(archived_runtime):
    runtime, submission, context = archived_runtime
    await runtime.submit_versioned_workflow(
        "c", "first", submission.workflow, "workflow = 1", context
    )
    await runtime.deliver_external_task_status(
        "c", task_id="task-123", status="failed", auto_run=False
    )
    await runtime.submit_versioned_workflow(
        "c", "retry", submission.workflow, "workflow = 2", context
    )
    versions = runtime.versions
    project = versions.store.bound_project("c")
    snapshots = versions.snapshots
    root = snapshots.path(
        "/.pyromind-agent/c/projects/" + project.id + "/experiments", context
    )
    assert not (root / "v2").exists()
    attempts = [
        json.loads(path.read_text())["attempts"]
        for path in (root / "v1").rglob("experiment.json")
    ]
    assert any(
        [(item["task_id"], item["status"]) for item in archive]
        == [("task-123", "failed"), ("task-124", "pending")]
        and all(item["archive_ref"] for item in archive)
        for archive in attempts
    )

    await runtime.version_call(
        "c",
        VersionToolRequest(
            action="new_iteration",
            arguments={
                "goal": "Improve training data",
                "expected_revision": project.revision,
            },
        ),
        "next",
        context,
    )
    await runtime.submit_versioned_workflow(
        "c", "next", submission.workflow, "workflow = 3", context
    )
    assert list((root / "v2").rglob("experiment.json"))


@pytest.mark.parametrize("entry", ["direct", "native"])
async def test_run_archive_failure_recovers_without_resubmitting(
    archived_runtime, monkeypatch, entry
):
    runtime, submission, context = archived_runtime
    snapshots = runtime.versions.snapshots
    original = snapshots.save

    def unavailable(source, target, context):
        if source.filename == "lineage.json":
            raise OSError("archive transport unavailable")
        return original(source, target, context)

    monkeypatch.setattr(snapshots, "save", unavailable)
    if entry == "direct":
        result = await runtime.submit_versioned_workflow(
            "c", "call", submission.workflow, "workflow = 1", context
        )
    else:
        result = await runtime.prepare_workflow_execution(
            "c", "/workspace/workflow.py", context, request_id="call"
        )
    assert result["task_id"] == "task-123"
    assert archived_records(runtime, context) == []
    monkeypatch.setattr(snapshots, "save", original)
    await runtime.reconcile_versions("c", context)
    assert len(archived_records(runtime, context)) == 1
    assert len(submission.submissions) == 1


async def test_run_archives_include_persisted_pipeline_result(archived_runtime):
    runtime, submission, context = archived_runtime
    snapshots = runtime.versions.snapshots
    assert isinstance(snapshots, FileArtifactSnapshots)
    for reference in ("/data.jsonl", "/model"):
        path = snapshots.path(reference, context)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
    store = runtime._store("c")
    pipelines = PipelineRuns(
        "c",
        store,
        WorkflowExternalTaskRegistry(runtime.conversation_root),
        runtime.versions,
        context,
    )
    run = pipelines.reserve(
        {
            "input_path": "/data.jsonl",
            "inference": {"model_path": "/model"},
            "dataset_config": {},
            "evaluation_config": {},
        },
        None,
    )
    pipelines.prepare_execution(run["run_id"], {}, "# frozen DSL", "# frozen worker")
    pipelines.finish_submission(
        run["run_id"], run["revision"], "task-pipeline", "submitted"
    )
    stored = store.load_pipeline_runs()[run["run_id"]]
    result = {"selected_step": 10, "meets_target": False}
    store.save_pipeline_run(
        stored.model_copy(update={"result": result, "revision": stored.revision + 1}),
        expected_revision=stored.revision,
    )
    await runtime.reconcile_versions("c", context)
    records = archived_records(runtime, context)
    assert len(records) == 1
    assert records[0]["run"]["source"]["kind"] == "pipeline"
    assert records[0]["execution"]["task_id"] == "task-pipeline"
    assert records[0]["execution"]["result"] == result
    assert any(
        document["text"] == "# frozen worker"
        for document in records[0]["documents"].values()
    )
    assert submission.submissions == []


@pytest.mark.parametrize("entry", ["direct", "native"])
async def test_run_archives_record_uncertain_without_retrying(
    archived_runtime, monkeypatch, entry
):
    runtime, submission, context = archived_runtime
    attempts = []

    async def lost_reply(*args, **kwargs):
        attempts.append(args)
        raise TimeoutError("platform response lost")

    monkeypatch.setattr(submission, "submit", lost_reply)
    for _ in range(2):
        if entry == "direct":
            with pytest.raises(ValueError, match="uncertain"):
                await runtime.submit_versioned_workflow(
                    "c", "call", submission.workflow, "workflow = 1", context
                )
        else:
            result = await runtime.prepare_workflow_execution(
                "c", "/workspace/workflow.py", context, request_id="call"
            )
            assert result["status"] == "uncertain"
    records = archived_records(runtime, context)
    assert len(records) == 1
    assert records[0]["execution"]["submission_status"] == "uncertain"
    assert records[0]["execution"]["task_id"] is None
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "context",
    [
        RequestContext(user_id="other"),
        RequestContext(user_id="42", profile_id="other"),
    ],
)
async def test_run_archives_preserve_authorization(archived_runtime, context):
    runtime, submission, authorized = archived_runtime
    await runtime.submit_versioned_workflow(
        "c", "call", submission.workflow, None, authorized
    )
    initial = archived_records(runtime, authorized)
    with pytest.raises(PermissionError):
        await runtime.reconcile_versions("c", context)
    assert archived_records(runtime, authorized) == initial


async def test_run_archives_recover_after_offline_completion(archived_runtime):
    runtime, submission, context = archived_runtime
    await runtime.submit_versioned_workflow(
        "c", "call", submission.workflow, "workflow = 1", context
    )
    await runtime.close()
    restored = ConversationRuntime(
        runtime.conversation_root,
        FakeAdapter(),
        workflow_submission=submission,
        versions=runtime.versions,
    )
    try:
        await restored.deliver_external_task_status(
            "c", task_id="task-123", status="succeeded", auto_run=False
        )
        assert len(archived_records(restored, context)) == 1
        await restored.recover_version_metadata()
        assert len(archived_records(restored, context)) == 1
        await restored.reconcile_versions("c", context)
        records = archived_records(restored, context)
        assert len(records) == 2
        assert any(row["execution"]["task_status"] == "succeeded" for row in records)
        assert len(submission.submissions) == 1
    finally:
        await restored.close()


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

    async def timeout(*args, **kwargs):
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


async def test_direct_submission_checks_snapshot_before_platform_submit(
    version_runtime, monkeypatch
):
    runtime, submission, context = version_runtime
    prepare_run = runtime.versions.prepare_run

    def tampered_run(*args, **kwargs):
        run = prepare_run(*args, **kwargs)
        project = runtime.versions.store.bound_project("c")
        version = runtime.versions.store.list_versions(project.id)[0]
        runtime.versions.snapshots.path(
            version.manifest.storage_ref, context
        ).write_text("tampered")
        return run

    monkeypatch.setattr(runtime.versions, "prepare_run", tampered_run)
    with pytest.raises(ValueError, match="digest changed"):
        await runtime.prepare_workflow_execution(
            "c", "/workspace/project/workflow.py", context, request_id="request"
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


async def test_successful_retry_delivers_checkpoint_and_local_scores_to_v1(
    archived_runtime, monkeypatch
):
    from .test_evaluation_cases import FakeStorage

    runtime, submission, context = archived_runtime
    for request, task, status in [
        ("first", "task-123", "failed"),
        ("retry", "task-124", "succeeded"),
    ]:
        await runtime.submit_versioned_workflow(
            "c", request, submission.workflow, None, context
        )
        await runtime.deliver_external_task_status(
            "c", task_id=task, status=status, auto_run=False
        )
    checkpoint = "/models/retry/checkpoint-3"
    runtime.versions.snapshots.path(checkpoint, context).mkdir(parents=True)
    nodes = [
        {
            "training_node_id": "train",
            "output_path": "/models/retry",
            "checkpoints": [
                {"step": 3, "kind": "lora", "path": checkpoint, "complete": True}
            ],
            "evaluations": [
                {
                    "step": 3,
                    "path": "/models/retry/eval_step_3.jsonl",
                    "sha256": "a" * 64,
                    "rows": 2,
                }
            ],
        }
    ]
    assert await runtime.record_training_artifacts("c", "task-123", nodes, context) == 0
    with pytest.raises(PermissionError):
        await runtime.record_training_artifacts(
            "c", "task-124", nodes, replace(context, profile_id="other")
        )
    for _ in range(2):
        assert (
            await runtime.record_training_artifacts("c", "task-124", nodes, context)
            == 1
        )

    directory = "/.pyromind-agent/c/public_data"
    report = directory + "/eval/report.md"
    result = {
        "schema_version": 1,
        "task_id": "task-124",
        "training_node_id": "train",
        "dataset_role": "validation",
        "dataset_sha256": "b" * 64,
        "scoring_sha256": "c" * 64,
        "report_path": report,
        "selected_step": 3,
        "steps": [
            {
                "step": 3,
                "checkpoint_path": checkpoint,
                "prediction_sha256": "a" * 64,
                "metrics": {"accuracy": 0.0},
            }
        ],
    }
    names = {
        "eval_result-ok.json",
        "eval_result-invalid.json",
        "eval_result-wrong.json",
    }
    storage = FakeStorage(
        {
            report: b"# Report",
            directory + "/eval_result-ok.json": json.dumps(result).encode(),
            directory + "/eval_result-invalid.json": json.dumps(
                {**result, "selected_step": 99}
            ).encode(),
            directory + "/eval_result-wrong.json": json.dumps(
                {**result, "task_id": "task-123"}
            ).encode(),
        }
    )
    monkeypatch.setattr(storage, "list", lambda _: names)
    monkeypatch.setattr(runtime, "_evaluation_storage_factory", lambda _: storage)

    runtime._active["c"].adapter.emit(
        "c",
        "operation.completed",
        {"operation_id": "eval", "category": "subagent", "name": "Eval"},
    )
    async with asyncio.timeout(3):
        while not await runtime.get_experiment_metrics("c", context):
            await asyncio.sleep(0.01)
    await runtime.reconcile_versions("c", context)
    (experiment,) = (await runtime.get_experiments("c", context)).versions
    assert experiment.version == "v1"
    assert len(experiment.outputs) == 2
    (evaluation,) = experiment.evaluations
    assert evaluation.training_run_id == experiment.attempts[1].run_id
    assert evaluation.report_path == report
    model = next(output for output in experiment.outputs if output.kind == "model")
    assert model.prediction["sha256"] == "a" * 64
    (metric,) = await runtime.get_experiment_metrics("c", context)
    assert metric.metric == "accuracy" and metric.value == 0 and metric.selected


@pytest.mark.parametrize("status", ["succeeded", "failed"])
async def test_inference_callback_registers_result_under_source_checkpoint(
    archived_runtime, monkeypatch, tmp_path, status
):
    from .test_evaluation_cases import FakeStorage

    runtime, submission, context = archived_runtime
    await runtime.submit_versioned_workflow(
        "c", "train", submission.workflow, None, context
    )
    await runtime.deliver_external_task_status(
        "c", task_id="task-123", status="succeeded", auto_run=False
    )
    service = runtime.versions
    checkpoint = "/models/train/checkpoint-12"
    service.snapshots.path(checkpoint, context).mkdir(parents=True)
    service.snapshots.path("/validation.jsonl", context).write_text("{}\n")
    await runtime.record_training_artifacts(
        "c",
        "task-123",
        [
            {
                "training_node_id": "train",
                "output_path": "/models/train",
                "checkpoints": [
                    {"path": checkpoint, "step": 12, "kind": "full", "complete": True}
                ],
            }
        ],
        context,
    )
    project = service.store.bound_project("c")
    (artifact,) = [
        item
        for item in service.store.list_versions(project.id)
        if item.manifest.output_role == "checkpoint"
    ]
    product = runtime._store("c")
    registry = WorkflowExternalTaskRegistry(tmp_path)
    runs = PipelineRuns("c", product, registry, service, context)
    monkeypatch.setattr(runtime, "_external_tasks", registry)
    definition = {
        "input_path": "/validation.jsonl",
        "inference": {"model_path": checkpoint},
        "dataset_config": {},
        "evaluation_config": {"model_source": {"type": "checkpoint", "kind": "full"}},
    }
    with pytest.raises(ValueError, match="actual inference input"):
        runs.reserve(
            {
                **definition,
                "inference": {
                    "model_path": "/models/unrelated",
                    "source_checkpoint_version_id": artifact.id,
                },
            },
            None,
        )
    run = runs.reserve(definition, None)
    assert run["definition"]["inference"]["source_checkpoint_version_id"] == artifact.id
    run_id, root = run["run_id"], run["output_dir"]
    runs.prepare_execution(run_id, {"nodes": []}, "# DSL", "# evaluator")
    runs.finish_submission(run_id, run["revision"], "eval-1", "submitted")
    metrics = {"pass_rate": 0.0, "total": 2}
    manifest = {
        "run_id": run_id,
        "model_reference": checkpoint,
        "generation": {"temperature": 0},
        "dataset_sha256": "a" * 64,
        "evaluation_config_sha256": "b" * 64,
    }
    report = {
        "status": "succeeded",
        "evaluation_status": "validated",
        "metrics": metrics,
        "artifacts": {
            name: f"{root}/{name}" for name in ("metrics.json", "predictions.jsonl")
        },
    }
    if status == "failed":
        report = {"status": "failed", "error": "inference failed"}
    storage = FakeStorage(
        {
            f"{root}/{name}.json": json.dumps(value).encode()
            for name, value in (
                ("run_manifest", manifest),
                ("report", report),
                ("metrics", metrics),
            )
        }
    )
    monkeypatch.setattr(runtime, "_evaluation_storage_factory", lambda _: storage)
    await runtime.deliver_external_task_status(
        "c", task_id="eval-1", status=status, auto_run=False
    )
    before = product.load_pipeline_runs()[run_id]
    await runtime.reconcile_versions("c", context)
    assert product.load_pipeline_runs()[run_id] == before
    (version,) = (await runtime.get_experiments("c", context)).versions
    (evaluation,) = version.evaluations
    assert evaluation.training_run_id == version.execution.run_id
    assert evaluation.report_path == f"{root}/report.json"
    assert evaluation.status == status
    if status == "failed":
        assert await runtime.get_experiment_metrics("c", context) == ()
        return
    assert evaluation.prediction_path == f"{root}/predictions.jsonl"
    (metric,) = await runtime.get_experiment_metrics("c", context, metric="pass_rate")
    assert (metric.version, metric.step, metric.value, metric.selected) == (
        "v1",
        12,
        0.0,
        False,
    )
