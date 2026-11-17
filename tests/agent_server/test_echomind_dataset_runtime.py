import json
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest

from openhands.agent_server.echomind_dataset_runtime import (
    DatasetExport,
    DatasetRuntimeError,
    DatasetRuntimeExecutor,
    HttpTracerBatchController,
    PlatformDatasetRuntimeExecutor,
    StorageDatasetSourceProvider,
    TracerBatchReference,
    TracerBatchSourceProvider,
)
from openhands.agent_server.echomind_graph_runs import TaskRun
from openhands.agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectNodeOutput,
    ProjectRecord,
    ProjectStore,
)
from openhands.agent_server.echomind_runtime_contracts import RuntimeStageContext
from openhands.agent_server.echomind_storage import canonical_sha256
from openhands.agent_server.echomind_train_runtime import (
    TrainTaskState,
    TrainWorkflowBackend,
)


def test_dataset_executor_materializes_standard_train_and_validation_ports():
    executor = DatasetRuntimeExecutor()

    result = executor.prepare(
        _context(
            inputs={
                "source_batch": {
                    "kind": "dataset.materialized",
                    "train_uri": "storage://datasets/train.jsonl",
                    "val_uri": "storage://datasets/val.jsonl",
                    "train_samples": 8,
                    "val_samples": 2,
                    "source_uri": "storage://datasets/raw",
                }
            }
        )
    )

    assert result.status == "completed"
    artifacts = {artifact.port: artifact for artifact in result.artifacts}
    assert set(artifacts) == {"train_messages", "val_messages"}
    assert artifacts["train_messages"].payload.kind == "dataset.messages.train"
    assert artifacts["train_messages"].payload.content["uri"] == (
        "storage://datasets/train.jsonl"
    )
    assert artifacts["val_messages"].payload.kind == ("dataset.messages.validation")
    assert artifacts["val_messages"].payload.content["sample_count"] == 2


def test_storage_provider_reads_materialization_manifest(tmp_path):
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "train_uri": "/storage/train.jsonl",
                "val_uri": "/storage/val.jsonl",
                "train_samples": 3,
                "val_samples": 2,
            }
        ),
        encoding="utf-8",
    )

    exported = StorageDatasetSourceProvider().materialize(
        source=str(manifest),
        context=_context(),
    )

    assert exported.train_uri == "/storage/train.jsonl"
    assert exported.train_samples == 3


def test_dataset_executor_rejects_incomplete_source():
    result = DatasetRuntimeExecutor().prepare(
        _context(inputs={"source_batch": {"source_uri": "/datasets/raw"}})
    )

    assert result.status == "failed"
    assert result.failure_category == "validation"
    assert "train_uri" in (result.error or "")


def test_tracer_provider_claims_exports_and_releases_on_failure():
    controller = _FakeBatchController()
    provider = TracerBatchSourceProvider(
        controller=controller,
        exporter=_FakeDatasetExporter(),
        expected_tracer_resource_id="tracer-1",
    )

    exported = provider.materialize(
        source={
            "resource_id": "tracer-1",
            "batch_id": "7",
            "batch_index": 7,
            "uri": "tracer://tracer-1/batches/7",
        },
        context=_context(),
    )

    assert exported.train_samples == 8
    assert exported.stats["batch_id"] == "7"
    assert controller.claimed == ["tracer-1:7"]

    provider.release(
        source=TracerBatchReference(
            resource_id="tracer-1",
            batch_id="7",
            batch_index=7,
        ),
        context=_context(),
    )
    assert controller.released == ["tracer-1:7"]


def test_http_tracer_batch_controller_uses_admin_batch_routes():
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        assert request.headers["authorization"] == "Bearer admin-secret"
        return httpx.Response(200, json={"ok": True, "note": "already_in_training"})

    controller = HttpTracerBatchController(
        endpoint="https://tracer.example.test/",
        admin_key="admin-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    note = controller.claim(TracerBatchReference(batch_id="7"))
    controller.release(TracerBatchReference(batch_id="7"))

    assert note == "already_in_training"
    assert requests == ["/admin/batches/7/claim", "/admin/batches/7/release"]


def test_http_tracer_batch_controller_requires_numeric_index():
    controller = HttpTracerBatchController(
        endpoint="https://tracer.example.test",
        admin_key="admin-secret",
    )

    with pytest.raises(DatasetRuntimeError, match="numeric index"):
        controller.claim(TracerBatchReference(batch_id="batch-alpha"))


@pytest.fixture
def platform_dataset(tmp_path):
    context = _context(inputs={"source_batch": "/workspace/raw/batch 2"})
    context = context.model_copy(update={"project_id": str(uuid4())})
    workflow = {
        "nodes": [{"id": "dataset", "data": {"config": {"command": "original"}}}]
    }
    node = context.node.model_copy(
        update={
            "config": {
                "dataset_runtime": {
                    "workflow": workflow,
                    "output_uri": "/workspace/exports/{graph_run_id}/{stage_run_id}",
                    "command_template": (
                        "source_path={source_batch}; out={output_uri}; "
                        'run "${{source_path}}" "$out"'
                    ),
                    "command_binding_path": "nodes[0].data.config.command",
                }
            }
        }
    )
    context = context.model_copy(update={"node": node})
    ProjectStore(tmp_path).save(
        ProjectInfo(
            project=ProjectRecord(
                project_id=context.project_id,
                name="Dataset",
                description="Dataset workflow test",
                skill="experiment-planning",
                conversation_id="conversation-1",
                cluster="us-west-1#pre",
            ),
            graph=ProjectGraph(
                graph_id=str(uuid4()),
                project_id=context.project_id,
                nodes=[node],
            ),
        )
    )
    backend = Mock(spec=TrainWorkflowBackend)
    backend.find_task_by_name.return_value = None
    backend.submit_workflow.return_value = "task-12"
    executor = DatasetRuntimeExecutor(
        workflow_executor=PlatformDatasetRuntimeExecutor(
            projects_root=tmp_path, backend=backend
        ),
    )
    task = TaskRun(
        task_run_id=str(uuid4()),
        stage_run_id=context.stage_run_id,
        provider="pyromind.studio.dataset",
        idempotency_key="dataset-key",
        request_digest=canonical_sha256({"task": "dataset"}),
    )
    return executor, context, task, backend


def test_platform_dataset_binds_inputs_and_emits_both_splits(platform_dataset):
    executor, context, task, backend = platform_dataset
    output_uri = "/workspace/exports/graph-run-1/stage-run-1"
    prepared = executor.prepare(context)

    assert prepared.status == "waiting_external"
    assert prepared.external_task is not None
    assert prepared.external_task.metadata["output_uri"] == output_uri
    backend.submit_workflow.assert_not_called()

    submitted = executor.submit(context, task)
    assert submitted.provider_run_id == "task-12"
    spec = backend.submit_workflow.call_args.kwargs["spec"]
    command = spec.workflow["nodes"][0]["data"]["config"]["command"]
    assert command == (
        f"source_path='/workspace/raw/batch 2'; out={output_uri}; "
        'run "${source_path}" "$out"'
    )
    assert (
        context.node.config["dataset_runtime"]["workflow"]["nodes"][0]["data"][
            "config"
        ]["command"]
        == "original"
    )

    backend.query_task.return_value = TrainTaskState(
        status="succeeded",
        message="Completed",
        output_uri=output_uri,
    )
    completed = executor.poll(
        context,
        task.model_copy(update={"provider_run_id": "task-12"}),
    )

    assert completed.status == "completed"
    assert {
        artifact.port: artifact.metadata["source_uri"]
        for artifact in completed.artifacts
    } == {
        "train_messages": f"{output_uri}/train_messages.jsonl",
        "val_messages": f"{output_uri}/val_messages.jsonl",
    }
    assert all(
        artifact.metadata["provider_task_id"] == "task-12"
        for artifact in completed.artifacts
    )


def test_platform_dataset_reconciles_unknown_submit_without_resubmitting(
    platform_dataset,
):
    executor, context, task, backend = platform_dataset
    uncertain = task.model_copy(update={"status": "unknown"})

    with pytest.raises(DatasetRuntimeError, match="Could not confirm"):
        executor.submit(context, uncertain)

    backend.find_task_by_name.return_value = "task-12"
    submitted = executor.submit(context, uncertain)

    assert submitted.provider_run_id == "task-12"
    backend.submit_workflow.assert_not_called()


class _FakeBatchController:
    def __init__(self) -> None:
        self.claimed: list[str] = []
        self.released: list[str] = []

    def claim(self, batch: TracerBatchReference) -> str:
        self.claimed.append(f"{batch.resource_id}:{batch.batch_id}")
        return "claimed"

    def release(self, batch: TracerBatchReference) -> None:
        self.released.append(f"{batch.resource_id}:{batch.batch_id}")


class _FakeDatasetExporter:
    def export(
        self,
        *,
        batch: TracerBatchReference,
        context: RuntimeStageContext,
    ) -> DatasetExport:
        del batch, context
        return DatasetExport(
            train_uri="storage://datasets/train.jsonl",
            val_uri="storage://datasets/val.jsonl",
            train_samples=8,
            val_samples=2,
        )


def _context(inputs: dict[str, object] | None = None) -> RuntimeStageContext:
    return RuntimeStageContext(
        project_id="project-1",
        release_id="release-1",
        graph_run_id="graph-run-1",
        stage_run_id="stage-run-1",
        node=ProjectNode(
            id="dataset",
            type="dataset",
            title="Dataset",
            inputs={
                "source_batch": ProjectNodeInput(
                    artifact_type="dataset.source",
                    schema_ref="echomind://schemas/dataset-source/v1",
                    binding=ProjectNodeInputBinding(
                        kind="run_input",
                        name="data",
                    ),
                )
            },
            outputs={
                "train_messages": ProjectNodeOutput(
                    artifact_type="dataset.messages.train",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                ),
                "val_messages": ProjectNodeOutput(
                    artifact_type="dataset.messages.validation",
                    schema_ref="echomind://schemas/messages-dataset/v1",
                ),
            },
        ),
        inputs=inputs or {},
    )
