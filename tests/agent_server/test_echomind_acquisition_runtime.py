from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from pyromind_agent_server.echomind_acquisition import (
    AcquisitionBatch,
    AcquisitionSimulation,
    AcquisitionSimulationStore,
    AcquisitionSource,
)
from pyromind_agent_server.echomind_acquisition_runtime import (
    AcquisitionPaused,
    AcquisitionRuntimeError,
    AcquisitionSimulationRunner,
    HttpTracerAcquisitionTransport,
    JsonlAcquisitionSampleSource,
    TracerBatchMaterializer,
    TracerCollectionStatus,
    _platform_task_id,
    _task_id,
)
from pyromind_agent_server.echomind_dataset_runtime import (
    DatasetRuntimeExecutor,
    DatasetWorkflowSpec,
    PlatformDatasetRuntimeExecutor,
)
from pyromind_agent_server.echomind_graph_runs import GraphRunStore
from pyromind_agent_server.echomind_projects import (
    ProjectEdge,
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectNodeOutput,
    ProjectRecord,
    ProjectStore,
)
from pyromind_agent_server.echomind_publication import (
    PromotionPolicy,
    PublicationStore,
    PublishedGraph,
    RuntimeRequirements,
)
from pyromind_agent_server.echomind_runtime_scheduler import (
    ReadyBatch,
    RuntimeGraphScheduler,
)
from pyromind_agent_server.echomind_runtime_slots import (
    RuntimeHealth,
    RuntimeSlot,
    RuntimeSlotStore,
    ServingInferenceRuntime,
    TracerBufferConfig,
    TracerRuntime,
)
from pyromind_agent_server.echomind_storage import canonical_sha256
from pyromind_runtime.ports.storage import EvaluationStorageMissingError


def test_runner_feeds_two_rounds_and_dispatches_each_batch(tmp_path, monkeypatch):
    project_id = "6a1170d0-c1ce-4e37-9cd8-336fcfcff625"
    project = cast(
        ProjectInfo,
        SimpleNamespace(project=SimpleNamespace(project_id=project_id)),
    )
    slot = _slot(project_id)
    reader = _MemoryReader(
        {
            "datasets/avi/train.jsonl": _jsonl(
                [
                    _sample_row("sample-1", "images/1.jpg"),
                    _sample_row("sample-2", "images/2.jpg"),
                    _sample_row("sample-3", "images/3.jpg"),
                    _sample_row("sample-4", "images/4.jpg"),
                    _sample_row("sample-1", "images/1.jpg"),
                ]
            ),
            "datasets/avi/images/1.jpg": b"image-1",
            "datasets/avi/images/2.jpg": b"image-2",
            "datasets/avi/images/3.jpg": b"image-3",
            "datasets/avi/images/4.jpg": b"image-4",
        }
    )
    transport = _FakeTransport(block_once=True)
    scheduler = _FakeScheduler(wait_for_recovery=True)
    store = AcquisitionSimulationStore(tmp_path / "projects")
    simulation = _simulation(store=store, project_id=project_id)
    sleeps: list[float] = []
    runner = AcquisitionSimulationRunner(
        projects_root=str(tmp_path / "projects"),
        project=project,
        published=cast(PublishedGraph, SimpleNamespace()),
        slot=slot,
        tracer_secret="tracer-secret",
        sample_source=JsonlAcquisitionSampleSource(reader),
        transport=transport,
        scheduler=cast(RuntimeGraphScheduler, scheduler),
        sleep=sleeps.append,
        poll_seconds=1,
        backpressure_timeout_seconds=5,
        dispatch_timeout_seconds=5,
    )

    def active(run_project_id: str):
        assert run_project_id == project_id
        return SimpleNamespace(run_id="run-batch-00000000", status="running")

    monkeypatch.setattr(runner.graph_run_store, "active", active)
    completed = runner.run(simulation)

    assert completed.status == "succeeded"
    assert completed.completed_samples == 4
    assert completed.completed_rounds == 2
    assert completed.batch_ids == ["batch-00000000", "batch-00000001"]
    assert completed.graph_run_ids == [
        "run-batch-00000000",
        "run-batch-00000001",
    ]
    assert len(completed.attempted_sample_ids) == 4
    assert len(transport.chat_task_ids) == 4
    assert len(set(transport.chat_task_ids)) == 4
    assert transport.end_task_ids == transport.chat_task_ids
    assert transport.seal_count == 2
    assert sleeps == [1, 1]
    assert scheduler.recoveries == [(project_id, "run-batch-00000000")]

    unchanged = runner.run(completed)
    assert unchanged == completed
    assert len(transport.chat_task_ids) == 4


def test_runner_pauses_and_preserves_progress_when_tracer_is_unavailable(tmp_path):
    project_id = "3ee67d2f-b2ec-4a9f-b23e-0d2379ea3571"
    project = cast(
        ProjectInfo,
        SimpleNamespace(project=SimpleNamespace(project_id=project_id)),
    )
    store = AcquisitionSimulationStore(tmp_path / "projects")
    simulation = _simulation(store=store, project_id=project_id)
    runner = AcquisitionSimulationRunner(
        projects_root=str(tmp_path / "projects"),
        project=project,
        published=cast(PublishedGraph, SimpleNamespace()),
        slot=_slot(project_id),
        tracer_secret="tracer-secret",
        sample_source=JsonlAcquisitionSampleSource(_MemoryReader({})),
        transport=_UnavailableTransport(),
        scheduler=cast(RuntimeGraphScheduler, _FakeScheduler()),
        sleep=lambda _seconds: None,
        dispatch_timeout_seconds=0,
    )

    paused = runner.run(simulation)

    assert paused.status == "paused"
    assert paused.attempted_sample_ids == []
    assert "AcquisitionRuntimeError" in (paused.error or "")


@pytest.mark.parametrize("platform_batch_trigger", [False, True])
def test_sealed_tracer_batch_materializes_platform_dataset_and_recovers(
    tmp_path, platform_batch_trigger
):
    project_id = "6a1170d0-c1ce-4e37-9cd8-336fcfcff625"
    conversation_id = "f5a9bf4181ed4a25b75bc545d217a1f1"
    root = tmp_path / "projects"
    project, published, slot = _published_platform_inputs(
        root, project_id=project_id, conversation_id=conversation_id
    )
    store = AcquisitionSimulationStore(root)
    simulation = _simulation(store=store, project_id=project_id).model_copy(
        update={
            "requested_samples": 2,
            "requested_rounds": 1,
            "splits": ["train", "val"],
        }
    )
    store.save(simulation)
    image_root = "/workspace/datasets/avi/images"
    train_images = [f"{image_root}/{index}.jpg" for index in range(1, 4)]
    val_images = [f"{image_root}/{index}.jpg" for index in range(4, 7)]
    ground_truth = {
        "boxes": [{"x": 2, "y": 3}],
        "reason": "scratch",
        "result": "defect",
    }
    train_row = {
        **_sample_row("training", train_images[0]),
        "images": train_images,
        "gt": ground_truth,
    }
    val_row = {
        **_sample_row("validation", "images/4.jpg"),
        "images": [f"images/{index}.jpg" for index in range(4, 7)],
        "gt": ground_truth,
    }
    storage = _BatchStorage(
        {
            "datasets/avi/train.jsonl": _jsonl([train_row]),
            "datasets/avi/val.jsonl": _jsonl([val_row]),
            **{
                path.removeprefix("/workspace/"): b"image"
                for path in train_images + val_images
            },
        }
    )
    sample_source = JsonlAcquisitionSampleSource(storage)
    samples = sample_source.load(
        source=simulation.source, splits=["train", "val"], limit=2, balance_splits=True
    )
    task_ids = [
        _platform_task_id(simulation.simulation_id, sample)
        if platform_batch_trigger
        else _task_id(simulation.simulation_id, sample.sample_id)
        for sample in samples
    ]
    prefix = "/EchoMind/tracer-1/data/spans/0"
    storage.files[f"{prefix}/spans_20261113_120000.parquet"] = _tracer_shard(
        task_ids, ground_truth=None if platform_batch_trigger else train_row["gt"]
    )
    transport = (
        _FakeTransport(legacy_status=True)
        if platform_batch_trigger
        else _BatchTransport()
    )
    backend = Mock()
    backend.find_task_by_name.return_value = None
    backend.submit_workflow.return_value = "dataset-task-1"
    scheduler = RuntimeGraphScheduler(
        projects_root=root,
        executors={
            "dataset_runtime": DatasetRuntimeExecutor(
                workflow_executor=PlatformDatasetRuntimeExecutor(
                    projects_root=root, backend=backend
                )
            )
        },
    )
    runner = AcquisitionSimulationRunner(
        projects_root=str(root),
        project=project,
        published=published,
        slot=slot,
        tracer_secret="tracer-secret",
        tracer_admin_secret="admin-secret",
        sample_source=sample_source,
        batch_materializer=TracerBatchMaterializer(storage),
        platform_batch_trigger=platform_batch_trigger,
        backpressure_timeout_seconds=0,
        transport=transport,
        scheduler=scheduler,
    )
    if platform_batch_trigger:
        partial_round = store.save(
            simulation.model_copy(update={"requested_samples": 1})
        )
        rejected = runner.run(partial_round)
        assert "requires 2 samples per round" in (rejected.error or "")
        assert transport.chat_task_ids == []
        store.save(simulation)

        waiting = runner.run(simulation)
        assert waiting.status == "paused"
        assert "trigger marker" in (waiting.error or "")
        assert waiting.completed_samples == 2
        assert waiting.completed_rounds == 0
        assert transport.end_task_ids == task_ids
        assert waiting.collecting_batch is not None
        assert waiting.collecting_batch.index == 0
        assert GraphRunStore(root).list(project_id) == []
        backend.submit_workflow.assert_not_called()
        marker = "/EchoMind/tracer-1/data/project-batches/0.json"
        storage.files[marker] = b'{"batch_index": 1}'
        rejected = runner.run(waiting)
        assert "invalid trigger marker" in (rejected.error or "")
        assert rejected.completed_rounds == 0
        storage.files[marker] = b'{"batch_index": 0}'

        storage.files["datasets/avi/train.jsonl"] = _jsonl(
            [{**train_row, "gt": {"result": "pass"}}]
        )
        unmatched = runner.run(rejected)
        assert "changed source row" in (unmatched.error or "")
        assert GraphRunStore(root).list(project_id) == []
        storage.files["datasets/avi/train.jsonl"] = _jsonl([train_row])
        simulation = unmatched

    paused = runner.run(simulation)
    assert paused.status == "paused"
    assert len(paused.batch_ids) == 1, paused.error
    [batch_id] = paused.batch_ids
    assert paused.batches[batch_id].claimed
    assert paused.batches[batch_id].input_directory is None
    assert paused.completed_samples == 2
    assert GraphRunStore(root).list(project_id) == []
    backend.submit_workflow.assert_not_called()

    succeeded = runner.run(paused)
    assert succeeded.status == "succeeded"
    assert transport.chat_task_ids == task_ids
    assert transport.end_task_ids == task_ids
    if isinstance(transport, _BatchTransport):
        assert transport.claimed == [0]
        assert transport.seal_count == 1
    else:
        assert transport.seal_count == 0
    assert storage.written == ["train.jsonl", "val.jsonl"]
    directory = succeeded.batches[batch_id].input_directory
    assert directory is not None
    assert directory.startswith(f"/workspace/.pyromind-agent/{conversation_id}/")
    [run] = GraphRunStore(root).list(project_id)
    assert run.status == "waiting_external"
    assert run.inputs["data"] == directory
    assert run.inputs["base_model"] == "/workspace/models/training"
    assert run.inputs["tracer_batch"]["batch_id"] == batch_id
    backend.submit_workflow.assert_called_once()
    spec = backend.submit_workflow.call_args.kwargs["spec"]
    assert isinstance(spec, DatasetWorkflowSpec)
    assert spec.workflow is not None
    command = spec.workflow["nodes"][0]["data"]["config"]["command"]
    assert command == f"convert {directory} {spec.output_directory}"
    for split, source, images in (
        ("train", train_row, train_images),
        ("val", val_row, val_images),
    ):
        payload = storage.files[f"{directory.removeprefix('/workspace')}/{split}.jsonl"]
        row = json.loads(payload)
        assert row["id"] == source["id"]
        assert re.match(
            r'\{"result"\s*:\s*"(?:pass|defect|fail)"', row["assistant_output"]
        )
        assert json.loads(row["assistant_output"]) == source["gt"]
        assert row["label"] == "defect"
        assert row["images"] == images


@pytest.mark.parametrize(
    ("platform_batch_trigger", "extra_task", "error"),
    [
        (False, True, "another producer"),
        (True, True, "another producer"),
        (True, False, "ground truth differs"),
    ],
)
def test_materializer_refuses_a_batch_with_unverified_rows(
    tmp_path, platform_batch_trigger, extra_task, error
):
    project_id = "6a1170d0-c1ce-4e37-9cd8-336fcfcff625"
    simulation = _simulation(
        store=AcquisitionSimulationStore(tmp_path / "projects"),
        project_id=project_id,
    )
    row = _sample_row("training", "images/1.jpg")
    storage = _BatchStorage({"datasets/avi/train.jsonl": _jsonl([row])})
    samples = JsonlAcquisitionSampleSource(storage).load(
        source=simulation.source, splits=["train"], limit=1
    )
    task_id = (
        _platform_task_id(simulation.simulation_id, samples[0])
        if platform_batch_trigger
        else _task_id(simulation.simulation_id, row["id"])
    )
    task_ids = [task_id, "foreign"] if extra_task else [task_id]
    gt = row["gt"] if extra_task else {"result": "pass"}
    storage.files["/EchoMind/tracer-1/data/spans/0/spans_0.parquet"] = _tracer_shard(
        task_ids, ground_truth=None if platform_batch_trigger and extra_task else gt
    )
    with pytest.raises(AcquisitionPaused, match=error):
        TracerBatchMaterializer(storage).rows(
            tracer_resource_id="tracer-1",
            simulation_id=simulation.simulation_id,
            batch=AcquisitionBatch(index=0, sample_ids=[row["id"]]),
            samples=samples,
            completed_sample_ids={row["id"]} if platform_batch_trigger else None,
        )
    assert not storage.written


def _published_platform_inputs(
    root, *, project_id: str, conversation_id: str
) -> tuple[ProjectInfo, PublishedGraph, RuntimeSlot]:
    def input_for(name: str) -> ProjectNodeInput:
        return ProjectNodeInput(
            artifact_type="fixture",
            schema_ref="fixture",
            binding=ProjectNodeInputBinding(kind="run_input", name=name),
        )

    nodes = [
        ProjectNode(
            id="dataset",
            type="dataset",
            title="Dataset",
            executor="dataset_runtime",
            config={
                "dataset_runtime": {
                    "workflow": {"nodes": [{"data": {"config": {"command": ""}}}]},
                    "command_binding_path": "nodes[0].data.config.command",
                    "command_template": "convert {source_batch} {output_uri}",
                    "output_uri": "/workspace/runs/{graph_run_id}/dataset",
                }
            },
            inputs={"source_batch": input_for("data")},
            outputs={
                "train_messages": ProjectNodeOutput(
                    artifact_type="dataset.messages.train", schema_ref="fixture"
                ),
                "val_messages": ProjectNodeOutput(
                    artifact_type="dataset.messages.validation", schema_ref="fixture"
                ),
            },
        ),
        ProjectNode(
            id="train",
            type="train",
            title="Train",
            executor="train_runtime",
            inputs={"base_model": input_for("base_model")},
            config={
                "train_runtime": {
                    "workflow": {
                        "nodes": [
                            {
                                "data": {
                                    "config": {
                                        "model_path": "/workspace/models/training"
                                    }
                                }
                            }
                        ]
                    },
                    "workflow_bindings": {
                        "nodes[0].data.config.model_path": "base_model"
                    },
                }
            },
        ),
    ]
    release_id, graph_id = str(uuid4()), str(uuid4())
    edges = [ProjectEdge(source="dataset", target="train")]
    published = PublishedGraph.build(
        release_id=release_id,
        project_id=project_id,
        source_graph_id=graph_id,
        source_graph_digest=canonical_sha256({"source": "tracer fixture"}),
        runtime_requirements=RuntimeRequirements(),
        promotion_policy=PromotionPolicy(metric="accuracy"),
        publication_mode="production",
        nodes=nodes,
        edges=edges,
        published_by="test",
    )
    project = ProjectStore(root).save(
        ProjectInfo(
            project=ProjectRecord(
                project_id=project_id,
                name="test",
                description="test",
                skill="test",
                conversation_id=conversation_id,
                cluster="test",
                lifecycle="frozen",
                active_release_id=release_id,
            ),
            graph=ProjectGraph(
                graph_id=graph_id,
                project_id=project_id,
                lifecycle="frozen",
                nodes=nodes,
                edges=edges,
            ),
        )
    )
    PublicationStore(root).activate(published)
    tracer = _slot(project_id).tracer
    assert tracer is not None
    health = RuntimeHealth(status="healthy")
    slot = RuntimeSlotStore(root).save(
        RuntimeSlot(
            project_id=project_id,
            status="ready",
            batch_execution_available=True,
            active_release_id=release_id,
            tracer=tracer.model_copy(
                update={
                    "admin_secret_ref": "secret://test/admin",
                    "proxy_secret_ref": "secret://test/proxy",
                    "target_inference_resource_id": "inference-1",
                    "health": health,
                }
            ),
            serving_inference=ServingInferenceRuntime(
                resource_id="inference-1",
                endpoint="https://inference.test",
                secret_ref="secret://test/inference",
                model_uri="/workspace/models/serving",
                health=health,
            ),
        )
    )
    return project, published, slot


def _tracer_shard(task_ids: list[str], *, ground_truth: dict[str, str] | None) -> bytes:
    attributes = (
        {"task.gt": json.dumps(ground_truth)} if ground_truth is not None else {}
    )
    table = pa.Table.from_pylist(
        [
            {
                "name": "llm.completion",
                "rollout_id": task_id,
                "attributes": json.dumps(attributes),
            }
            for task_id in task_ids
        ]
    )
    output = BytesIO()
    parquet.write_table(table, output)
    return output.getvalue()


def test_http_transport_uses_tracer_task_and_batch_endpoints():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/status":
            return httpx.Response(
                200,
                json={
                    "capacity": 20,
                    "held": 0,
                    "buffer_size": 2,
                    "batch_mode": True,
                },
            )
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "origin-model"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"result": "ok", "reason": "clear"}'}}
                    ]
                },
            )
        if request.url.path == "/end_task":
            return httpx.Response(200, json={"task_id": "task-1"})
        if request.url.path == "/admin/flush":
            return httpx.Response(200, json={"sealed_batch_index": 3})
        if request.url.path == "/admin/batches":
            return httpx.Response(
                200, json={"ok": True, "batch_mode": True, "batches": []}
            )
        if request.url.path == "/admin/batches/3/claim":
            return httpx.Response(200, json={"ok": True, "claimed": True})
        return httpx.Response(404)

    transport = HttpTracerAcquisitionTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    endpoint = "http://tracer.test"

    status = transport.status(endpoint=endpoint, access_key="secret")
    assert transport.models(endpoint=endpoint, access_key="secret") == ["origin-model"]
    response = transport.chat_completion(
        endpoint=endpoint,
        access_key="secret",
        model="origin-model",
        task_id="task-1",
        messages=[{"role": "user", "content": "inspect"}],
    )
    transport.end_task(
        endpoint=endpoint,
        access_key="secret",
        task_id="task-1",
        final_reward=1.0,
        ground_truth='{"result": "ok", "reason": "clear"}',
    )
    sealed = transport.seal_batch(endpoint=endpoint, admin_key="admin-secret")
    transport.check_batch_admin(endpoint=endpoint, admin_key="admin-secret")
    assert transport.claim_batch(
        endpoint=endpoint, admin_key="admin-secret", batch_index=3
    )

    assert status.capacity == 20
    assert response["choices"][0]["message"]["content"]
    assert sealed == 3
    assert [request.url.path for request in requests] == [
        "/status",
        "/v1/models",
        "/v1/chat/completions",
        "/end_task",
        "/admin/flush",
        "/admin/batches",
        "/admin/batches/3/claim",
    ]
    assert requests[2].headers["Authorization"] == "Bearer secret"
    assert requests[2].headers["X-Task-ID"] == "task-1"
    assert requests[2].headers["X-Task-Trace"] == "true"
    assert requests[4].headers["Authorization"] == "Bearer admin-secret"


class _MemoryReader:
    def __init__(self, files: Mapping[str, bytes]) -> None:
        self.files = dict(files)

    def read(self, path: str) -> bytes:
        return self.files[path]


class _BatchStorage(_MemoryReader):
    def __init__(self, files: Mapping[str, bytes]) -> None:
        super().__init__(files)
        self.written: list[str] = []

    def read(self, path: str) -> bytes:
        if path not in self.files:
            raise EvaluationStorageMissingError(path)
        return super().read(path)

    def list(self, prefix: str) -> set[str]:
        return {
            path.removeprefix(prefix + "/").split("/", 1)[0]
            for path in self.files
            if path.startswith(prefix + "/")
        }

    def write(self, path: str, payload: bytes) -> None:
        basename = path.rsplit("/", 1)[-1]
        if basename == "val.jsonl" and self.written == ["train.jsonl"]:
            self.written.append("interrupted")
            raise OSError("simulated interrupted upload")
        if self.written and self.written[-1] == "interrupted":
            self.written.pop()
        self.written.append(basename)
        self.files[path] = payload


class _FakeTransport:
    def __init__(
        self, *, block_once: bool = False, legacy_status: bool = False
    ) -> None:
        self.block_once = block_once
        self.legacy_status = legacy_status
        self.status_count = 0
        self.chat_task_ids: list[str] = []
        self.end_task_ids: list[str] = []
        self.seal_count = 0
        self.current_batch_index = 0

    def status(
        self,
        *,
        endpoint: str,
        access_key: str,
    ) -> TracerCollectionStatus:
        del endpoint, access_key
        self.status_count += 1
        if self.legacy_status:
            return TracerCollectionStatus(collected_data_size=len(self.end_task_ids))
        held = 4 if self.block_once and self.status_count == 2 else 1
        if self.status_count == 2 and self.block_once:
            self.block_once = False
        return TracerCollectionStatus(
            capacity=4,
            held=held,
            buffer_size=2,
            current_batch_index=self.current_batch_index,
            ready_count=self.current_batch_index,
            inflight_count=min(self.current_batch_index, 1),
            batch_mode=True,
        )

    def models(self, *, endpoint: str, access_key: str) -> list[str]:
        del endpoint, access_key
        return ["tracer-model"]

    def chat_completion(
        self,
        *,
        endpoint: str,
        access_key: str,
        model: str,
        task_id: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        assert endpoint == "http://tracer.test"
        assert access_key == "tracer-secret"
        assert model == "tracer-model"
        assert messages
        self.chat_task_ids.append(task_id)
        return {
            "choices": [
                {"message": {"content": '{"result": "defect", "reason": "scratch"}'}}
            ]
        }

    def end_task(
        self,
        *,
        endpoint: str,
        access_key: str,
        task_id: str,
        final_reward: float,
        ground_truth: str,
    ) -> Mapping[str, Any]:
        del endpoint, access_key, ground_truth
        assert final_reward == 1.0
        self.end_task_ids.append(task_id)
        return {"task_id": task_id}

    def seal_batch(
        self,
        *,
        endpoint: str,
        admin_key: str,
    ) -> int | None:
        assert endpoint == "http://tracer.test"
        assert admin_key == "tracer-secret"
        sealed = self.current_batch_index
        self.current_batch_index += 1
        self.seal_count += 1
        return sealed

    def check_batch_admin(self, *, endpoint: str, admin_key: str) -> None:
        raise AssertionError("Unexpected batch admin probe")

    def claim_batch(self, *, endpoint: str, admin_key: str, batch_index: int) -> bool:
        raise AssertionError("Unexpected batch claim")


class _UnavailableTransport(_FakeTransport):
    def status(
        self,
        *,
        endpoint: str,
        access_key: str,
    ) -> TracerCollectionStatus:
        del endpoint, access_key
        raise AcquisitionRuntimeError("Tracer is unavailable")


class _BatchTransport(_FakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.claimed: list[int] = []

    def status(self, *, endpoint: str, access_key: str) -> TracerCollectionStatus:
        return (
            super()
            .status(endpoint=endpoint, access_key=access_key)
            .model_copy(update={"buffer_size": 4})
        )

    def check_batch_admin(self, *, endpoint: str, admin_key: str) -> None:
        assert endpoint == "http://tracer.test"
        assert admin_key == "admin-secret"

    def seal_batch(self, *, endpoint: str, admin_key: str) -> int | None:
        self.check_batch_admin(endpoint=endpoint, admin_key=admin_key)
        return super().seal_batch(endpoint=endpoint, admin_key="tracer-secret")

    def claim_batch(self, *, endpoint: str, admin_key: str, batch_index: int) -> bool:
        self.check_batch_admin(endpoint=endpoint, admin_key=admin_key)
        self.claimed.append(batch_index)
        return True


class _FakeScheduler:
    def __init__(self, *, wait_for_recovery: bool = False) -> None:
        self.runs: dict[str, SimpleNamespace] = {}
        self.batches: dict[str, ReadyBatch] = {}
        self.wait_for_recovery = wait_for_recovery
        self.recoveries: list[tuple[str, str]] = []

    def dispatch_ready_batches(
        self,
        *,
        project: ProjectInfo,
        published: PublishedGraph,
        source: Any,
        tracer_resource_id: str,
    ) -> list[SimpleNamespace]:
        del published
        started: list[SimpleNamespace] = []
        for batch in cast(
            list[ReadyBatch],
            source.list_ready_batches(
                project_id=project.project.project_id,
                tracer_resource_id=tracer_resource_id,
            ),
        ):
            if batch.batch_id in self.runs:
                continue
            if self.wait_for_recovery and self.runs and not self.recoveries:
                break
            run = SimpleNamespace(run_id=f"run-{batch.batch_id}")
            self.runs[batch.batch_id] = run
            self.batches[batch.batch_id] = batch
            started.append(run)
        return started

    def recover(self, *, project: ProjectInfo, run_id: str) -> None:
        self.recoveries.append((project.project.project_id, run_id))


def _slot(project_id: str) -> RuntimeSlot:
    return RuntimeSlot(
        project_id=project_id,
        status="initializing",
        tracer=TracerRuntime(
            resource_id="tracer-1",
            endpoint="http://tracer.test",
            buffer_config=TracerBufferConfig(
                trajectory_buffer_size=20,
                batch_size=2,
                time_per_round_seconds=60,
                training_save_path="/storage/training",
            ),
        ),
    )


def _simulation(
    *,
    store: AcquisitionSimulationStore,
    project_id: str,
) -> AcquisitionSimulation:
    return store.create(
        project_id=project_id,
        request_id="request-1",
        source=AcquisitionSource(
            train_uri="storage://datasets/avi/train.jsonl",
            val_uri="storage://datasets/avi/val.jsonl",
            image_root="storage://datasets/avi",
        ),
        requested_samples=4,
        requested_rounds=2,
        splits=["train"],
        auto_start_graph_run=True,
    )


def _sample_row(sample_id: str, image: str) -> dict[str, Any]:
    return {
        "id": sample_id,
        "system_prompt": "Inspect the PCB image.",
        "user_prompt": "Return result and reason.",
        "images": [image],
        "gt": {"result": "defect", "reason": "scratch"},
    }


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    ).encode()
