import json
from datetime import UTC, datetime
from io import BytesIO
from uuid import uuid4

import httpx
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from pyromind_agent_server.echomind_batch_schedule import PlatformBatchScheduleRuntime
from pyromind_agent_server.echomind_projects import (
    ProjectGraph,
    ProjectInfo,
    ProjectNode,
    ProjectNodeInput,
    ProjectNodeInputBinding,
    ProjectRecord,
)
from pyromind_agent_server.echomind_publication import (
    PromotionPolicy,
    PublishedGraph,
    RuntimeRequirements,
)
from pyromind_agent_server.echomind_runtime_scheduler import RuntimeGraphScheduler
from pyromind_agent_server.echomind_runtime_slots import (
    RuntimeHealth,
    RuntimeSlot,
    ServingInferenceRuntime,
    TracerBufferConfig,
    TracerRuntime,
)
from pyromind_runtime.domain.batch_schedule import BatchReceipt, BatchScheduleConfig
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.ports.batch_schedule import BatchScheduleBlocked
from pyromind_runtime.ports.storage import (
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


class MemoryStorage:
    def __init__(self):
        self.files: dict[str, bytes] = {}

    def read(self, path: str) -> bytes:
        try:
            return self.files[to_storage_api_path(path)]
        except KeyError:
            raise EvaluationStorageMissingError(path) from None

    def write(self, path: str, payload: bytes) -> None:
        self.files[to_storage_api_path(path)] = payload

    def list(self, prefix: str) -> set[str]:
        prefix = to_storage_api_path(prefix).rstrip("/") + "/"
        return {
            name.removeprefix(prefix) for name in self.files if name.startswith(prefix)
        }


@pytest.fixture
def platform_batch(tmp_path):
    project_id, release_id = str(uuid4()), str(uuid4())
    node = ProjectNode(
        id="dataset",
        type="dataset",
        title="Dataset",
        executor="dataset_runtime",
        config={"dataset_runtime": {}},
        inputs={
            "source_batch": ProjectNodeInput(
                artifact_type="dataset.source",
                schema_ref="test://source",
                binding=ProjectNodeInputBinding(
                    kind="run_input", name="batch_directory"
                ),
            )
        },
    )
    project = ProjectInfo(
        project=ProjectRecord(
            project_id=project_id,
            name="test",
            description="test",
            skill="test",
            conversation_id=uuid4().hex,
            cluster="cn-east-1",
            lifecycle="frozen",
            owner_user_id="owner",
            active_release_id=release_id,
        ),
        graph=ProjectGraph(
            graph_id=str(uuid4()),
            project_id=project_id,
            lifecycle="frozen",
            nodes=[node],
        ),
    )
    published = PublishedGraph.build(
        project_id=project_id,
        release_id=release_id,
        source_graph_id=project.graph.graph_id,
        source_graph_digest="sha256:" + "a" * 64,
        runtime_requirements=RuntimeRequirements(),
        promotion_policy=PromotionPolicy(metric="pass_rate"),
        nodes=[node],
        edges=[],
        published_by="owner",
        publication_mode="production",
    )
    slot = RuntimeSlot(
        project_id=project_id,
        status="ready",
        active_release_id=release_id,
        tracer=TracerRuntime(
            resource_id="tracer-1",
            endpoint="http://tracer.test",
            target_inference_resource_id="inf-1",
            admin_secret_ref="secret://admin",
            proxy_secret_ref="secret://proxy",
            health=RuntimeHealth(status="healthy"),
            buffer_config=TracerBufferConfig(
                trajectory_buffer_size=20,
                batch_size=5,
                time_per_round_seconds=60,
                training_save_path="/workspace/train",
            ),
        ),
        serving_inference=ServingInferenceRuntime(
            resource_id="inf-1",
            endpoint="http://inference.test",
            model_uri="/workspace/model",
            secret_ref="secret://inference",
            health=RuntimeHealth(status="healthy"),
        ),
    )
    storage = MemoryStorage()
    train = {
        "id": "train-1",
        "task_id": "task-1",
        "confirmed": True,
        "user_prompt": "inspect",
        "images": ["/workspace/images/train.jpg"],
        "gt": {"result": "fail"},
    }
    val = {
        **train,
        "id": "val-1",
        "task_id": "val-task-1",
        "images": ["/workspace/images/val.jpg"],
    }
    storage.write("/confirmed.jsonl", json.dumps(train).encode())
    storage.write("/holdout.jsonl", json.dumps(val).encode())
    buffer = BytesIO()
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "name": "llm.completion",
                    "rollout_id": "task-1",
                    "attributes": json.dumps({"gt": train["gt"]}),
                }
            ]
        ),
        buffer,
    )
    storage.write("/EchoMind/tracer-1/data/spans/0/spans_0.parquet", buffer.getvalue())
    calls = []
    inventory = [{"index": 0, "state": "ready"}]

    def handle(request):
        calls.append((request.method, request.url.path))
        assert request.headers["Authorization"] == "Bearer private-admin"
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "batch_mode": True,
                    "current_batch_index": 1,
                    "batches": inventory,
                },
            )
        if request.url.path.endswith("/claim"):
            inventory[0]["state"] = "in_training"
            return httpx.Response(200, json={"ok": True, "claimed": True})
        inventory.clear()
        return httpx.Response(200, json={"ok": True, "released_batch_index": 0})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        runtime = PlatformBatchScheduleRuntime(
            projects_root=tmp_path,
            project=project,
            published=published,
            slot=slot,
            scheduler=RuntimeGraphScheduler(projects_root=tmp_path),
            storage=storage,
            admin_key="private-admin",
            report_archived=lambda _: True,
            http_client=client,
        )
        config = BatchScheduleConfig(
            enabled=True,
            source_manifest_uri="/workspace/confirmed.jsonl",
            validation_dataset_uri="/workspace/holdout.jsonl",
        )
        receipt = BatchReceipt(
            batch_id="0",
            batch_index=0,
            tracer_resource_id="tracer-1",
            release_id=release_id,
            created_at=datetime.now(UTC),
        )
        yield runtime, config, receipt, storage, calls


def test_platform_batch_materializes_confirmed_gt_and_fixed_holdout(platform_batch):
    runtime, config, receipt, storage, calls = platform_batch
    runtime.preflight(config)
    assert runtime.list_ready()[0].batch_id == "batch-00000000"
    runtime.claim(receipt)
    inputs = runtime.materialize(receipt, config)
    directory = inputs["batch_directory"]
    assert isinstance(directory, str)
    training = json.loads(storage.read(directory + "/train.jsonl"))
    validation = json.loads(storage.read(directory + "/val.jsonl"))
    assert training["id"] == "train-1" and training["gt"] == {"result": "fail"}
    assert validation["id"] == "val-1"
    assert runtime.materialize(receipt, config) == inputs
    runtime.release(receipt)
    runtime.release(receipt)
    assert calls.count(("POST", "/admin/batches/0/release")) == 1


def test_archived_batch_release_verifies_before_deleting_and_on_retry(platform_batch):
    runtime, _, receipt, _, calls = platform_batch

    def verify_archive():
        calls.append(("verify", "archive"))

    assert runtime.release_archived(receipt, verify_archive)
    assert runtime.release_archived(receipt, verify_archive)
    assert calls == [
        ("GET", "/admin/batches"),
        ("verify", "archive"),
        ("POST", "/admin/batches/0/release"),
        ("GET", "/admin/batches"),
        ("verify", "archive"),
    ]


@pytest.mark.parametrize(("current", "state"), [(0, "ready"), (1, "collecting")])
def test_archived_batch_release_rejects_current_and_collecting_batches(
    platform_batch, monkeypatch, current, state
):
    runtime, _, receipt, _, calls = platform_batch

    def request(method, path):
        calls.append((method, path))
        return {
            "current_batch_index": current,
            "batches": [{"index": receipt.batch_index, "state": state}],
        }

    monkeypatch.setattr(runtime, "_request", request)

    def verify_archive():
        pytest.fail("An ineligible batch must not be archived for release")

    if current == receipt.batch_index:
        assert not runtime.release_archived(receipt, verify_archive)
    else:
        with pytest.raises(BatchScheduleBlocked, match="batch is sealed"):
            runtime.release_archived(receipt, verify_archive)
    assert calls == [("GET", "/admin/batches")]


def test_archived_batch_release_preserves_data_if_archive_verification_fails(
    platform_batch,
):
    runtime, _, receipt, _, calls = platform_batch

    def verify_archive():
        raise EvaluationStorageUnavailableError("Archive verification failed")

    with pytest.raises(EvaluationStorageUnavailableError, match="verification failed"):
        runtime.release_archived(receipt, verify_archive)
    assert calls == [("GET", "/admin/batches")]


def test_platform_missing_admin_is_blocked_without_requests(platform_batch):
    runtime, config, _receipt, _storage, calls = platform_batch
    runtime.admin_key = None
    with pytest.raises(BatchScheduleBlocked, match="admin authorization"):
        runtime.preflight(config)
    assert calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "train-1"),
        ("images", ["/workspace/images/train.jpg"]),
        ("confirmed", False),
    ],
)
def test_platform_blocks_holdout_leakage_and_unconfirmed_data(
    platform_batch, field, value
):
    runtime, config, _receipt, storage, _calls = platform_batch
    row = json.loads(storage.read("/holdout.jsonl"))
    row[field] = value
    storage.write("/holdout.jsonl", json.dumps(row).encode())
    with pytest.raises(BatchScheduleBlocked):
        runtime.preflight(config)


def test_platform_blocks_gt_mutation_after_materialization(platform_batch):
    runtime, config, receipt, storage, _calls = platform_batch
    runtime.materialize(receipt, config)
    row = json.loads(storage.read("/confirmed.jsonl"))
    row["gt"] = {"result": "pass"}
    storage.write("/confirmed.jsonl", json.dumps(row).encode())
    with pytest.raises(BatchScheduleBlocked, match="disagrees"):
        runtime.materialize(receipt, config)


def test_platform_recognizes_completed_manual_replay_batch(platform_batch):
    runtime, _config, receipt, _storage, calls = platform_batch
    run = runtime.runs.create(
        project_id=runtime.project.project.project_id,
        release_id=receipt.release_id,
        release_digest=runtime.published.content_sha256,
        request_id="manual-replay",
        trigger="manual",
        inputs={"tracer_batch": {"resource_id": "tracer-1", "batch_index": 0}},
    )
    now = datetime.now(UTC)
    runtime.runs.save(
        run.model_copy(
            update={
                "status": "succeeded",
                "outcome": "baseline_retained",
                "started_at": now,
                "finished_at": now,
            }
        )
    )
    observed = runtime.find_run(receipt)
    assert observed is not None
    assert observed.run_id == run.run_id
    assert observed.status == "succeeded" and observed.report_archived
    assert calls == []
