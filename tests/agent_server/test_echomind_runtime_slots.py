import logging
from datetime import UTC, datetime
from typing import cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from pyromind_sdk.client.base import PyroMindAPIError
from pyromind_sdk.client.client import PyroMindAPIClient
from pyromind_sdk.client.models import (
    EchoMindJobRequest,
    InferenceJobRequest,
    InferenceJobResponse,
)

from pyromind_agent_server.echomind_runtime_slots import (
    EchoMindTracerProvider,
    RuntimeHealth,
    RuntimeSlot,
    RuntimeSlotEnsureError,
    RuntimeSlotEnsureRequest,
    RuntimeSlotService,
    RuntimeSlotSetupRequest,
    RuntimeSlotStore,
    ServingInferenceBinding,
    ServingInferenceCreateConfig,
    ServingInferenceRuntime,
    TracerBufferConfig,
    TracerCreateConfig,
    TracerRuntime,
)


def test_tracer_endpoint_uses_cluster_base_url():
    provider = EchoMindTracerProvider(
        client=cast(PyroMindAPIClient, object()),
        cluster="us-west-1#pre",
    )

    assert (
        provider.endpoint("tracer-1")
        == "https://us-west-1.pyromind.ai/echomind/tracer-1"
    )


def test_serving_binding_uses_platform_job_details_and_job_secret():
    client = Mock(spec=PyroMindAPIClient)
    client.inference = Mock()
    client.inference.get.return_value = {
        "data": {
            "job_id": "inf-a",
            "name": "serving",
            "model_path": "/workspace/models/serving",
            "status": "Running",
            "endpoint_url": "https://inference.example.test/inference/inf-a/v1/",
            "inf_sk": "inference-only-key",
        }
    }
    provider = EchoMindTracerProvider(client=client, cluster="us-west-1")

    binding, key = provider.serving_binding("inf-a")

    client.inference.get.assert_called_once_with("/inference/inf-a")
    assert binding.endpoint == "https://inference.example.test/inference/inf-a/v1"
    assert binding.model_uri == "/workspace/models/serving"
    assert key == "inference-only-key"
    assert key not in binding.model_dump_json()
    with pytest.raises(RuntimeSlotEnsureError, match="Invalid"):
        provider.serving_binding("../other")


def test_tracer_admin_key_lookup_does_not_log_secrets(caplog):
    client = Mock(spec=PyroMindAPIClient)
    client.echomind = Mock()
    sdk_logger = logging.getLogger("pyromind_sdk")

    def get(path):
        sdk_logger.error("SDK body: %s", {"admin_key": "tracer-admin-key"})
        return {"data": {"admin_key": "tracer-admin-key"}}

    client.echomind.get.side_effect = get
    provider = EchoMindTracerProvider(client=client, cluster="us-west-1")

    assert provider.batch_admin_key("tracer-1") == "tracer-admin-key"
    client.echomind.get.assert_called_once_with("/echomind/tracer-1/batch-admin")
    sdk_logger.error("ordinary SDK activity remains logged")
    assert "ordinary SDK activity" in caplog.text
    assert "tracer-admin-key" not in caplog.text


@pytest.mark.parametrize(
    ("cluster", "gpu_card", "gpu_count", "platform_card", "cpu", "memory"),
    [
        ("us-west-1", "L40S", 1, "L40S", "4", "32Gi"),
        ("us-west-1", "NVIDIA-L40S", 1, "L40S", "4", "32Gi"),
        ("cn-east-1", "NVIDIA-H20", 1, "H20", "11", "88Gi"),
        ("cn-east-1", "H20", 2, "H20", "22", "176Gi"),
        ("cn-east-1", "NVIDIA-A10", 1, "A10", "6", "48Gi"),
        ("us-west-2", "NVIDIA-H200", 1, "H200", "11", "88Gi"),
    ],
)
def test_inference_submission_uses_catalog_and_approved_gpu_settings(
    cluster, gpu_card, gpu_count, platform_card, cpu, memory
):
    client = Mock(spec=PyroMindAPIClient)
    client.inference = Mock()
    client.inference.get_framework.return_value = ["sglang"]
    client.inference.get_inf_image.return_value = ["catalog/sglang:1"]
    provider = EchoMindTracerProvider(client=client, cluster=cluster)
    config = _setup_request().serving_inference_create
    assert config is not None
    config = config.model_copy(update={"gpu_card": gpu_card, "gpu_count": gpu_count})
    request = provider.prepare_serving_inference(name="project-serving", config=config)
    assert request.name == "project-serving"
    assert request.model_path == config.model_path
    assert request.model_name == config.model_name
    assert request.inf_image == "catalog/sglang:1"
    assert request.resources is not None
    assert request.resources.gpu == str(gpu_count)
    assert request.resources.gpu_card == platform_card
    assert request.resources.cpu == cpu
    assert request.resources.memory == memory


def test_inference_lookup_asks_platform_for_the_project_job_by_name():
    client = Mock(spec=PyroMindAPIClient)
    client.inference = Mock()
    client.inference.get.return_value = {
        "data": {
            "inference_jobs": [
                {
                    "job_id": "inference-created",
                    "name": "project-serving",
                    "model_path": "/workspace/models/serving",
                    "status": "Running",
                }
            ]
        }
    }
    provider = EchoMindTracerProvider(client=client, cluster="us-west-1")

    job = provider.find_serving_inference("project-serving")

    client.inference.get.assert_called_once_with(
        "/inference", params={"name": "project-serving"}
    )
    assert job is not None
    assert job.id == "inference-created"


async def test_runtime_slot_setup_resumes_inference_before_creating_tracer(tmp_path):
    project_id, release_id = str(uuid4()), str(uuid4())
    provider = _FakeTracerProvider()
    provider.inference_healthy = False
    service = RuntimeSlotService(projects_root=tmp_path, provider=provider)
    request = _setup_request()

    async def setup(config=request):
        return await service.setup(
            project_id=project_id,
            release_id=release_id,
            request=config,
            timeout_seconds=0,
            poll_seconds=0.01,
        )

    with pytest.raises(RuntimeSlotEnsureError, match="still starting"):
        await setup()
    partial = service.store.load(project_id)
    assert partial.status == "degraded"
    assert partial.serving_inference is not None
    assert partial.serving_inference.resource_id == "inference-created"
    assert len(provider.inference_requests) == 1
    assert provider.created == 0

    provider.inference_healthy = True
    ready = await setup()
    assert ready.status == "ready"
    assert ready.tracer is not None
    assert ready.tracer.target_inference_resource_id == "inference-created"
    assert provider.requests[0].access_key == "serving-access-key"
    assert "serving-access-key" not in ready.model_dump_json()
    assert await setup() == ready
    assert len(provider.inference_requests) == provider.created == 1

    assert request.serving_inference_create is not None
    different = request.model_copy(
        update={
            "serving_inference_create": request.serving_inference_create.model_copy(
                update={"gpu_card": "H100"}
            )
        }
    )
    with pytest.raises(RuntimeSlotEnsureError, match="different Inference"):
        await setup(different)
    assert service.store.load(project_id) == ready
    assert len(provider.inference_requests) == provider.created == 1


async def test_runtime_slot_allows_retry_after_platform_rejects_gpu_request(tmp_path):
    project_id, release_id = str(uuid4()), str(uuid4())
    provider = _FakeTracerProvider()
    provider.inference_submission_error = PyroMindAPIError(
        "There are not enough GPU Card to start the instance",
        status_code=500,
        response={
            "error": {
                "code": "INSUFFICIENT_RESOURCES",
                "message": "There are not enough GPU Card to start the instance",
            },
        },
    )
    service = RuntimeSlotService(projects_root=tmp_path, provider=provider)
    request = _setup_request()

    with pytest.raises(RuntimeSlotEnsureError, match="INSUFFICIENT_RESOURCES|GPU"):
        await service.setup(
            project_id=project_id, release_id=release_id, request=request
        )

    failed = service.store.load(project_id)
    assert failed.status == "failed"
    assert failed.error is not None
    assert "GPU" in failed.error or "INSUFFICIENT_RESOURCES" in failed.error
    assert failed.serving_inference is None
    assert not failed.serving_inference_submission_pending
    assert len(provider.inference_requests) == 1
    assert provider.created == 0

    provider.inference_submission_error = None
    ready = await service.setup(
        project_id=project_id, release_id=release_id, request=request
    )
    assert ready.status == "ready"
    assert len(provider.inference_requests) == 2
    assert provider.created == 1


@pytest.mark.parametrize(
    ("message", "expected_error"),
    [
        (
            "Service echomind is not activated. Please activate it first.",
            "EchoMind 服务未开通",
        ),
        ("custom_runtime_script_path is required", "仍要求独立执行脚本"),
    ],
)
async def test_runtime_slot_can_retry_tracer_after_platform_rejection(
    tmp_path, message, expected_error
):
    project_id, release_id = str(uuid4()), str(uuid4())
    provider = _FakeTracerProvider()
    provider.tracer_submission_error = PyroMindAPIError(
        message,
        status_code=500,
        response={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": message,
            },
        },
    )
    service = RuntimeSlotService(projects_root=tmp_path, provider=provider)
    request = _setup_request()

    with pytest.raises(RuntimeSlotEnsureError, match=expected_error):
        await service.setup(
            project_id=project_id, release_id=release_id, request=request
        )

    failed = service.store.load(project_id)
    assert failed.status == "failed"
    assert failed.error is not None and expected_error in failed.error
    assert not failed.tracer_submission_pending
    assert failed.tracer is None
    assert failed.serving_inference is not None
    assert failed.serving_inference.resource_id == "inference-created"
    assert failed.serving_inference.health.status == "healthy"
    assert len(provider.inference_requests) == provider.created == 1

    provider.tracer_submission_error = None
    ready = await service.setup(
        project_id=project_id, release_id=release_id, request=request
    )
    assert ready.status == "ready"
    assert ready.tracer is not None
    assert ready.tracer.target_inference_resource_id == "inference-created"
    assert len(provider.inference_requests) == 1
    assert provider.created == 2


async def test_runtime_slot_retries_legacy_gpu_alias_submission_only_once(tmp_path):
    project_id, release_id = str(uuid4()), str(uuid4())
    provider = _FakeTracerProvider()
    provider.inference_submission_error = PyroMindAPIError("timeout")
    service = RuntimeSlotService(projects_root=tmp_path, provider=provider)
    request = _setup_request()
    assert request.serving_inference_create is not None
    legacy_config = request.serving_inference_create.model_copy(
        update={"gpu_card": "NVIDIA-L40S"}
    )
    corrected = request.model_copy(
        update={
            "serving_inference_create": request.serving_inference_create.model_copy(
                update={"model_path": "/workspace/models/merged-model"}
            )
        }
    )
    service.store.save(
        RuntimeSlot(
            project_id=project_id,
            active_release_id=release_id,
            status="failed",
            error="Inference setup failed: PyroMindAPIError",
            serving_inference_create=legacy_config,
            serving_inference_submission_pending=True,
        )
    )

    with pytest.raises(PyroMindAPIError):
        await service.setup(
            project_id=project_id, release_id=release_id, request=corrected
        )

    with pytest.raises(RuntimeSlotEnsureError, match="submission is uncertain"):
        await service.setup(
            project_id=project_id, release_id=release_id, request=corrected
        )

    assert len(provider.inference_requests) == 1
    assert provider.inference_requests[0].model_path == "/workspace/models/merged-model"
    assert provider.created == 0


@pytest.mark.parametrize(
    ("failed_resource", "status_code"),
    [("inference", None), ("inference", 500), ("tracer", None), ("tracer", 500)],
)
async def test_runtime_slot_does_not_repeat_uncertain_submission(
    tmp_path, failed_resource, status_code
):
    project_id, release_id = str(uuid4()), str(uuid4())
    provider = _FakeTracerProvider()
    message = "unconfirmed submission"
    response = (
        {"error": {"code": "INTERNAL_SERVER_ERROR", "message": message}}
        if status_code == 500
        else None
    )
    error = PyroMindAPIError(message, status_code=status_code, response=response)
    if failed_resource == "inference":
        provider.inference_submission_error = error
    else:
        provider.tracer_submission_error = error
    service = RuntimeSlotService(projects_root=tmp_path, provider=provider)
    request = _setup_request()

    async def setup():
        return await service.setup(
            project_id=project_id, release_id=release_id, request=request
        )

    with pytest.raises(PyroMindAPIError):
        await setup()
    slot = service.store.load(project_id)
    if failed_resource == "inference":
        assert slot.serving_inference_submission_pending
        assert slot.serving_inference is None
    else:
        assert slot.tracer_submission_pending
        assert slot.tracer is None
    with pytest.raises(RuntimeSlotEnsureError, match="submission is uncertain"):
        await setup()
    assert len(provider.inference_requests) == 1
    assert provider.created == int(failed_resource == "tracer")

    if failed_resource == "inference":
        submitted = provider.inference_requests[0]
        provider.inference_found = InferenceJobResponse(
            job_id="inference-created",
            name=submitted.name or "",
            model_path=submitted.model_path,
            status="running",
        )
    else:
        provider.existing_resource_id = "tracer-created"
    assert (await setup()).status == "ready"
    assert len(provider.inference_requests) == provider.created == 1


def test_ready_runtime_slot_round_trip_persists_secret_references(tmp_path):
    project_id = str(uuid4())
    health = RuntimeHealth(status="healthy", checked_at=datetime.now(UTC))
    slot = RuntimeSlot(
        project_id=project_id,
        status="ready",
        tracer=TracerRuntime(
            resource_id="tracer-1",
            endpoint="https://tracer.example.test",
            admin_secret_ref="secret://echomind/tracer/admin",
            proxy_secret_ref="secret://echomind/tracer/proxy",
            target_inference_resource_id="inference-a",
            buffer_config=TracerBufferConfig(
                trajectory_buffer_size=100,
                batch_size=10,
                time_per_round_seconds=60,
                training_save_path="/target-workspace/models",
            ),
            health=health,
        ),
        serving_inference=ServingInferenceRuntime(
            resource_id="inference-a",
            endpoint="https://inference.example.test",
            secret_ref="secret://echomind/inference/a",
            model_uri="storage://models/baseline",
            health=health,
        ),
    )

    store = RuntimeSlotStore(tmp_path / "projects")
    store.save(slot)

    assert store.load(project_id) == slot
    persisted = (tmp_path / "projects" / project_id / "runtime_slot.json").read_text(
        encoding="utf-8"
    )
    assert "secret://echomind/tracer/admin" in persisted
    assert "tracer-proxy-secret" not in persisted


def test_runtime_slot_rejects_plaintext_secret_and_incomplete_ready_state():
    with pytest.raises(ValidationError):
        TracerRuntime(
            resource_id="tracer-1",
            admin_secret_ref="plaintext-token",
            buffer_config=TracerBufferConfig(
                trajectory_buffer_size=100,
                batch_size=10,
                time_per_round_seconds=60,
                training_save_path="/target-workspace/models",
            ),
        )

    with pytest.raises(ValidationError):
        RuntimeSlot(project_id=str(uuid4()), status="ready")


async def test_runtime_slot_ensure_is_idempotent(tmp_path):
    project_id = str(uuid4())
    release_id = str(uuid4())
    provider = _FakeTracerProvider()
    service = RuntimeSlotService(
        projects_root=tmp_path / "projects",
        provider=provider,
    )
    request = _ensure_request()

    first = await service.ensure(
        project_id=project_id,
        release_id=release_id,
        request=request,
        access_key="serving-access-key",
        timeout_seconds=1,
        poll_seconds=0.01,
    )
    second = await service.ensure(
        project_id=project_id,
        release_id=release_id,
        request=request,
        access_key="serving-access-key",
        timeout_seconds=1,
        poll_seconds=0.01,
    )

    assert first.status == "ready"
    assert second == first
    assert provider.created == 1
    persisted = (tmp_path / "projects" / project_id / "runtime_slot.json").read_text(
        encoding="utf-8"
    )
    assert "serving-access-key" not in persisted
    assert "tracer-proxy-secret" not in persisted


async def test_runtime_slot_attaches_existing_tracer_and_reports_degraded(
    tmp_path,
):
    provider = _FakeTracerProvider(
        existing_resource_id="tracer-existing",
        health=RuntimeHealth(status="unhealthy", message="offline"),
    )
    service = RuntimeSlotService(
        projects_root=tmp_path / "projects",
        provider=provider,
    )

    slot = await service.ensure(
        project_id=str(uuid4()),
        release_id=str(uuid4()),
        request=_ensure_request(),
        access_key="serving-access-key",
        timeout_seconds=1,
        poll_seconds=0.01,
    )

    assert slot.status == "degraded"
    assert slot.tracer is not None
    assert slot.tracer.resource_id == "tracer-existing"
    assert provider.created == 0


async def test_runtime_slot_rejects_a_tracer_that_points_to_another_model(tmp_path):
    provider = _FakeTracerProvider(existing_resource_id="tracer-existing")
    provider.matches = False
    service = RuntimeSlotService(projects_root=tmp_path, provider=provider)
    project_id = str(uuid4())
    request = _ensure_request()
    setup = RuntimeSlotSetupRequest(
        serving_inference_id="inference-a", tracer=request.tracer
    )
    assert setup.resource_id() == request.serving_inference.resource_id

    with pytest.raises(RuntimeSlotEnsureError, match="routes to another Inference"):
        await service.ensure(
            project_id=project_id,
            release_id=str(uuid4()),
            request=request,
            access_key="serving-access-key",
        )
    assert service.store.load(project_id).status == "failed"
    assert provider.created == 0


def _ensure_request() -> RuntimeSlotEnsureRequest:
    return RuntimeSlotEnsureRequest(
        serving_inference=ServingInferenceBinding(
            resource_id="inference-a",
            endpoint="https://serving.example.test",
            model_uri="storage://models/baseline",
        ),
        tracer=TracerCreateConfig(
            origin_model="qwen",
            training_model="qwen-avi",
            training_batch_size=10,
            trajectory_buffer_size=100,
            time_per_round_seconds=60,
            training_rounds=1,
            training_save_path="storage://models/tracer",
        ),
    )


def _setup_request() -> RuntimeSlotSetupRequest:
    return RuntimeSlotSetupRequest(
        serving_inference_create=ServingInferenceCreateConfig(
            model_path="/workspace/models/serving",
            model_name="qwen",
            gpu_card="L40S",
        ),
        tracer=_ensure_request().tracer,
    )


class _FakeTracerProvider:
    def __init__(
        self,
        *,
        existing_resource_id: str | None = None,
        health: RuntimeHealth | None = None,
    ) -> None:
        self.existing_resource_id = existing_resource_id
        self.health = health or RuntimeHealth(status="healthy")
        self.created = 0
        self.requests: list[EchoMindJobRequest] = []
        self.matches = True
        self.inference_healthy = True
        self.inference_found: InferenceJobResponse | None = None
        self.inference_requests: list[InferenceJobRequest] = []
        self.inference_submission_error: PyroMindAPIError | None = None
        self.tracer_submission_error: PyroMindAPIError | None = None

    def find_serving_inference(self, name: str) -> InferenceJobResponse | None:
        if self.inference_found and self.inference_found.name == name:
            return self.inference_found
        return None

    def prepare_serving_inference(
        self, *, name: str, config: ServingInferenceCreateConfig
    ) -> InferenceJobRequest:
        return InferenceJobRequest(name=name, model_path=config.model_path)

    def create_serving_inference(self, request: InferenceJobRequest) -> str:
        self.inference_requests.append(request)
        if self.inference_submission_error is not None:
            raise self.inference_submission_error
        return "inference-created"

    def batch_admin_key(self, resource_id: str) -> str | None:
        return None

    def serving_binding(self, resource_id: str) -> tuple[ServingInferenceBinding, str]:
        return _ensure_request().serving_inference.model_copy(
            update={
                "resource_id": resource_id,
                "model_uri": "/workspace/models/serving",
            }
        ), "serving-access-key"

    def tracer_targets(self, resource_id: str, endpoint: str) -> bool:
        del resource_id, endpoint
        return self.matches

    def tracer_script_path(self, resource_id: str) -> str | None:
        return None

    def verify_batch_admin(self, *, resource_id: str, admin_key: str) -> bool:
        return bool(resource_id and admin_key)

    def find_existing(self, name: str) -> str | None:
        del name
        return self.existing_resource_id

    def create(self, request: EchoMindJobRequest) -> str:
        self.created += 1
        self.requests.append(request)
        if self.tracer_submission_error is not None:
            raise self.tracer_submission_error
        return "tracer-created"

    def status(self, resource_id: str) -> str:
        del resource_id
        return "running"

    def secret_key(self, resource_id: str) -> str | None:
        del resource_id
        return "tracer-proxy-secret"

    def endpoint(self, resource_id: str) -> str:
        return f"https://{resource_id}.echomind.example.test/v1"

    def probe(self, *, endpoint: str, access_key: str) -> RuntimeHealth:
        del endpoint, access_key
        return self.health

    def probe_inference(self, *, endpoint: str, access_key: str) -> RuntimeHealth:
        if not self.inference_healthy:
            return RuntimeHealth(status="unhealthy")
        return self.probe(endpoint=endpoint, access_key=access_key)
