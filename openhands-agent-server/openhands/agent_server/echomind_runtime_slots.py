from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Protocol
from urllib.parse import urlsplit

import httpx
from filelock import AsyncFileLock, Timeout
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)
from pyromind_runtime.domain.storage_paths import normalize_workspace_path
from pyromind_sdk.client.base import PyroMindAPIError
from pyromind_sdk.client.client import PyroMindAPIClient
from pyromind_sdk.client.models import (
    EchoMindJobRequest,
    InferenceJobRequest,
    InferenceJobResponse,
    ResourceConfig,
)

from openhands.agent_server.echomind_storage import (
    canonical_uuid,
    project_directory,
    read_model,
    write_model,
)
from openhands.agent_server.echomind_tracer_batch_admin import (
    is_tracer_batch_admin_response,
)


HealthStatus = Literal["unknown", "healthy", "unhealthy"]
RuntimeSlotStatus = Literal[
    "uninitialized",
    "initializing",
    "ready",
    "degraded",
    "failed",
]
SecretReference = Annotated[
    str,
    StringConstraints(pattern=r"^(env|secret|vault|k8s-secret)://[^\s]+$"),
]
Endpoint = Annotated[str, StringConstraints(pattern=r"^https?://[^\s]+$")]
_RESOURCE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,128}$"
RuntimeResourceId = Annotated[str, StringConstraints(pattern=_RESOURCE_ID_PATTERN)]
_runtime_sdk_request = ContextVar("runtime_sdk_request", default=False)
_GPU_PRODUCT_LABELS = {
    "NVIDIA-A10": "A10",
    "NVIDIA-H20": "H20",
    "NVIDIA-H100-NVL": "H100",
    "NVIDIA-H100-80GB-HBM3": "H100-SXM",
    "NVIDIA-L40S": "L40S",
    "NVIDIA-H200": "H200",
    "NVIDIA-B200": "B200",
}
_GPU_MIN_CPU_CORES = {"H100-SXM": 11, "H100": 11, "H200": 11, "L40S": 11, "B200": 11}
_INFERENCE_MIN_CPU_CORES_PER_GPU = {
    "us-west-2": _GPU_MIN_CPU_CORES,
    "cn-east-1": {**_GPU_MIN_CPU_CORES, "A10": 6, "H20": 11},
}


@contextmanager
def _private_sdk_request() -> Iterator[None]:
    # The SDK's log masking does not cover its Inference and Tracer key names.
    token = _runtime_sdk_request.set(True)
    try:
        yield
    finally:
        _runtime_sdk_request.reset(token)


def _filter_runtime_sdk_body(_record: logging.LogRecord) -> bool:
    return not _runtime_sdk_request.get()


logging.getLogger("pyromind_sdk").addFilter(_filter_runtime_sdk_body)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RuntimeHealth(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: HealthStatus = "unknown"
    checked_at: datetime | None = None
    message: str | None = Field(default=None, max_length=2000)


class TracerResources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu: str = "4"
    memory: str = "8Gi"
    gpu: str | None = None
    gpu_card: str | None = None


class TracerCreateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_mode: Literal["openai", "gemini", "anthropic"] = "openai"
    origin_model: str = Field(min_length=1, max_length=256)
    training_model: str = Field(min_length=1, max_length=256)
    training_batch_size: PositiveInt
    trajectory_buffer_size: PositiveInt
    time_per_round_seconds: float = Field(gt=0)
    training_rounds: PositiveInt
    training_save_path: str = Field(min_length=1, max_length=2000)
    execution_mode: Literal["manual", "automatic"] = "manual"
    custom_runtime_script_path: str | None = Field(default=None, max_length=2000)
    resources: TracerResources = Field(default_factory=TracerResources)

    @model_validator(mode="after")
    def validate_buffer_capacity(self) -> TracerCreateConfig:
        if self.trajectory_buffer_size <= self.training_batch_size:
            raise ValueError("trajectory_buffer_size must be greater than batch size")
        return self


class ServingInferenceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(default="echomind", min_length=1, max_length=128)
    resource_id: RuntimeResourceId
    endpoint: Endpoint
    model_uri: str = Field(min_length=1, max_length=2000)
    secret_ref: SecretReference = "secret://echomind/pyromind-access-key"


class ServingInferenceCreateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    model_path: str = Field(min_length=1, max_length=2000)
    model_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    gpu_card: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    gpu_count: int = Field(default=1, ge=1, le=8)
    max_model_len: int = Field(default=8192, ge=512, le=131072)
    framework: str = Field(default="sglang", pattern=r"^[A-Za-z0-9_-]{1,32}$")
    image: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: str) -> str:
        return normalize_workspace_path(value, field="model_path")


class RuntimeSlotEnsureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    serving_inference: ServingInferenceBinding
    tracer: TracerCreateConfig
    attach_existing_tracer_id: str | None = Field(default=None, max_length=128)


class RuntimeSlotSetupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    serving_inference_id: RuntimeResourceId | None = None
    serving_inference: ServingInferenceBinding | None = None
    serving_inference_create: ServingInferenceCreateConfig | None = None
    tracer: TracerCreateConfig
    attach_existing_tracer_id: RuntimeResourceId | None = None

    @model_validator(mode="after")
    def validate_serving_selection(self) -> RuntimeSlotSetupRequest:
        if (
            sum(
                item is not None
                for item in (
                    self.serving_inference_id,
                    self.serving_inference,
                    self.serving_inference_create,
                )
            )
            != 1
        ):
            raise ValueError("Select an existing Inference job or configure a new one.")
        if (
            self.serving_inference_create is not None
            and self.serving_inference_create.model_name != self.tracer.origin_model
        ):
            raise ValueError("Tracer origin_model must match the Inference model name.")
        return self

    def resource_id(self) -> str:
        if self.serving_inference_id is not None:
            return self.serving_inference_id
        assert self.serving_inference is not None
        return self.serving_inference.resource_id


class TracerBufferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_buffer_size: PositiveInt
    batch_size: PositiveInt
    time_per_round_seconds: PositiveInt
    training_save_path: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_buffer_capacity(self) -> TracerBufferConfig:
        if self.trajectory_buffer_size <= self.batch_size:
            raise ValueError("trajectory_buffer_size must be greater than batch_size")
        return self


class TracerRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(default="echomind", min_length=1, max_length=128)
    resource_id: str | None = Field(default=None, max_length=128)
    endpoint: Endpoint | None = None
    admin_secret_ref: SecretReference | None = None
    proxy_secret_ref: SecretReference | None = None
    target_inference_resource_id: str | None = Field(default=None, max_length=128)
    custom_runtime_script_path: str | None = Field(default=None, max_length=2000)
    buffer_config: TracerBufferConfig
    health: RuntimeHealth = Field(default_factory=RuntimeHealth)


class ServingInferenceRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(default="echomind", min_length=1, max_length=128)
    resource_id: str | None = Field(default=None, max_length=128)
    endpoint: Endpoint | None = None
    secret_ref: SecretReference | None = None
    model_uri: str | None = Field(default=None, max_length=2000)
    health: RuntimeHealth = Field(default_factory=RuntimeHealth)


class BaselineSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_uri: str = Field(min_length=1, max_length=2000)
    dataset_uri: str = Field(min_length=1, max_length=2000)
    metrics: dict[str, float] = Field(default_factory=dict)
    evaluated_at: datetime


class RuntimeSlot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    project_id: str
    status: RuntimeSlotStatus = "uninitialized"
    tracer: TracerRuntime | None = None
    serving_inference: ServingInferenceRuntime | None = None
    serving_inference_create: ServingInferenceCreateConfig | None = None
    serving_inference_submission_pending: bool = False
    tracer_submission_pending: bool = False
    baseline: BaselineSnapshot | None = None
    batch_execution_available: bool = False
    active_release_id: str | None = Field(default=None, max_length=128)
    active_graph_run_id: str | None = Field(default=None, max_length=128)
    error: str | None = Field(default=None, max_length=4000)
    updated_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def validate_ready_resources(self) -> RuntimeSlot:
        if self.status != "ready":
            return self
        if self.tracer is None or not all(
            (
                self.tracer.resource_id,
                self.tracer.endpoint,
                self.tracer.admin_secret_ref,
                self.tracer.proxy_secret_ref,
                self.tracer.target_inference_resource_id,
            )
        ):
            raise ValueError("ready runtime slot requires a complete Tracer")
        if self.tracer.health.status != "healthy":
            raise ValueError("ready runtime slot requires a healthy Tracer")
        if self.serving_inference is None or not all(
            (
                self.serving_inference.resource_id,
                self.serving_inference.endpoint,
                self.serving_inference.secret_ref,
                self.serving_inference.model_uri,
            )
        ):
            raise ValueError("ready runtime slot requires a complete Serving Inference")
        if self.serving_inference.health.status != "healthy":
            raise ValueError("ready runtime slot requires a healthy Serving Inference")
        return self


class RuntimeSlotStore:
    def __init__(self, projects_root: Path | str) -> None:
        self.projects_root = Path(projects_root)

    def load(self, project_id: str) -> RuntimeSlot:
        slot = self.load_optional(project_id)
        if slot is None:
            raise FileNotFoundError(f"runtime slot not found: {project_id}")
        return slot

    def load_optional(self, project_id: str) -> RuntimeSlot | None:
        path = self._path(project_id)
        if not path.is_file():
            return None
        slot = read_model(path, RuntimeSlot)
        if canonical_uuid(slot.project_id, kind="project") != canonical_uuid(
            project_id,
            kind="project",
        ):
            raise ValueError("runtime slot belongs to another project")
        return slot

    def save(self, slot: RuntimeSlot) -> RuntimeSlot:
        write_model(self._path(slot.project_id), slot)
        return slot

    def _path(self, project_id: str) -> Path:
        return project_directory(self.projects_root, project_id) / "runtime_slot.json"

    def lease(self, project_id: str) -> AsyncFileLock:
        return AsyncFileLock(
            str(self._path(project_id).with_suffix(".lock")), timeout=0
        )


class RuntimeSlotProvider(Protocol):
    def serving_binding(
        self, resource_id: str
    ) -> tuple[ServingInferenceBinding, str]: ...

    def tracer_targets(self, resource_id: str, endpoint: str) -> bool: ...

    def tracer_script_path(self, resource_id: str) -> str | None: ...

    def find_serving_inference(self, name: str) -> InferenceJobResponse | None: ...

    def prepare_serving_inference(
        self, *, name: str, config: ServingInferenceCreateConfig
    ) -> InferenceJobRequest: ...

    def create_serving_inference(self, request: InferenceJobRequest) -> str: ...

    def batch_admin_key(self, resource_id: str) -> str | None: ...

    def verify_batch_admin(self, *, resource_id: str, admin_key: str) -> bool: ...

    def find_existing(self, name: str) -> str | None: ...

    def create(self, request: EchoMindJobRequest) -> str: ...

    def status(self, resource_id: str) -> str: ...

    def secret_key(self, resource_id: str) -> str | None: ...

    def endpoint(self, resource_id: str) -> str: ...

    def probe(self, *, endpoint: str, access_key: str) -> RuntimeHealth: ...

    def probe_inference(self, *, endpoint: str, access_key: str) -> RuntimeHealth: ...


class RuntimeSlotEnsureError(RuntimeError):
    pass


class _ServingInferencePending(RuntimeSlotEnsureError):
    pass


class EchoMindTracerProvider:
    def __init__(
        self,
        *,
        client: PyroMindAPIClient,
        cluster: str,
    ) -> None:
        self.client = client
        self.cluster = cluster

    @_private_sdk_request()
    def find_serving_inference(self, name: str) -> InferenceJobResponse | None:
        response = self.client.inference.get("/inference", params={"name": name})
        data = response.get("data", response)
        jobs = data.get("inference_jobs") if isinstance(data, dict) else data
        if not isinstance(jobs, list):
            raise RuntimeSlotEnsureError("The platform Inference list is unavailable.")
        matches = [
            InferenceJobResponse.model_validate(job)
            for job in jobs
            if isinstance(job, dict) and job.get("name") == name
        ]
        if len(matches) > 1:
            raise RuntimeSlotEnsureError("Multiple Inference jobs match this Project.")
        return matches[0] if matches else None

    def prepare_serving_inference(
        self, *, name: str, config: ServingInferenceCreateConfig
    ) -> InferenceJobRequest:
        if config.framework not in self.client.inference.get_framework():
            raise RuntimeSlotEnsureError(
                "Inference framework is not in the cluster catalog."
            )
        images = self.client.inference.get_inf_image(config.framework)
        if not images or (config.image is not None and config.image not in images):
            raise RuntimeSlotEnsureError(
                "Inference image is not in the cluster catalog."
            )
        gpu_card = _gpu_product_code(config.gpu_card)
        minimum_cpu = _INFERENCE_MIN_CPU_CORES_PER_GPU.get(self.cluster, {}).get(
            gpu_card, 0
        )
        cpu = max(4, minimum_cpu * config.gpu_count)
        return InferenceJobRequest(
            name=name,
            model_path=config.model_path,
            model_name=config.model_name,
            inference_framework=config.framework,
            inf_image=config.image or images[0],
            model_length=config.max_model_len,
            resources=ResourceConfig(
                cpu=str(cpu),
                memory=f"{8 * cpu}Gi",
                gpu=str(config.gpu_count),
                gpu_card=gpu_card,
            ),
        )

    @_private_sdk_request()
    def create_serving_inference(self, request: InferenceJobRequest) -> str:
        return self.client.inference.create(request)

    @_private_sdk_request()
    def serving_binding(self, resource_id: str) -> tuple[ServingInferenceBinding, str]:
        if not re.fullmatch(_RESOURCE_ID_PATTERN, resource_id):
            raise RuntimeSlotEnsureError("Invalid Serving Inference job ID.")
        response = self.client.inference.get(f"/inference/{resource_id}")
        data = response.get("data", response)
        if not isinstance(data, dict):
            raise RuntimeSlotEnsureError("Serving Inference detail is unavailable.")
        job = InferenceJobResponse.model_validate(data)
        if job.id != resource_id:
            raise RuntimeSlotEnsureError("Serving Inference job ID did not match.")
        status = job.status.lower()
        if status in {
            "failed",
            "error",
            "terminated",
            "stopped",
            "deleted",
            "canceled",
        }:
            raise RuntimeSlotEnsureError("Serving Inference entered a terminal state.")
        if status not in {"running", "ready"}:
            raise _ServingInferencePending("Serving Inference job is still starting.")
        endpoint = (job.endpoint_url or "").rstrip("/")
        if not endpoint:
            raise _ServingInferencePending(
                "Serving Inference endpoint is still starting."
            )
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeSlotEnsureError("Serving Inference has no usable endpoint.")
        access_key = data.get("inf_sk")
        if not isinstance(access_key, str) or not access_key.strip():
            raise _ServingInferencePending(
                "Serving Inference access key is unavailable."
            )
        return ServingInferenceBinding(
            provider="pyromind.inference",
            resource_id=job.id,
            endpoint=endpoint,
            model_uri=job.model_path,
            secret_ref=f"secret://inference/{job.id}/access",
        ), access_key.strip()

    @_private_sdk_request()
    def tracer_targets(self, resource_id: str, endpoint: str) -> bool:
        job = self.client.echomind.get_job(resource_id)
        return (job.api_url or "").rstrip("/").removesuffix("/v1") == (
            endpoint.rstrip("/").removesuffix("/v1")
        )

    @_private_sdk_request()
    def tracer_script_path(self, resource_id: str) -> str | None:
        return self.client.echomind.get_job(resource_id).custom_runtime_script_path

    @_private_sdk_request()
    def batch_admin_key(self, resource_id: str) -> str | None:
        if not re.fullmatch(_RESOURCE_ID_PATTERN, resource_id):
            raise RuntimeSlotEnsureError("Invalid Tracer job ID.")
        try:
            response = self.client.echomind.get(f"/echomind/{resource_id}/batch-admin")
        except PyroMindAPIError as exc:
            if exc.status_code in {404, 501}:
                return None
            raise
        data = response.get("data", response)
        key = data.get("admin_key") if isinstance(data, dict) else None
        return key.strip() if isinstance(key, str) and key.strip() else None

    def verify_batch_admin(self, *, resource_id: str, admin_key: str) -> bool:
        if not re.fullmatch(_RESOURCE_ID_PATTERN, resource_id):
            return False
        try:
            response = httpx.get(
                f"{self.endpoint(resource_id).rstrip('/')}/admin/batches",
                headers={"Authorization": f"Bearer {admin_key}"},
                timeout=5,
            )
            return response.status_code == 200 and is_tracer_batch_admin_response(
                response.json()
            )
        except (httpx.HTTPError, ValueError):
            return False

    @_private_sdk_request()
    def find_existing(self, name: str) -> str | None:
        for job in self.client.echomind.list():
            if job.name == name:
                return job.job_id
        return None

    @_private_sdk_request()
    def create(self, request: EchoMindJobRequest) -> str:
        return self.client.echomind.create(request)

    @_private_sdk_request()
    def status(self, resource_id: str) -> str:
        return self.client.echomind.get_job(resource_id).status

    @_private_sdk_request()
    def secret_key(self, resource_id: str) -> str | None:
        return self.client.echomind.get_job(resource_id).secret_key

    def endpoint(self, resource_id: str) -> str:
        base_url = os.getenv("PYROMIND_ECHOMIND_BASE_URL", "").strip()
        if not base_url:
            cluster_code = self.cluster.split("#", 1)[0]
            endpoint_by_cluster = {
                "us-west-1": "https://us-west-1.pyromind.ai",
                "us-west-2": "https://echomind-us-west-2.pyromind.ai",
                "us-west-3": "https://echomind-us-west-3.pyromind.ai",
                "cn-east-1": "https://echomind-cn-east-1.pyromind-asia.cn",
            }
            try:
                base_url = endpoint_by_cluster[cluster_code]
            except KeyError as exc:
                raise RuntimeSlotEnsureError(
                    f"No EchoMind endpoint mapping for cluster: {self.cluster}"
                ) from exc
        return f"{base_url.rstrip('/')}/echomind/{resource_id}"

    def probe(self, *, endpoint: str, access_key: str) -> RuntimeHealth:
        return self._probe(
            endpoint=endpoint.rstrip("/").removesuffix("/v1"),
            access_key=access_key,
            paths=("/status", "/v1/models"),
        )

    def probe_inference(self, *, endpoint: str, access_key: str) -> RuntimeHealth:
        base = endpoint.rstrip("/").removesuffix("/v1")
        return self._probe(endpoint=base, access_key=access_key, paths=("/v1/models",))

    def _probe(
        self, *, endpoint: str, access_key: str, paths: tuple[str, ...]
    ) -> RuntimeHealth:
        checked_at = _utc_now()
        headers = {"Authorization": f"Bearer {access_key}"}
        try:
            for path in paths:
                response = httpx.get(
                    f"{endpoint.rstrip('/')}{path}",
                    headers=headers,
                    timeout=15,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return RuntimeHealth(
                status="unhealthy",
                checked_at=checked_at,
                message=f"EchoMind health check failed: {type(exc).__name__}",
            )
        return RuntimeHealth(status="healthy", checked_at=checked_at)


class RuntimeSlotService:
    def __init__(
        self,
        *,
        projects_root: Path | str,
        provider: RuntimeSlotProvider,
    ) -> None:
        self.provider = provider
        self.store = RuntimeSlotStore(projects_root)

    async def setup(
        self,
        *,
        project_id: str,
        release_id: str,
        request: RuntimeSlotSetupRequest,
        timeout_seconds: float = 180,
        poll_seconds: float = 3.0,
    ) -> RuntimeSlot:
        try:
            async with self.store.lease(project_id):
                existing = self.store.load_optional(project_id)
                if request.serving_inference_create is None:
                    resource_id = request.resource_id()
                    if existing is not None and (
                        existing.serving_inference_submission_pending
                        or (
                            existing.serving_inference_create is not None
                            and existing.serving_inference is not None
                            and existing.serving_inference.resource_id != resource_id
                        )
                    ):
                        raise RuntimeSlotEnsureError(
                            "Continue this Project's existing Inference setup."
                        )
                    binding, key = await asyncio.to_thread(
                        self.provider.serving_binding, resource_id
                    )
                else:
                    binding, key = await self._provision_inference(
                        project_id=project_id,
                        release_id=release_id,
                        config=request.serving_inference_create,
                        existing=existing,
                        timeout_seconds=timeout_seconds,
                        poll_seconds=poll_seconds,
                    )
                return await self._ensure(
                    project_id=project_id,
                    release_id=release_id,
                    request=RuntimeSlotEnsureRequest(
                        serving_inference=binding,
                        tracer=request.tracer,
                        attach_existing_tracer_id=request.attach_existing_tracer_id,
                    ),
                    access_key=key,
                    timeout_seconds=timeout_seconds,
                    poll_seconds=poll_seconds,
                )
        except Timeout as exc:
            raise RuntimeSlotEnsureError(
                "Runtime resources are being updated."
            ) from exc

    async def _provision_inference(
        self,
        *,
        project_id: str,
        release_id: str,
        config: ServingInferenceCreateConfig,
        existing: RuntimeSlot | None,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> tuple[ServingInferenceBinding, str]:
        previous = existing or RuntimeSlot(project_id=project_id)
        resource_id = (
            previous.serving_inference.resource_id
            if previous.serving_inference
            else None
        )
        previous_config = previous.serving_inference_create
        legacy_gpu_submission = (
            previous.serving_inference_submission_pending
            and previous.status == "failed"
            and previous.error == "Inference setup failed: PyroMindAPIError"
            and previous_config is not None
            and previous_config.gpu_card != _gpu_product_code(previous_config.gpu_card)
            and _gpu_product_code(previous_config.gpu_card)
            == _gpu_product_code(config.gpu_card)
        )
        if previous_config != config and (
            resource_id
            or (
                previous.serving_inference_submission_pending
                and not legacy_gpu_submission
            )
        ):
            raise RuntimeSlotEnsureError(
                "This Project already has a different Inference configuration."
            )
        if previous.status == "ready" and resource_id:
            return await asyncio.to_thread(self.provider.serving_binding, resource_id)

        name = _inference_name(project_id)
        slot = previous.model_copy(
            update={
                "active_release_id": release_id,
                "status": "initializing",
                "serving_inference_create": config,
                "error": None,
                "updated_at": _utc_now(),
            }
        )
        self.store.save(slot)
        try:
            if not resource_id:
                job = await asyncio.to_thread(
                    self.provider.find_serving_inference, name
                )
                if job:
                    _require_matching_inference(job, config)
                    resource_id = job.id
                elif (
                    slot.serving_inference_submission_pending
                    and not legacy_gpu_submission
                ):
                    raise RuntimeSlotEnsureError(
                        "The previous Inference submission is uncertain. "
                        "Wait for the Project job to appear and try again."
                    )
                else:
                    submission = await asyncio.to_thread(
                        self.provider.prepare_serving_inference,
                        name=name,
                        config=config,
                    )
                    slot = self.store.save(
                        slot.model_copy(
                            update={
                                "serving_inference_submission_pending": True,
                                "updated_at": _utc_now(),
                            }
                        )
                    )
                    try:
                        resource_id = await asyncio.to_thread(
                            self.provider.create_serving_inference, submission
                        )
                    except PyroMindAPIError as exc:
                        rejection = _inference_rejection(exc, config)
                        if rejection:
                            slot = slot.model_copy(
                                update={"serving_inference_submission_pending": False}
                            )
                            raise RuntimeSlotEnsureError(rejection) from exc
                        raise
                if not resource_id or not re.fullmatch(
                    _RESOURCE_ID_PATTERN, resource_id
                ):
                    raise RuntimeSlotEnsureError(
                        "The platform returned an invalid Inference job ID."
                    )
                slot = self.store.save(
                    slot.model_copy(
                        update={
                            "serving_inference": ServingInferenceRuntime(
                                provider="pyromind.inference",
                                resource_id=resource_id,
                                model_uri=config.model_path,
                            ),
                            "serving_inference_submission_pending": False,
                            "updated_at": _utc_now(),
                        }
                    )
                )

            binding, key, health = await self._wait_for_inference(
                resource_id, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds
            )
            if binding.model_uri != config.model_path:
                raise RuntimeSlotEnsureError(
                    "Inference job model path differs from the approved model."
                )
            self.store.save(
                slot.model_copy(
                    update={
                        "serving_inference": ServingInferenceRuntime(
                            **binding.model_dump(), health=health
                        ),
                        "updated_at": _utc_now(),
                    }
                )
            )
            return binding, key
        except Exception as exc:
            self.store.save(
                slot.model_copy(
                    update={
                        "status": "degraded"
                        if isinstance(exc, _ServingInferencePending)
                        else "failed",
                        "error": str(exc)
                        if isinstance(exc, RuntimeSlotEnsureError)
                        else _inference_failure(
                            exc,
                            submission_pending=slot.serving_inference_submission_pending,
                        ),
                        "updated_at": _utc_now(),
                    }
                )
            )
            raise

    async def _wait_for_inference(
        self, resource_id: str, *, timeout_seconds: float, poll_seconds: float
    ) -> tuple[ServingInferenceBinding, str, RuntimeHealth]:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            try:
                binding, key = await asyncio.to_thread(
                    self.provider.serving_binding, resource_id
                )
            except _ServingInferencePending:
                pass
            else:
                health = await asyncio.to_thread(
                    self.provider.probe_inference,
                    endpoint=binding.endpoint,
                    access_key=key,
                )
                if health.status == "healthy":
                    return binding, key, health
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise _ServingInferencePending(
                    "Inference is still starting. Continue setup to check the same job."
                )
            await asyncio.sleep(min(poll_seconds, remaining))

    async def ensure(
        self,
        *,
        project_id: str,
        release_id: str,
        request: RuntimeSlotEnsureRequest,
        access_key: str,
        timeout_seconds: float = 180,
        poll_seconds: float = 3.0,
    ) -> RuntimeSlot:
        try:
            async with self.store.lease(project_id):
                return await self._ensure(
                    project_id=project_id,
                    release_id=release_id,
                    request=request,
                    access_key=access_key,
                    timeout_seconds=timeout_seconds,
                    poll_seconds=poll_seconds,
                )
        except Timeout as exc:
            raise RuntimeSlotEnsureError(
                "Runtime resources are being updated."
            ) from exc

    async def _ensure(
        self,
        *,
        project_id: str,
        release_id: str,
        request: RuntimeSlotEnsureRequest,
        access_key: str,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> RuntimeSlot:
        existing = self.store.load_optional(project_id)
        if (
            existing is not None
            and existing.tracer is not None
            and existing.tracer.resource_id is not None
            and (
                existing.tracer.target_inference_resource_id
                != request.serving_inference.resource_id
                or request.attach_existing_tracer_id
                not in (None, existing.tracer.resource_id)
            )
        ):
            raise RuntimeSlotEnsureError(
                "This Project already has different runtime resources."
            )
        if (
            existing is not None
            and existing.status == "ready"
            and existing.active_release_id == release_id
        ):
            return existing

        existing_serving = existing.serving_inference if existing else None
        serving_runtime = ServingInferenceRuntime(
            provider=request.serving_inference.provider,
            resource_id=request.serving_inference.resource_id,
            endpoint=request.serving_inference.endpoint,
            secret_ref=request.serving_inference.secret_ref,
            model_uri=request.serving_inference.model_uri,
            health=(
                existing_serving.health
                if existing_serving
                and existing_serving.resource_id
                == request.serving_inference.resource_id
                else RuntimeHealth()
            ),
        )
        initializing = RuntimeSlot(
            project_id=project_id,
            status="initializing",
            serving_inference=serving_runtime,
            serving_inference_create=existing.serving_inference_create
            if existing
            else None,
            tracer=existing.tracer if existing else None,
            tracer_submission_pending=existing.tracer_submission_pending
            if existing
            else False,
            active_release_id=release_id,
            baseline=(
                existing.baseline
                if existing
                and existing.baseline
                and existing.baseline.model_uri == serving_runtime.model_uri
                else None
            ),
            batch_execution_available=(
                existing.batch_execution_available if existing else False
            ),
        )
        self.store.save(initializing)

        tracer_resource_id = (
            existing.tracer.resource_id
            if existing is not None and existing.tracer is not None
            else None
        )
        try:
            tracer_resource_id = (
                tracer_resource_id
                or request.attach_existing_tracer_id
                or await asyncio.to_thread(
                    self.provider.find_existing, _tracer_name(project_id)
                )
            )
            if tracer_resource_id is None:
                if initializing.tracer_submission_pending:
                    raise RuntimeSlotEnsureError(
                        "The previous Tracer submission is uncertain. "
                        "Wait for the Project job to appear and try again."
                    )
                tracer_request = _tracer_job_request(
                    project_id=project_id,
                    config=request.tracer,
                    serving=request.serving_inference,
                    access_key=access_key,
                )
                initializing = self.store.save(
                    initializing.model_copy(
                        update={
                            "tracer_submission_pending": True,
                            "updated_at": _utc_now(),
                        }
                    )
                )
                try:
                    tracer_resource_id = await asyncio.to_thread(
                        self.provider.create,
                        tracer_request,
                    )
                except PyroMindAPIError as exc:
                    rejection = _tracer_rejection(exc)
                    if rejection:
                        initializing = initializing.model_copy(
                            update={"tracer_submission_pending": False}
                        )
                        raise RuntimeSlotEnsureError(rejection) from exc
                    raise
            elif not await asyncio.to_thread(
                self.provider.tracer_targets,
                tracer_resource_id,
                request.serving_inference.endpoint,
            ):
                raise RuntimeSlotEnsureError(
                    "The selected Tracer routes to another Inference."
                )
            if not tracer_resource_id or not re.fullmatch(
                _RESOURCE_ID_PATTERN, tracer_resource_id
            ):
                raise RuntimeSlotEnsureError(
                    "The platform returned an invalid Tracer job ID."
                )
            initializing = self.store.save(
                initializing.model_copy(
                    update={
                        "tracer": _tracer_runtime(tracer_resource_id, request),
                        "tracer_submission_pending": False,
                        "updated_at": _utc_now(),
                    }
                )
            )
            status = await self._wait_until_running(
                tracer_resource_id,
                timeout_seconds=timeout_seconds,
                poll_seconds=poll_seconds,
            )
            tracer_endpoint = await asyncio.to_thread(
                self.provider.endpoint,
                tracer_resource_id,
            )
            tracer_secret = await asyncio.to_thread(
                self.provider.secret_key,
                tracer_resource_id,
            )
            if not tracer_secret:
                raise RuntimeSlotEnsureError(
                    "EchoMind Tracer did not expose a proxy secret"
                )
            tracer_health = await asyncio.to_thread(
                self.provider.probe,
                endpoint=tracer_endpoint,
                access_key=tracer_secret,
            )
            serving_health = await asyncio.to_thread(
                self.provider.probe_inference,
                endpoint=request.serving_inference.endpoint,
                access_key=access_key,
            )

            slot = initializing.model_copy(
                update={
                    "status": (
                        "ready"
                        if tracer_health.status == "healthy"
                        and serving_health.status == "healthy"
                        else "degraded"
                    ),
                    "tracer": _tracer_runtime(tracer_resource_id, request).model_copy(
                        update={
                            "endpoint": tracer_endpoint,
                            "health": tracer_health,
                        }
                    ),
                    "serving_inference": serving_runtime.model_copy(
                        update={"health": serving_health}
                    ),
                    "error": (
                        None
                        if tracer_health.status == "healthy"
                        and serving_health.status == "healthy"
                        else "Tracer or Serving Inference health check failed"
                    ),
                    "updated_at": _utc_now(),
                }
            )
            if status.lower() != "running":
                slot = slot.model_copy(
                    update={
                        "status": "degraded",
                        "error": f"Unexpected EchoMind status: {status}",
                        "updated_at": _utc_now(),
                    }
                )
            return self.store.save(RuntimeSlot.model_validate(slot.model_dump()))
        except Exception as exc:
            self.store.save(
                initializing.model_copy(
                    update={
                        "status": "failed",
                        "error": (
                            str(exc)
                            if isinstance(exc, RuntimeSlotEnsureError)
                            else f"Tracer setup failed: {type(exc).__name__}"
                        ),
                        "updated_at": _utc_now(),
                    }
                )
            )
            raise

    async def refresh_health(
        self,
        *,
        project_id: str,
    ) -> RuntimeSlot:
        try:
            async with self.store.lease(project_id):
                return await self._refresh_health(project_id)
        except Timeout as exc:
            raise RuntimeSlotEnsureError(
                "Runtime resources are being updated."
            ) from exc

    async def _refresh_health(self, project_id: str) -> RuntimeSlot:
        slot = self.store.load(project_id)
        if slot.tracer is None or slot.serving_inference is None:
            raise RuntimeSlotEnsureError("RuntimeSlot resources are incomplete")
        if not slot.tracer.resource_id or not slot.tracer.endpoint:
            raise RuntimeSlotEnsureError("Tracer resource is incomplete")
        if not slot.serving_inference.resource_id:
            raise RuntimeSlotEnsureError("Serving Inference job is incomplete")
        binding, inference_key = await asyncio.to_thread(
            self.provider.serving_binding, slot.serving_inference.resource_id
        )
        if not await asyncio.to_thread(
            self.provider.tracer_targets, slot.tracer.resource_id, binding.endpoint
        ):
            updated = slot.model_copy(
                update={
                    "status": "degraded",
                    "error": "The selected Tracer routes to another Inference.",
                    "updated_at": _utc_now(),
                }
            )
            return self.store.save(updated)
        tracer_secret = await asyncio.to_thread(
            self.provider.secret_key,
            slot.tracer.resource_id,
        )
        if not tracer_secret:
            raise RuntimeSlotEnsureError("Tracer proxy secret is unavailable")
        tracer_health = await asyncio.to_thread(
            self.provider.probe,
            endpoint=slot.tracer.endpoint,
            access_key=tracer_secret,
        )
        serving_health = await asyncio.to_thread(
            self.provider.probe_inference,
            endpoint=binding.endpoint,
            access_key=inference_key,
        )
        healthy = (
            tracer_health.status == "healthy" and serving_health.status == "healthy"
        )
        updated = slot.model_copy(
            update={
                "status": "ready" if healthy else "degraded",
                "tracer": slot.tracer.model_copy(update={"health": tracer_health}),
                "serving_inference": slot.serving_inference.model_copy(
                    update={**binding.model_dump(), "health": serving_health}
                ),
                "error": None if healthy else "Runtime health check failed",
                "updated_at": _utc_now(),
            }
        )
        return self.store.save(RuntimeSlot.model_validate(updated.model_dump()))

    async def _wait_until_running(
        self,
        resource_id: str,
        *,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> str:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            status = await asyncio.to_thread(self.provider.status, resource_id)
            if status.lower() == "running":
                return status
            if status.lower() in {"error", "failed", "deleted"}:
                raise RuntimeSlotEnsureError(
                    f"EchoMind Tracer entered terminal status: {status}"
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeSlotEnsureError(
                    f"Timed out waiting for EchoMind Tracer: {resource_id}"
                )
            await asyncio.sleep(poll_seconds)


def _tracer_name(project_id: str) -> str:
    return f"echomind-project-{project_id}"


def _inference_name(project_id: str) -> str:
    return f"echomind-serving-{canonical_uuid(project_id, kind='project')}"


def _gpu_product_code(value: str) -> str:
    return _GPU_PRODUCT_LABELS.get(value, value)


def _submission_was_rejected(exc: PyroMindAPIError) -> bool:
    return exc.status_code in {400, 401, 403, 404, 422, 429}


def _tracer_rejection(exc: PyroMindAPIError) -> str | None:
    response = exc.response if isinstance(exc.response, dict) else {}
    error = response.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if (
        code == "INTERNAL_SERVER_ERROR"
        and message == "custom_runtime_script_path is required"
    ):
        return "当前集群的 Tracer 仍要求独立执行脚本。请升级平台以支持由 Project 调度。"
    if (
        code == "INTERNAL_SERVER_ERROR"
        and isinstance(message, str)
        and re.fullmatch(
            r"Service echomind is not activated"
            r"(?:\. Please activate it first\.| \(status: [a-z_]+\))",
            message,
        )
    ):
        return "当前集群的 EchoMind 服务未开通。请在 PyroMind 控制台开通后重试。"
    if _submission_was_rejected(exc) or code in (
        "PARAM_ERROR",
        "INVALID_INSTANCE_NAME",
        "INSUFFICIENT_RESOURCES",
    ):
        return "The platform rejected the Tracer configuration or quota."
    return None


def _inference_rejection(
    exc: PyroMindAPIError, config: ServingInferenceCreateConfig
) -> str | None:
    response = exc.response if isinstance(exc.response, dict) else {}
    error = response.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    if code == "INSUFFICIENT_RESOURCES":
        return (
            f"平台没有足够的 GPU（{config.gpu_count} × "
            f"{_gpu_product_code(config.gpu_card)}）。请等待资源释放后重试。"
        )
    if code in ("PARAM_ERROR", "INVALID_INSTANCE_NAME") or exc.status_code in (
        400,
        422,
    ):
        return "平台拒绝了 Inference 参数，请核对模型路径、GPU 和推理配置。"
    if exc.status_code in (401, 403):
        return "平台拒绝了 Inference 请求，请检查当前登录和 Inference 权限。"
    if _submission_was_rejected(exc):
        return f"平台拒绝了 Inference 请求（HTTP {exc.status_code}）。"
    return None


def _inference_failure(exc: Exception, *, submission_pending: bool) -> str:
    if not isinstance(exc, PyroMindAPIError):
        return f"Inference setup failed: {type(exc).__name__}"
    status = f"（HTTP {exc.status_code}）" if exc.status_code else ""
    response = exc.response if isinstance(exc.response, dict) else {}
    metadata = response.get("metadata")
    request_id = metadata.get("request_id") if isinstance(metadata, dict) else None
    reference = (
        f" 请求 ID {request_id}。"
        if isinstance(request_id, str)
        and re.fullmatch(
            r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", request_id
        )
        else ""
    )
    if submission_pending:
        return (
            f"平台未确认 Inference 创建结果{status}。"
            f"重试时会先查找已有任务。{reference}"
        )
    return f"无法读取平台 Inference 状态或配置{status}，请重试。{reference}"


def _require_matching_inference(
    job: InferenceJobResponse, config: ServingInferenceCreateConfig
) -> None:
    if (
        job.model_path != config.model_path
        or job.model_name not in {None, config.model_name}
        or job.inference_framework not in {None, config.framework}
        or job.model_length not in {None, config.max_model_len}
        or (config.image is not None and job.inf_image not in {None, config.image})
        or (
            job.resources is not None
            and (
                job.resources.gpu not in {None, str(config.gpu_count)}
                or (
                    job.resources.gpu_card is not None
                    and _gpu_product_code(job.resources.gpu_card)
                    != _gpu_product_code(config.gpu_card)
                )
            )
        )
    ):
        raise RuntimeSlotEnsureError(
            "An existing Inference for this Project differs from the approved settings."
        )


def _tracer_runtime(
    resource_id: str, request: RuntimeSlotEnsureRequest
) -> TracerRuntime:
    return TracerRuntime(
        resource_id=resource_id,
        admin_secret_ref=f"secret://echomind/{resource_id}/admin",
        proxy_secret_ref=f"secret://echomind/{resource_id}/proxy",
        target_inference_resource_id=request.serving_inference.resource_id,
        custom_runtime_script_path=request.tracer.custom_runtime_script_path,
        buffer_config=TracerBufferConfig(
            trajectory_buffer_size=request.tracer.trajectory_buffer_size,
            batch_size=request.tracer.training_batch_size,
            time_per_round_seconds=max(1, int(request.tracer.time_per_round_seconds)),
            training_save_path=request.tracer.training_save_path,
        ),
    )


def _tracer_job_request(
    *,
    project_id: str,
    config: TracerCreateConfig,
    serving: ServingInferenceBinding,
    access_key: str,
) -> EchoMindJobRequest:
    return EchoMindJobRequest(
        name=_tracer_name(project_id),
        api_url=serving.endpoint,
        api_mode=config.api_mode,
        origin_model=config.origin_model,
        access_key=access_key,
        training_model=config.training_model,
        training_batch_size=config.training_batch_size,
        trajectory_buffer_size=config.trajectory_buffer_size,
        time_per_round=config.time_per_round_seconds,
        training_round=config.training_rounds,
        training_save_path=config.training_save_path,
        resources=ResourceConfig(
            cpu=config.resources.cpu,
            memory=config.resources.memory,
            gpu=config.resources.gpu,
            gpu_card=config.resources.gpu_card,
        ),
        execution_mode=config.execution_mode,
        custom_runtime_script_path=config.custom_runtime_script_path,
    )
