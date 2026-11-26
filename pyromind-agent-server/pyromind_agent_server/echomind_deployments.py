from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import httpx
from pyromind_sdk.client.base import PyroMindAPIError
from pyromind_sdk.client.models import EchoMindJobRequest

from pyromind_agent_server.echomind_graph_runs import GraphRunStore
from pyromind_agent_server.echomind_projects import ProjectStore
from pyromind_agent_server.echomind_runtime_access import (
    platform_request_token,
    refresh_runtime_access,
)
from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.echomind_runtime_slots import (
    EchoMindTracerProvider,
    RuntimeHealth,
    RuntimeSlotStore,
    ServingInferenceRuntime,
    _private_sdk_request,
    _ServingInferencePending,
    _submission_was_rejected,
)
from pyromind_agent_server.echomind_storage import ProjectLease
from pyromind_common.tools.utils.platform_env import (
    platform_cluster_route,
    platform_cluster_target,
)
from pyromind_common.tools.utils.pyromind_api_client import (
    get_api_key,
    get_pyromind_api_client,
)
from pyromind_runtime.application.echomind_deployments import ManualDeploymentService
from pyromind_runtime.domain.base import utc_now
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_deployments import (
    DeploymentCandidate,
    DeploymentConflictError,
    DeploymentTarget,
)
from pyromind_runtime.infrastructure.file_deployment_store import FileDeploymentStore
from pyromind_runtime.ports.storage import EvaluationStorageUnavailableError


CandidateResolver = Callable[[str, str], DeploymentCandidate]


class ProjectDeploymentStore(FileDeploymentStore):
    @contextmanager
    def lease(self, project_id: str) -> Iterator[None]:
        with ProjectLease(self.root, project_id, timeout_seconds=0).acquire():
            with super().lease(project_id):
                yield


class PlatformDeploymentProvider:
    def __init__(
        self,
        *,
        projects_root: Path,
        project_id: str,
        context: RequestContext,
        platform: EchoMindTracerProvider,
        resolve_candidate: CandidateResolver,
    ) -> None:
        self.projects = ProjectStore(projects_root)
        self.slots = RuntimeSlotStore(projects_root)
        self.project_id = str(UUID(project_id))
        self.context = context
        self.platform = platform
        self.resolve_candidate = resolve_candidate

    def authorize(self, project_id: str) -> None:
        if UUID(project_id) != UUID(self.project_id):
            raise PermissionError("Deployment belongs to another Project")
        project = self.projects.load(project_id, owner_user_id=self.context.user_id)
        if self.context.x_cluster:
            self.projects.require_cluster(project, self.context.x_cluster)

    def candidate(self, candidate_id: str) -> DeploymentCandidate:
        candidate = self.resolve_candidate(self.project_id, candidate_id)
        if candidate.candidate_id != candidate_id:
            raise ValueError("Candidate ID does not match the selected report")
        return candidate

    def current(self) -> DeploymentTarget:
        slot = self.slots.load(self.project_id)
        if (
            slot.status != "ready"
            or slot.active_graph_run_id is not None
            or GraphRunStore(self.slots.projects_root).active(self.project_id)
            is not None
        ):
            raise DeploymentConflictError(
                "Project must be ready with no active training round"
            )
        if slot.serving_inference is None or not slot.serving_inference.resource_id:
            raise DeploymentConflictError(
                "No serving Inference to preserve for rollback"
            )
        binding, _ = self.platform.serving_binding(slot.serving_inference.resource_id)
        target = DeploymentTarget(
            resource_id=binding.resource_id,
            model_uri=binding.model_uri,
            endpoint=binding.endpoint,
        )
        if not self.routes_to(target):
            raise DeploymentConflictError(
                "Tracer does not target the recorded serving Inference"
            )
        return target

    def find(self, name: str, model_uri: str) -> DeploymentTarget | None:
        job = self.platform.find_serving_inference(name)
        if job is None:
            return None
        if job.model_path != model_uri:
            raise ValueError(
                "Matching deployment name points at different model weights"
            )
        return DeploymentTarget(resource_id=job.id, model_uri=job.model_path)

    def create(self, name: str, model_uri: str) -> DeploymentTarget:
        config = self.slots.load(self.project_id).serving_inference_create
        if config is None:
            raise ValueError("Project Inference resource configuration is missing")
        config = config.model_copy(update={"model_path": model_uri, "model_name": name})
        request = self.platform.prepare_serving_inference(name=name, config=config)
        try:
            resource_id = self.platform.create_serving_inference(request)
        except PyroMindAPIError as exc:
            if _submission_was_rejected(exc):
                raise ValueError("Platform rejected the candidate Inference") from None
            raise
        return DeploymentTarget(resource_id=resource_id, model_uri=model_uri)

    def ready(self, target: DeploymentTarget) -> DeploymentTarget | None:
        try:
            binding, key = self.platform.serving_binding(target.resource_id)
        except _ServingInferencePending:
            return None
        if binding.model_uri != target.model_uri:
            raise ValueError("Inference weights do not match the selected candidate")
        health = self.platform.probe_inference(
            endpoint=binding.endpoint, access_key=key
        )
        if health.status != "healthy":
            return None
        return target.model_copy(update={"endpoint": binding.endpoint})

    def _tracer_id(self) -> str:
        tracer = self.slots.load(self.project_id).tracer
        if tracer is None or tracer.resource_id is None:
            raise ValueError("Project Tracer is unavailable")
        return tracer.resource_id

    def routes_to(self, target: DeploymentTarget) -> bool:
        return bool(target.endpoint) and self.platform.tracer_targets(
            self._tracer_id(), target.endpoint or ""
        )

    @_private_sdk_request()
    def switch(self, target: DeploymentTarget) -> None:
        binding, key = self.platform.serving_binding(target.resource_id)
        if binding.model_uri != target.model_uri:
            raise ValueError("Inference model changed before switching")
        inference = self.platform.client.inference.get_job(target.resource_id)
        old = self.platform.client.echomind.get_job(self._tracer_id())
        payload = old.model_dump(include=set(EchoMindJobRequest.model_fields))
        payload.update(
            api_url=binding.endpoint,
            access_key=key,
            origin_model=inference.model_name,
        )
        request = EchoMindJobRequest.model_validate(payload)
        self.platform.client.echomind.update(self._tracer_id(), request)

    @_private_sdk_request()
    def route_ready(self, target: DeploymentTarget) -> bool:
        if not self.routes_to(target):
            return False
        inference = self.platform.client.inference.get_job(target.resource_id)
        tracer_id = self._tracer_id()
        key = self.platform.secret_key(tracer_id)
        if not key:
            return False
        base = self.platform.endpoint(tracer_id).rstrip("/").removesuffix("/v1")
        try:
            response = httpx.post(
                f"{base}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "X-Task-Trace": "false"},
                json={
                    "model": inference.model_name,
                    "messages": [{"role": "user", "content": "Reply OK."}],
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=20,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return False
        return isinstance(body, dict) and body.get("model") == inference.model_name

    @_private_sdk_request()
    def commit(self, target: DeploymentTarget) -> None:
        slot = self.slots.load(self.project_id)
        assert slot.tracer is not None
        binding, _ = self.platform.serving_binding(target.resource_id)
        inference = self.platform.client.inference.get_job(target.resource_id)
        config = slot.serving_inference_create
        if config is not None:
            config = config.model_copy(
                update={
                    "model_path": target.model_uri,
                    "model_name": inference.model_name,
                }
            )
        self.slots.save(
            slot.model_copy(
                update={
                    "status": "ready",
                    "error": None,
                    "updated_at": utc_now(),
                    "serving_inference": ServingInferenceRuntime(
                        **binding.model_dump(),
                        health=RuntimeHealth(status="healthy", checked_at=utc_now()),
                    ),
                    "serving_inference_create": config,
                    "tracer": slot.tracer.model_copy(
                        update={
                            "target_inference_resource_id": target.resource_id,
                            "health": RuntimeHealth(
                                status="healthy", checked_at=utc_now()
                            ),
                        }
                    ),
                    "baseline": None,
                }
            )
        )

    @_private_sdk_request()
    def release_candidate(
        self, target: DeploymentTarget, *, name: str, previous: DeploymentTarget
    ) -> bool:
        slot = self.slots.load(self.project_id)
        protected = {previous.resource_id}
        if slot.serving_inference is not None and slot.serving_inference.resource_id:
            protected.add(slot.serving_inference.resource_id)
        if slot.tracer is not None and slot.tracer.target_inference_resource_id:
            protected.add(slot.tracer.target_inference_resource_id)
        if target.resource_id in protected:
            return False
        try:
            job = self.platform.client.inference.get_job(target.resource_id)
        except PyroMindAPIError as exc:
            if exc.status_code == 404:
                return True
            raise
        if (
            job.id != target.resource_id
            or job.name != name
            or job.model_path != target.model_uri
        ):
            return False
        tracer = self.platform.client.echomind.get_job(self._tracer_id())
        endpoint = (job.endpoint_url or target.endpoint or "").rstrip("/")
        if tracer.origin_model == job.model_name or (
            endpoint
            and (tracer.api_url or "").rstrip("/").removesuffix("/v1")
            == endpoint.removesuffix("/v1")
        ):
            return False
        self.platform.client.inference.delete(target.resource_id)
        try:
            self.platform.client.inference.get_job(target.resource_id)
        except PyroMindAPIError as exc:
            if exc.status_code == 404:
                return True
            raise
        return False


@contextmanager
def build_manual_deployment_service(
    *,
    projects_root: Path | str,
    project_id: str,
    context: RequestContext,
    credential_store: RuntimeCredentialStore | None,
    resolve_candidate: CandidateResolver,
) -> Iterator[ManualDeploymentService]:
    root = Path(projects_root)
    project = ProjectStore(root).load(project_id, owner_user_id=context.user_id)
    if context.x_cluster:
        ProjectStore(root).require_cluster(project, context.x_cluster)
    if not project.project.cluster:
        raise ValueError("Project cluster must be selected before deployment")
    route = platform_cluster_route(project.project.cluster)
    env, _ = platform_cluster_target(route)
    refresh_runtime_access(credential_store, project, context)
    key = credential_store.load(context.user_id, route) if credential_store else None
    if not key:
        token = platform_request_token(context)
        if not token:
            raise EvaluationStorageUnavailableError(
                "Current owner credentials are required for deployment"
            )
        headers = {
            name: value
            for name, value in (
                ("cookie", context.cookie),
                ("authorization", context.authorization),
                ("x-cluster", route),
            )
            if value
        }
        key = get_api_key(env, token, headers, timeout=5)
    platform = EchoMindTracerProvider(
        client=get_pyromind_api_client(
            env=env, cluster=route, api_key=key, max_retries=0
        ),
        cluster=route,
    )
    try:
        yield ManualDeploymentService(
            ProjectDeploymentStore(root),
            PlatformDeploymentProvider(
                projects_root=root,
                project_id=project_id,
                context=context,
                platform=platform,
                resolve_candidate=resolve_candidate,
            ),
        )
    finally:
        platform.client.close()
