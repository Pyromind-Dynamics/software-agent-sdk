from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Protocol

import httpx
from minio.error import S3Error
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pyromind_sdk.client import StorageClient
from pyromind_sdk.client.base import DEFAULT_API_BASE_URL

from pyromind_agent_server.echomind_runtime_contracts import (
    RuntimeCredentialUnavailableError,
)
from pyromind_agent_server.echomind_runtime_credentials import RuntimeCredentialStore
from pyromind_agent_server.echomind_sandbox_storage import (
    build_bound_evaluation_storage,
)
from pyromind_agent_server.echomind_train_runtime import TrainWorkflowSpec
from pyromind_agent_server.evaluation_storage import (
    REPORT_MAX_BYTES,
    STORAGE_TIMEOUT_SECONDS,
    build_evaluation_storage,
)
from pyromind_common.tools.utils.platform_env import (
    platform_cluster_route,
    platform_headers,
    platform_portal_url,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.ports.storage import (
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


class WorkflowStorageReader(Protocol):
    def read(self, path: str) -> bytes: ...

    def list(self, prefix: str) -> set[str]: ...


class _StorageInfo(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True, str_strip_whitespace=True)

    access_key: SecretStr = Field(min_length=1)
    secret_key: SecretStr = Field(min_length=1)
    url: str = Field(min_length=1)
    bucket_name: str = Field(min_length=1)


class _SDKStorageReader:
    def __init__(self, info: _StorageInfo) -> None:
        self._bucket = info.bucket_name
        self._storage = StorageClient(
            endpoint=info.url,
            access_key=info.access_key.get_secret_value(),
            secret_key=info.secret_key.get_secret_value(),
            bucket_name=self._bucket,
        )

    def list(self, prefix: str) -> set[str]:
        folder = to_storage_api_path(prefix).strip("/") + "/"
        try:
            # StorageClient.list_files may create a missing bucket.
            objects = self._storage.client.list_objects(
                self._bucket, prefix=folder, recursive=False
            )
            names = set()
            for obj in objects:
                if obj.object_name is None or not obj.object_name.startswith(folder):
                    continue
                name = obj.object_name.removeprefix(folder).rstrip("/")
                if name and "/" not in name:
                    names.add(name)
        except Exception:
            raise EvaluationStorageUnavailableError(
                f"Workflow Storage listing failed for {prefix}"
            ) from None
        return names

    def read(self, path: str) -> bytes:
        key = to_storage_api_path(path).lstrip("/")
        try:
            response = self._storage.client.get_object(self._bucket, key)
            try:
                data = response.read(REPORT_MAX_BYTES + 1)
            finally:
                response.close()
                response.release_conn()
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                raise EvaluationStorageMissingError(
                    f"Workflow Storage object does not exist: {path}"
                ) from None
            raise EvaluationStorageUnavailableError(
                f"Workflow Storage read failed for {path}"
            ) from None
        except Exception:
            raise EvaluationStorageUnavailableError(
                f"Workflow Storage read failed for {path}"
            ) from None
        if len(data) > REPORT_MAX_BYTES:
            raise EvaluationStorageUnavailableError(
                f"Workflow Storage object exceeds the read limit: {path}"
            )
        return data


def _build_api_key_storage(*, cluster: str, access_key: str) -> _SDKStorageReader:
    headers = platform_headers({"Authorization": f"Bearer {access_key}"}, route=cluster)
    base_url = platform_portal_url(DEFAULT_API_BASE_URL, headers).rstrip("/")
    try:
        # The SDK profile model omits bucket_name and its debug logs include keys.
        response = httpx.post(
            f"{base_url}/storage_info",
            headers=headers,
            timeout=STORAGE_TIMEOUT_SECONDS,
        )
        if response.status_code in {401, 403}:
            raise RuntimeCredentialUnavailableError(
                "The workflow owner's access key was rejected by Storage."
            )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get("success") is False:
            raise ValueError("Storage credentials response was unsuccessful")
        data = body.get("data", body)
        if isinstance(data, dict) and (
            data.get("credentials_masked") is True
            or data.get("has_credentials") is False
        ):
            raise RuntimeCredentialUnavailableError(
                "Storage verification requires the project owner's browser session."
            )
        return _SDKStorageReader(_StorageInfo.model_validate(data))
    except (httpx.HTTPError, ValueError):
        raise EvaluationStorageUnavailableError(
            "Could not resolve the workflow owner's Storage credentials."
        ) from None


def build_runtime_storage(
    *,
    spec: TrainWorkflowSpec,
    request_context: RequestContext | None,
    access_key: str | None = None,
    projects_root: Path | str | None = None,
    credential_store: RuntimeCredentialStore | None = None,
) -> WorkflowStorageReader:
    project_id = spec.template_values.get("project_id")
    if projects_root is not None and project_id:
        if spec.owner_user_id is None:
            raise RuntimeCredentialUnavailableError(
                "Project Storage verification requires the workflow owner's identity."
            )
        if request_context is not None and (
            request_context.user_id != spec.owner_user_id
            or (
                request_context.x_cluster is not None
                and platform_cluster_route(request_context.x_cluster)
                != platform_cluster_route(spec.cluster)
            )
        ):
            raise PermissionError(
                "Workflow Storage context belongs to another owner or cluster."
            )
        context = (
            replace(request_context, x_cluster=spec.cluster)
            if request_context is not None
            else RequestContext(user_id=spec.owner_user_id, x_cluster=spec.cluster)
        )
        storage = build_bound_evaluation_storage(
            projects_root=projects_root,
            project_id=project_id,
            context=context,
            credential_store=credential_store,
            max_read_bytes=REPORT_MAX_BYTES,
        )
        if storage is not None:
            return storage
    if request_context is not None and (
        request_context.cookie or request_context.authorization
    ):
        return build_evaluation_storage(
            replace(
                request_context, x_cluster=request_context.x_cluster or spec.cluster
            )
        )
    if not access_key or not access_key.strip():
        raise RuntimeCredentialUnavailableError(
            "Workflow Storage access requires request credentials or the owner's "
            "platform access key."
        )
    return _build_api_key_storage(cluster=spec.cluster, access_key=access_key.strip())
