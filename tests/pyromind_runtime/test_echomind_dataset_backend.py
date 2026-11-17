from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from minio.error import S3Error
from pyromind_agent_server import (
    echomind_dataset_backend as backend_module,
    echomind_workflow_storage as storage_module,
)
from pyromind_agent_server.echomind_dataset_backend import (
    PlatformDatasetWorkflowBackend,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.storage import (
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)
from urllib3.response import HTTPResponse

from openhands.agent_server.echomind_dataset_runtime import DatasetWorkflowSpec
from openhands.agent_server.echomind_runtime_contracts import (
    RuntimeCredentialUnavailableError,
)


@pytest.fixture
def dataset_spec():
    return DatasetWorkflowSpec(
        task_name="dataset-task",
        conversation_id="conversation-1",
        cluster="us-west-1#pre",
        workflow={"nodes": []},
        output_uri="/workspace/datasets/graph-run/stage-run",
        train_filename="train_messages.jsonl",
        val_filename="val_messages.jsonl",
    )


def test_dataset_backend_requires_both_platform_storage_outputs(
    dataset_spec, monkeypatch
):
    client = Mock()
    client.studio.get_task.return_value = SimpleNamespace(status="Succeeded")
    monkeypatch.setattr(backend_module, "_client", lambda **_: client)
    storage = Mock()
    monkeypatch.setattr(backend_module, "build_runtime_storage", lambda **_: storage)
    storage.list.side_effect = [
        {"train_messages.jsonl"},
        {"train_messages.jsonl", "val_messages.jsonl"},
    ]
    backend = PlatformDatasetWorkflowBackend()

    missing = backend.query_task(task_id="1", spec=dataset_spec, request_context=None)
    succeeded = backend.query_task(task_id="1", spec=dataset_spec, request_context=None)

    assert missing.status == "failed"
    assert "val_messages.jsonl" in missing.message
    assert succeeded.status == "succeeded"
    assert succeeded.output_uri == dataset_spec.output_uri
    storage.list.assert_called_with("/datasets/graph-run/stage-run")
    assert backend.task_kind == "echomind_dataset"
    assert client.close.call_count == 2


def test_workflow_storage_preserves_browser_request_credentials(
    dataset_spec, monkeypatch
):
    build = Mock()
    monkeypatch.setattr(storage_module, "build_evaluation_storage", build)
    request_context = RequestContext(user_id="user", cookie="request-cookie")

    storage_module.build_runtime_storage(
        spec=dataset_spec, request_context=request_context, access_key="owner-key"
    )
    request = build.call_args.args[0]
    assert request.cookie == "request-cookie"
    assert request.x_cluster == dataset_spec.cluster
    assert request.authorization is None


def test_workflow_storage_does_not_use_global_service_key(dataset_spec, monkeypatch):
    monkeypatch.setenv("PYROMIND_RUNTIME_API_KEY", "unrelated-service-key")
    with pytest.raises(RuntimeCredentialUnavailableError, match="owner's"):
        storage_module.build_runtime_storage(spec=dataset_spec, request_context=None)


@pytest.mark.parametrize(
    "credential_status", [{"credentials_masked": True}, {"has_credentials": False}]
)
def test_workflow_storage_waits_for_browser_credentials(
    dataset_spec, monkeypatch, credential_status
):
    profile = httpx.Response(
        200,
        request=httpx.Request(
            "POST", "https://pre-api-portal.pyromind.ai/api/v1/storage_info"
        ),
        json={"success": True, "data": credential_status},
    )
    monkeypatch.setattr(storage_module.httpx, "post", Mock(return_value=profile))
    sdk_client = Mock()
    monkeypatch.setattr(storage_module, "StorageClient", sdk_client)

    with pytest.raises(RuntimeCredentialUnavailableError, match="browser session"):
        storage_module.build_runtime_storage(
            spec=dataset_spec, request_context=None, access_key="owner-key"
        )

    sdk_client.assert_not_called()


@pytest.fixture
def owner_storage(dataset_spec, monkeypatch):
    storage_info_url = "https://pre-api-portal.pyromind.ai/api/v1/storage_info"
    profile_request = Mock(
        return_value=httpx.Response(
            200,
            request=httpx.Request("POST", storage_info_url),
            json={
                "success": True,
                "data": {
                    "bucket_name": "user-bucket",
                    "access_key": "storage-key",
                    "secret_key": "storage-secret",
                    "url": "https://storage.example.test",
                },
            },
        )
    )
    sdk_client = Mock()
    monkeypatch.setattr(storage_module.httpx, "post", profile_request)
    monkeypatch.setattr(storage_module, "StorageClient", sdk_client)

    storage = storage_module.build_runtime_storage(
        spec=dataset_spec, request_context=None, access_key="owner-key"
    )

    assert profile_request.call_args.args == (storage_info_url,)
    assert profile_request.call_args.kwargs["headers"] == {
        "Authorization": "Bearer owner-key",
        "x-cluster": "us-west-1#pre",
    }
    sdk_client.assert_called_once_with(
        endpoint="https://storage.example.test",
        access_key="storage-key",
        secret_key="storage-secret",
        bucket_name="user-bucket",
    )
    return storage, sdk_client.return_value.client


def test_workflow_storage_uses_scoped_bucket_and_bounded_read(owner_storage):
    storage, s3 = owner_storage
    s3.list_objects.return_value = [
        SimpleNamespace(object_name="run/output/report.json"),
        SimpleNamespace(object_name="run/output/checkpoint-10/"),
        SimpleNamespace(object_name="run/output/checkpoint-10/nested.json"),
        SimpleNamespace(object_name="run/other/report.json"),
    ]
    s3.get_object.return_value.read.return_value = b"{}"

    assert storage.list("/run/output") == {"report.json", "checkpoint-10"}
    assert storage.read("/workspace/run/output/report.json") == b"{}"
    s3.list_objects.assert_called_once_with(
        "user-bucket", prefix="run/output/", recursive=False
    )
    s3.get_object.assert_called_once_with("user-bucket", "run/output/report.json")
    s3.get_object.return_value.read.assert_called_once_with(
        storage_module.REPORT_MAX_BYTES + 1
    )
    s3.get_object.return_value.close.assert_called_once()
    s3.get_object.return_value.release_conn.assert_called_once()
    s3.make_bucket.assert_not_called()


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("NoSuchKey", EvaluationStorageMissingError),
        ("AccessDenied", EvaluationStorageUnavailableError),
    ],
)
def test_workflow_storage_distinguishes_missing_from_unavailable(
    owner_storage, code, error
):
    storage, s3 = owner_storage
    s3.get_object.side_effect = S3Error(
        response=HTTPResponse(),
        code=code,
        message="details",
        resource=None,
        request_id=None,
        host_id=None,
    )

    with pytest.raises(error, match="/run/report.json"):
        storage.read("/run/report.json")
