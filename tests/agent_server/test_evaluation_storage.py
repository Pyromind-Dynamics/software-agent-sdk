import json

import httpx
import pytest

from pyromind_agent_server import evaluation_storage
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.storage import EvaluationStorageUnavailableError


@pytest.mark.parametrize(("status", "retryable"), [(502, True), (403, False)])
def test_storage_listing_classifies_gateway_failures(monkeypatch, status, retryable):
    def post(url, **kwargs):
        return httpx.Response(status, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    storage = evaluation_storage.PyromindEvaluationStorage(
        base_url="https://storage.example.test/api", headers={}
    )
    with pytest.raises(EvaluationStorageUnavailableError) as error:
        storage.list("/outputs")
    assert error.value.retryable is retryable


def test_observation_storage_reads_large_run_manifest_with_bounded_metrics(monkeypatch):
    manifest = json.dumps({"run_id": "run", "metadata": "x" * (64 * 1024)}).encode()
    metrics = b'{"total":2700}'
    files = {"/run/run_manifest.json": manifest, "/run/metrics.json": metrics}

    def download(*, storage_path, max_bytes, timeout, attempts, **kwargs):
        assert timeout == 5
        assert attempts == 2
        payload = files[storage_path]
        if len(payload) > max_bytes:
            raise ValueError("file exceeds download limit")
        return payload

    monkeypatch.setattr(evaluation_storage, "download_file_from_pyromind", download)
    storage = evaluation_storage.build_observation_evaluation_storage(
        RequestContext(user_id="test")
    )

    assert storage.read("/run/run_manifest.json") == manifest
    assert storage.read("/run/metrics.json") == metrics

    files["/run/metrics.json"] = manifest
    with pytest.raises(EvaluationStorageUnavailableError):
        storage.read("/run/metrics.json")
