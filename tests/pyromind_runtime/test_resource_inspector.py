import hashlib

import httpx
import pytest
from pyromind_agent_server import resource_inspector
from pyromind_agent_server.resource_inspector import StorageResourceInspector
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.preparation import InspectResource


def test_storage_preview_is_bounded_and_never_returns_signed_urls(monkeypatch):
    data = b'{"prompt":"sample","answer":"ok"}\n' * 3000
    monkeypatch.setattr(
        resource_inspector,
        "request_storage_url",
        lambda **kwargs: "https://storage.example/object?signed=private",
    )
    seen = []

    def response(request):
        seen.append(request)
        return httpx.Response(200, content=data)

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        monkeypatch.setattr(resource_inspector.httpx, "stream", client.stream)
        result = StorageResourceInspector().inspect(
            InspectResource(path="/data.jsonl", max_bytes=32),
            RequestContext(user_id="42", cookie="auth_token=test"),
        )
    assert result["content"] == data[:32].decode()
    assert result["bytes_read"] == 32 and result["truncated"] is True
    assert result["preview_sha256"] == hashlib.sha256(data[:32]).hexdigest()
    assert "signed" not in str(result)
    assert seen[0].headers["Range"] == "bytes=0-32"
    assert "cookie" not in seen[0].headers and "authorization" not in seen[0].headers


@pytest.mark.parametrize(
    "path",
    ["../other", "https://example/data", "/data?token=private", "/data#fragment"],
)
def test_inspection_rejects_non_storage_paths(path):
    with pytest.raises(ValueError, match="Storage path"):
        StorageResourceInspector().inspect(
            InspectResource(path=path),
            RequestContext(user_id="42", cookie="auth_token=test"),
        )


def test_inspection_requires_current_credentials():
    with pytest.raises(ValueError, match="credentials"):
        StorageResourceInspector().inspect(
            InspectResource(path="/data"), RequestContext(user_id="42")
        )
