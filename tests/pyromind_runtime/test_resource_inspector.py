import hashlib

import httpx
import pytest
from pyromind_agent_server import resource_inspector
from pyromind_agent_server.resource_inspector import StorageResourceInspector
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.preparation import InspectResource


@pytest.mark.parametrize(
    "path",
    ["/datasets/data.jsonl", "/workspace/datasets/data.jsonl"],
)
def test_storage_preview_is_bounded_and_never_returns_signed_urls(monkeypatch, path):
    data = b'{"prompt":"sample","answer":"ok"}\n' * 3000
    storage_paths = []

    def storage_url(**kwargs):
        storage_paths.append(kwargs["storage_path"])
        return "https://storage.example/object?signed=private"

    monkeypatch.setattr(
        resource_inspector,
        "request_storage_url",
        storage_url,
    )
    seen = []

    def response(request):
        seen.append(request)
        return httpx.Response(200, content=data)

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        monkeypatch.setattr(resource_inspector.httpx, "stream", client.stream)
        result = StorageResourceInspector().inspect(
            InspectResource(path=path, max_bytes=32),
            RequestContext(user_id="42", cookie="auth_token=test"),
        )
    assert storage_paths == ["/datasets/data.jsonl"]
    assert result["path"] == "/workspace/datasets/data.jsonl"
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
    with pytest.raises(ValueError, match="path"):
        StorageResourceInspector().inspect(
            InspectResource(path=path),
            RequestContext(user_id="42", cookie="auth_token=test"),
        )


def test_inspection_requires_current_credentials():
    with pytest.raises(ValueError, match="credentials"):
        StorageResourceInspector().inspect(
            InspectResource(path="/workspace/data"), RequestContext(user_id="42")
        )


def test_complete_digest_streams_beyond_preview_and_counts_real_lines(monkeypatch):
    data = b'{"text":"sample"}\n' * 10000 + b'\n{"text":"last"}'
    seen = []
    monkeypatch.setattr(
        resource_inspector,
        "request_storage_url",
        lambda **_: "https://storage.example/object?signed=private",
    )

    def response(request):
        seen.append(request)
        return httpx.Response(200, content=data)

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        monkeypatch.setattr(resource_inspector.httpx, "stream", client.stream)
        result = StorageResourceInspector().inspect(
            InspectResource(
                path="/workspace/data.jsonl", max_bytes=32, compute_digest=True
            ),
            RequestContext(user_id="42", cookie="auth_token=test"),
        )
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["nonempty_lines"] == 10001
    assert result["total_bytes"] == len(data)
    assert result["content"] == data[:32].decode()
    assert result["bytes_read"] == 32 and result["truncated"]
    assert result["digest_scope"] == "complete file"
    assert "range" not in seen[0].headers
    assert "signed" not in str(result)


@pytest.mark.parametrize(
    "payload,accepted",
    [
        (b"\xff\xd8\xffimage", True),
        (b"<svg><script>bad</script></svg>", False),
        (b"\xff\xd8\xff" + b"x" * (8 * 1024 * 1024), False),
    ],
    ids=["jpeg", "svg", "too_large"],
)
def test_sample_media_streams_bounded_rasters_without_exposing_signed_urls(
    monkeypatch, payload, accepted
):
    monkeypatch.setattr(
        resource_inspector,
        "request_storage_url",
        lambda **_: "https://storage.example/image?signed=private",
    )
    seen = []

    def response(request):
        seen.append(request)
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        monkeypatch.setattr(resource_inspector.httpx, "stream", client.stream)
        reader = StorageResourceInspector()
        context = RequestContext(user_id="42", cookie="auth_token=current")
        if accepted:
            assert reader.read_media("/sample.jpg", context) == (payload, "image/jpeg")
        else:
            with pytest.raises(ValueError):
                reader.read_media("/sample.jpg", context)
    assert "cookie" not in seen[0].headers
    assert "authorization" not in seen[0].headers
