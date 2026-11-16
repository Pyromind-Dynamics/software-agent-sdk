import httpx
import pytest
from pyromind_agent_server import resource_inspector
from pyromind_agent_server.resource_inspector import StorageResourceInspector
from pyromind_runtime.domain.context import RequestContext


@pytest.mark.parametrize(
    "payload,accepted",
    [
        (b"\xff\xd8\xffimage", True),
        (b"<svg><script>bad</script></svg>", False),
        (b"\xff\xd8\xff" + b"x" * (8 * 1024 * 1024), False),
    ],
    ids=["jpeg", "svg", "too_large"],
)
def test_sample_media_remains_bounded_without_exposing_credentials(
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


@pytest.mark.parametrize(
    "path", ["../secret", "https://example/image", "/data?token=x"]
)
def test_media_rejects_paths_outside_storage(path):
    with pytest.raises(ValueError):
        StorageResourceInspector().read_media(path, RequestContext(user_id="42"))
