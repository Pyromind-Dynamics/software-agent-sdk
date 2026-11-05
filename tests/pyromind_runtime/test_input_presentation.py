"""Display content round-trips independently of decisions and task documents."""

import base64
import json

import httpx
import pytest
from pydantic import ValidationError
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.profiles import ConversationProfile
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.input_presentation import input_media_path
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.user_input import PendingUserInput, UserInputRequest
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter
from .test_product_api import _app


QUESTIONS: list[dict] = [{"id": "adopt", "prompt": "Use this model?", "options": []}]
PRESENTATION: dict = {
    "title": "Model comparison",
    "content": [
        {"type": "text", "text": "| model | score |\n|---|---|\n| A | .9 |"},
        {"type": "image_url", "url": "/results/plot.png"},
        {"type": "image", "image_urls": ["/results/second.jpg"]},
        {"type": "text", "text": "[Report](https://example.com/report.json)"},
    ],
}


def test_presentation_is_optional_bounded_content_and_cannot_declare_actions():
    legacy = UserInputRequest.model_validate({"questions": QUESTIONS})
    assert "presentation" not in legacy.model_dump(mode="json")
    request = UserInputRequest.model_validate(
        {"questions": QUESTIONS, "presentation": PRESENTATION}
    )
    pending = PendingUserInput(request_id="r", **request.model_dump(mode="json"))
    assert pending.presentation is not None
    assert (
        pending.presentation.model_dump(mode="json")["content"]
        == PRESENTATION["content"]
    )
    assert input_media_path(pending, 1, 0) == "/results/plot.png"
    assert input_media_path(pending, 2, 0) == "/results/second.jpg"
    for content in (
        [{"type": "html", "html": "<button>approve</button>"}],
        [{"type": "text", "text": "x" * 65536}],
        [{"type": "image_url", "url": "https://example.com/image?signature=secret"}],
    ):
        with pytest.raises(ValidationError):
            UserInputRequest.model_validate(
                {"questions": QUESTIONS, "presentation": {"content": content}}
            )
    with pytest.raises(ValidationError):
        UserInputRequest.model_validate(
            {
                "questions": QUESTIONS,
                "presentation": {**PRESENTATION, "command": "submit"},
            }
        )
    for index, image in ((-1, 0), (0, 0), (1, -1), (1, 1), (100, 0)):
        with pytest.raises(FileNotFoundError):
            input_media_path(pending, index, image)


@pytest.mark.parametrize(
    "path",
    [
        "/../secret.jpg",
        "//external/file",
        "https://example.com/img",
        "/data?token=x",
        "relative.jpg",
    ],
)
def test_presentation_media_cannot_escape_declared_storage_path(path):
    pending = PendingUserInput.model_validate(
        {
            "request_id": "r",
            "questions": QUESTIONS,
            "presentation": {"content": [{"type": "image_url", "url": path}]},
        }
    )
    with pytest.raises(ValueError):
        input_media_path(pending, 0, 0)


async def test_pending_media_and_snapshot_use_owner_scoped_runtime_without_execution(
    tmp_path,
):
    reads = []

    class Reader:
        def inspect(self, request, context):
            raise AssertionError("not text")

        def read_media(self, path, context):
            reads.append((path, context.cookie))
            return b"\x89PNG\r\n\x1a\nimage", "image/png"

    runtime = ConversationRuntime(
        tmp_path / "conversations", FakeAdapter(), resource_inspector=Reader()
    )
    (tmp_path / "conversations" / "c").mkdir(parents=True)
    context = RequestContext(user_id="42", profile_id="reviewer")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="c",
            user_id="42",
            profile_id="reviewer",
            workspace_root=str(tmp_path / "workspace"),
        ),
        context,
    )
    store = runtime._store("c")
    store.append(
        ProductEvent(
            event_id="input:r",
            conversation_id="c",
            type="input.requested",
            payload=json.loads(
                json.dumps(
                    {
                        "request_id": "r",
                        "questions": QUESTIONS,
                        "presentation": PRESENTATION,
                    }
                )
            ),
        )
    )
    app = _app(tmp_path, runtime)
    profile = ConversationProfile(
        profile_id="reviewer", route_prefix="/api/v2/reviewer/conversations"
    )
    app.include_router(create_product_router(profile))
    path = profile.route_prefix + "/c/pending-inputs/r/media/1/0"
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"x-pyromind-debug-user-id": "42", "cookie": "auth_token=current"},
        ) as client:
            snapshot = await client.get(profile.route_prefix + "/c/snapshot")
            assert (
                snapshot.json()["pending_inputs"][0]["presentation"]["content"]
                == PRESENTATION["content"]
            )
            response = await client.get(path)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "private, no-store"
            assert (
                response.json()["url"]
                == "data:image/png;base64,"
                + base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode()
            )
            assert reads == [("/results/plot.png", "auth_token=current")]
            assert (
                await client.get(path, headers={"x-pyromind-debug-user-id": "43"})
            ).status_code == 404
            assert (
                await client.get(
                    path.replace(profile.route_prefix, "/api/v2/pyromind/conversations")
                )
            ).status_code == 404
            for suffix in ("-1/0", "1/2", "0/0", "9/0"):
                assert (
                    await client.get(path.rsplit("/", 2)[0] + "/" + suffix)
                ).status_code == 404
            store.append(
                ProductEvent(
                    event_id="resolved:r",
                    conversation_id="c",
                    type="input.resolved",
                    payload={"request_id": "r", "outcome": "interrupted"},
                )
            )
            assert (await client.get(path)).status_code == 404
            assert len(reads) == 1
            assert store.load_snapshot().project is None
    finally:
        await runtime.close()
