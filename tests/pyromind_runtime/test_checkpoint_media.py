"""Checkpoint media is read-only, owner-scoped and limited to verified records."""

import base64
import json

import httpx
import pytest
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.profiles import ConversationProfile
from pyromind_runtime.application.checkpoint_media import checkpoint_media_path
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import SessionSpec

from . import test_execution_checkpoints as checkpoint_fixtures
from .fake_adapter import FakeAdapter
from .test_product_api import _app


sample_execution = checkpoint_fixtures.sample_execution
claim = checkpoint_fixtures.claim


async def test_media_http_uses_runtime_and_rejects_other_owners_and_old_checkpoints(
    sample_execution,
    tmp_path,
):
    service, owner, run, files, document, read, inspect = sample_execution
    rows = [json.loads(line) for line in files[document["sample_input"]].splitlines()]
    for row in rows:
        row["images"] = ["images/before.jpg"]
    files[document["sample_input"]] = (
        "\n".join(json.dumps(row) for row in rows)
    ).encode()
    scope = claim(service, run["id"])

    async def execute(*_):
        raise AssertionError("media must not submit work")

    result = await service.role_call(
        owner,
        "cleaner",
        "record_execution_checkpoint",
        document,
        scope,
        execute,
        read,
        inspect,
    )
    checkpoint = result["checkpoint"]["checkpoint_id"]
    calls = []

    class Preview:
        def inspect(self, request, context):
            raise AssertionError("not a text preview")

        def read_media(self, path, context):
            calls.append((path, context.cookie))
            return b"\xff\xd8\xffsample", "image/jpeg"

    runtime = ConversationRuntime(
        tmp_path / "product",
        {"pi": FakeAdapter("pi")},
        default_harness_id="pi",
        experiment_plan_store=service.plans,
        experiment_execution_store=service.store,
        execution_roles=("cleaner",),
        resource_inspector=Preview(),
    )
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)
    (tmp_path / "product" / owner.conversation_id).mkdir(parents=True)
    await runtime.create_conversation(
        SessionSpec(
            conversation_id=owner.conversation_id,
            user_id=owner.user_id,
            profile_id=owner.profile_id,
            workspace_root=str(tmp_path / "workspace"),
            tool_allowlist=("get_experiment_execution",),
        ),
        context,
    )
    profile = ConversationProfile(
        profile_id=owner.profile_id, route_prefix="/api/v2/sample/conversations"
    )
    app = _app(tmp_path, runtime)
    app.include_router(create_product_router(profile))
    base = f"{profile.route_prefix}/{owner.conversation_id}/experiment-executions"
    path = f"{base}/{run['id']}/checkpoints/{checkpoint}/media/0/0"
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={
                "x-pyromind-debug-user-id": owner.user_id,
                "cookie": "auth_token=current",
            },
        ) as client:
            response = await client.get(path)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "private, no-store"
            assert (
                response.json()["url"]
                == "data:image/jpeg;base64,"
                + base64.b64encode(b"\xff\xd8\xffsample").decode()
            )
            assert calls == [("/images/before.jpg", "auth_token=current")]
            for wrong in (
                path.replace(checkpoint, "old-checkpoint"),
                path.replace(run["id"], "another-task"),
                path.replace("/media/0/0", "/media/-1/0"),
                path.replace("/media/0/0", "/media/0/9"),
            ):
                assert (await client.get(wrong)).status_code == 404
            assert (
                await client.get(path, headers={"x-pyromind-debug-user-id": "other"})
            ).status_code == 404
            foreign_profile = path.replace(
                profile.route_prefix, "/api/v2/pyromind/conversations"
            )
            assert (await client.get(foreign_profile)).status_code == 404
            assert len(calls) == 1
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "path",
    [
        "../secret.jpg",
        "https://example/image.jpg",
        "/file?token=secret",
        "/file#fragment",
        "bad\\file",
    ],
)
def test_media_paths_cannot_escape_to_urls_or_parent_directories(path):
    with pytest.raises(ValueError):
        checkpoint_media_path(
            {
                "source_path": "/data/full.jsonl",
                "preview": [{"input_records": [{"images": [path]}]}],
            },
            0,
            0,
        )
