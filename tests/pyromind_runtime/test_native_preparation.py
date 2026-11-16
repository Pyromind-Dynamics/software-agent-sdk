"""Real Pi calls with model responses supplied by a local streaming server."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from harness_adapter.pi_adapter import PiAdapter
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import InputResponseCommand, UserMessageCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.harness import DelegatedRole, SessionSpec

from .test_preparation import (
    preparation_setup as preparation_setup,
    ready_request as ready_request,
)


def stream(tool=None, arguments=None, call="call"):
    delta = (
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": call,
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(arguments)},
                }
            ]
        }
        if tool
        else {"content": "Plan submitted for review."}
    )
    chunks = [
        {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if tool else "stop",
                }
            ]
        },
    ]
    return web.Response(
        text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        + "data: [DONE]\n\n",
        content_type="text/event-stream",
    )


@pytest.mark.parametrize("concurrent_edit", [False, True])
async def test_single_plan_tool_after_question_and_stale_model_request(
    preparation_setup, ready_request, tmp_path, monkeypatch, concurrent_edit
):
    store, owner, policy = preparation_setup
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)
    captured = []
    request_started, release = asyncio.Event(), asyncio.Event()

    async def respond(request):
        payload = await request.json()
        captured.append(payload)
        if len(captured) == 1:
            return stream(
                "ask_user",
                {
                    "questions": [
                        {"id": "scope", "prompt": "Which data?", "options": []}
                    ]
                },
                "question",
            )
        if len(captured) == 2:
            request_started.set()
            await release.wait()
            return stream(
                "set_preparation_plan",
                ready_request.model_dump(mode="json"),
                "plan-first",
            )
        if concurrent_edit and len(captured) == 3:
            return stream(
                "set_preparation_plan",
                ready_request.model_dump(mode="json"),
                "plan-latest",
            )
        return stream()

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    root = tmp_path / "conversations"
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(
        root, adapter, default_harness_id="pi", preparation_store=store
    )
    adapter.set_preparation_handler(
        runtime.preparation_call, runtime.preparation_revision
    )
    try:
        async with TestServer(app) as server:
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=owner.conversation_id,
                    user_id=owner.user_id,
                    profile_id=owner.profile_id,
                    workspace_root=str(root / owner.conversation_id),
                    user_input_enabled=True,
                    tool_allowlist=("ask_user", "set_preparation_plan"),
                    delegated_roles=(DelegatedRole(name="eval"),),
                    preparation_policy=policy,
                    model_configuration={
                        "model": "eval-test-model",
                        "base_url": str(server.make_url("/v1")),
                        "api_key": "local-test-key",
                    },
                ),
                context,
            )
            await runtime.submit_command(
                owner.conversation_id,
                UserMessageCommand(
                    command_id="start", content=(TextContent(text="Evaluate data"),)
                ),
                context,
            )
            async with asyncio.timeout(20):
                while not (
                    snapshot := await runtime.get_snapshot(
                        owner.conversation_id, context
                    )
                ).pending_inputs:
                    await asyncio.sleep(0.01)
                await runtime.submit_command(
                    owner.conversation_id,
                    InputResponseCommand.model_validate(
                        {
                            "command_id": "answer",
                            "request_id": snapshot.pending_inputs[0].request_id,
                            "status": "answered",
                            "answers": [{"question_id": "scope", "text": "All"}],
                        }
                    ),
                    context,
                )
                await request_started.wait()
                if concurrent_edit:
                    store.invalidate(owner, "newer-input")
                release.set()
                while (
                    await runtime.get_preparation(owner.conversation_id, context)
                ).phase != "ready":
                    await asyncio.sleep(0.01)
            ready = store.get(owner)
            assert ready.revision == (4 if concurrent_edit else 3)
            assert store.project(owner) is None
            names = {tool["function"]["name"] for tool in captured[0]["tools"]}
            assert names == {"ask_user", "set_preparation_plan"}
            answer = json.loads(captured[1]["messages"][-1]["content"])
            assert set(answer) == {"status", "answers"}
            properties = next(
                tool["function"]["parameters"]["properties"]
                for tool in captured[0]["tools"]
                if tool["function"]["name"] == "set_preparation_plan"
            )
            assert set(properties) == {"plan", "entry_task_id"}
            if concurrent_edit:
                assert "Review context changed" in json.dumps(captured[2]["messages"])
    finally:
        release.set()
        await runtime.close()
