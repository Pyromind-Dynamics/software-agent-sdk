"""Real Pi preparation through the platform-independent Runtime entry points."""

import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestServer
from harness_adapter.pi_adapter import PiAdapter
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import (
    InputResponseCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.preparation import (
    ChecklistField,
    PreparationPolicy,
)
from pyromind_runtime.infrastructure.sql_preparation import SQLPreparationStore
from pyromind_runtime.ports.harness import DelegatedRole, SessionSpec

from .test_preparation import preparation_setup as preparation_setup


def stream(tool=None, arguments=None):
    if tool:
        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call-{tool}",
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(arguments)},
                }
            ]
        }
    else:
        delta = {"content": "Preparation saved."}
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
    content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return web.Response(
        text=content + "data: [DONE]\n\n", content_type="text/event-stream"
    )


async def test_preparation_plan_uses_revision_delivered_with_answer(
    preparation_setup, tmp_path, monkeypatch
):
    store, owner, _ = preparation_setup
    engine = store.engine
    with engine.begin() as connection:
        from pyromind_runtime.infrastructure.sql_preparation import preparations

        connection.execute(preparations.delete())
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)
    captured = []

    async def respond(request):
        payload = await request.json()
        captured.append(payload)
        if len(captured) == 1:
            return stream(
                "update_preparation",
                {
                    "expected_revision": 1,
                    "facts": {
                        "goal": {
                            "value": "Review data",
                            "evidence_refs": list(
                                runtime._preparation_evidence(owner.conversation_id)
                            ),
                        }
                    },
                },
            )
        if len(captured) == 2:
            return stream(
                "ask_user",
                {
                    "questions": [
                        {"id": "scope", "prompt": "Review which data?", "options": []}
                    ],
                },
            )
        if len(captured) == 3:
            answer = json.loads(payload["messages"][-1]["content"])
            return stream(
                "set_preparation_plan",
                {
                    "expected_revision": answer.get("preparation_revision", 0),
                    "entry_task_id": "review",
                    "role_id": "eval",
                    "plan": {
                        "goal": "Review data",
                        "resources": [
                            {
                                "id": "data",
                                "name": "Data",
                                "storage_path": "/workspace/data.jsonl",
                            }
                        ],
                        "stages": [{"id": "eval", "kind": "eval", "name": "Eval"}],
                        "tasks": [
                            {
                                "id": "review",
                                "kind": "review",
                                "name": "Review",
                                "description": "Review the confirmed data.",
                                "stage_id": "eval",
                                "inputs": [{"name": "dataset", "resource_id": "data"}],
                            }
                        ],
                    },
                },
            )
        return stream()

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    root = tmp_path / "conversations"
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(
        root,
        adapter,
        default_harness_id="pi",
        preparation_store=SQLPreparationStore(engine),
    )
    adapter.set_preparation_handler(runtime.preparation_call)
    try:
        async with TestServer(app) as server:
            identity = owner.conversation_id
            await runtime.create_conversation(
                SessionSpec(
                    conversation_id=identity,
                    user_id=owner.user_id,
                    profile_id=owner.profile_id,
                    workspace_root=str(root / identity),
                    user_input_enabled=True,
                    tool_allowlist=(
                        "update_preparation",
                        "ask_user",
                        "set_preparation_plan",
                    ),
                    delegated_roles=(DelegatedRole(name="eval"),),
                    preparation_policy=PreparationPolicy(
                        checklist_id="review",
                        version=1,
                        allowed_roles=("eval",),
                        fields=(ChecklistField(key="goal", label="Goal"),),
                    ),
                    model_configuration={
                        "model": "eval-test-model",
                        "base_url": str(server.make_url("/v1")),
                        "api_key": "local-test-key",
                    },
                ),
                context,
            )
            await runtime.submit_command(
                identity,
                UserMessageCommand(
                    command_id="start", content=(TextContent(text="Review data"),)
                ),
                context,
            )
            async with asyncio.timeout(15):
                while not (
                    snapshot := await runtime.get_snapshot(identity, context)
                ).pending_inputs:
                    await asyncio.sleep(0.01)
                before = await runtime.get_preparation(identity, context)
                assert before.revision == 2 and before.missing == ()
                await runtime.submit_command(
                    identity,
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
                while len(captured) < 4:
                    await asyncio.sleep(0.01)
            assert (
                json.loads(captured[2]["messages"][-1]["content"])[
                    "preparation_revision"
                ]
                == 3
            )
            ready = await runtime.get_preparation(identity, context)
            assert ready.phase == "ready" and ready.revision == 4, captured[3][
                "messages"
            ][-1]
    finally:
        await runtime.close()
