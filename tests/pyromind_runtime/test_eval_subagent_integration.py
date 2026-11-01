"""Real native child process, local model transport, real SQL work queue."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, cast

from aiohttp import web
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.evaluation_configuration import (
    validate_evaluation_configuration,
)
from tests.pyromind_runtime.legacy_experiment_profile import LEGACY_EXPERIMENT_PROFILE as ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.experiment_plans import ExperimentPlans
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import RequestExperimentTask
from pyromind_runtime.domain.experiment_plans import (
    CreateExperimentPlan,
    PlanOwner,
    WorkflowPlan,
)
from pyromind_runtime.infrastructure.sql_experiment_execution import (
    SQLExperimentExecutionStore,
    metadata,
)
from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
    SQLExperimentPlanStore,
    metadata as plan_metadata,
)
from pyromind_runtime.ports.harness import SessionSpec
from sqlalchemy import create_engine


async def test_legacy_eval_native_child_prepares_proposal_through_scoped_tools(
    tmp_path, monkeypatch
):
    requests = []
    run = {}
    block_model = False
    model_started = asyncio.Event()
    release_model = asyncio.Event()
    data = b'{"prompt":"inspect","label":"ok"}\n'

    class Storage:
        def read(self, path):
            assert path == "/dataset.jsonl"
            return data

        def write(self, path, payload):
            raise AssertionError("preparation cannot write platform results")

        def list(self, prefix):
            return set()

    async def respond(request):
        payload = await request.json()
        if block_model:
            model_started.set()
            await release_model.wait()
        requests.append(payload)
        tools_used = [
            call["function"]["name"]
            for message in payload["messages"]
            for call in message.get("tool_calls", [])
        ]
        if not tools_used:
            name, arguments = (
                "read_experiment_resource",
                {"run_id": run["id"], "resource_id": "data"},
            )
        elif "submit_experiment_evaluation" not in tools_used:
            name, arguments = (
                "submit_experiment_evaluation",
                {
                    "run_id": run["id"],
                    "configuration": {
                        "input_path": "/dataset.jsonl",
                        "model_path": "/models/base",
                        "model_version": "frozen-version",
                        "dataset_config": {
                            "reference_field": "label",
                            "user_prompt_field": "prompt",
                        },
                        "evaluation_config": {
                            "mode": "agent_rubric",
                            "rubrics": [
                                {
                                    "name": "label",
                                    "criterion": "Correct label",
                                    "weight": 1,
                                    "evaluator": {"type": "exact_match"},
                                }
                            ],
                        },
                        "gpu_product": "NVIDIA-L40S",
                        "gpu_count": 1,
                    },
                    "input_fingerprints": {"data": hashlib.sha256(data).hexdigest()},
                    "expected_artifacts": ["metrics.json"],
                    "conditions": [
                        {"kind": "external_task_succeeded"},
                        {"kind": "artifacts_verified"},
                    ],
                },
            )
        else:
            name, arguments = None, None
        delta = (
            {"content": "评测方案已准备，请确认。"}
            if name is None
            else {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-" + str(len(requests)),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ]
            }
        )
        events = [
            {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "eval-test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", **delta},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "eval-test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if name else "stop",
                    }
                ],
            },
        ]
        return web.Response(
            text="".join("data: " + json.dumps(event) + "\n\n" for event in events)
            + "data: [DONE]\n\n",
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = cast(Any, site._server).sockets[0].getsockname()[1]
    monkeypatch.setenv(
        "PYROMIND_PI_RUNTIME",
        str(
            Path(__file__).resolve().parents[2]
            / "harness-adapter/pi-runtime/dist/index.js"
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    engine = create_engine(f"sqlite:///{tmp_path / 'execution.sqlite'}")
    metadata.create_all(engine)
    plan_metadata.create_all(engine)
    plans, executions = (
        SQLExperimentPlanStore(engine),
        SQLExperimentExecutionStore(engine),
    )
    root = tmp_path / "conversations"
    root.mkdir()
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(
        root,
        {"pi": adapter},
        default_harness_id="pi",
        experiment_plan_store=plans,
        experiment_execution_store=executions,
        execution_roles=("eval",),
        evaluation_storage_factory=lambda context: Storage(),
        evaluation_config_validator=validate_evaluation_configuration,
    )
    adapter.set_plan_handler(runtime.experiment_plan_call)
    adapter.set_role_handler(runtime.execution_role_call)
    profile = ECHOMIND_PRODUCT_PROFILE
    context = RequestContext(user_id="42", profile_id=profile.profile_id)
    owner = PlanOwner(
        user_id="42", conversation_id="manager", profile_id=profile.profile_id
    )
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="manager",
                user_id="42",
                profile_id=profile.profile_id,
                workspace_root=str(root / "manager"),
                product_instructions=profile.instructions,
                tool_allowlist=profile.tool_allowlist,
                skill_allowlist=profile.skill_allowlist,
                delegated_roles=profile.delegated_roles,
                managed_delegation=True,
                model_configuration={
                    "model": "eval-test-model",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "local-test-key",
                },
            ),
            context,
        )
        plan = WorkflowPlan.model_validate(
            {
                "name": "Baseline",
                "goal": "Measure the base model",
                "resources": [{"id": "data", "name": "Data", "uri": "/dataset.jsonl"}],
                "stages": [
                    {"id": "init", "name": "Init"},
                    {"id": "data", "name": "Data"},
                ],
                "edges": [{"source": "init", "target": "data"}],
                "tasks": [
                    {
                        "id": "bench",
                        "stage_id": "init",
                        "name": "Free name",
                        "inputs": [{"name": "Dataset", "resource_id": "data"}],
                    }
                ],
            }
        )
        draft = ExperimentPlans(plans).propose(owner, CreateExperimentPlan(plan=plan))
        plans.confirm(owner, draft.proposal_id, 0, "confirm-plan")
        run.update(
            await runtime.request_experiment_task(
                "manager",
                RequestExperimentTask(
                    task_id="bench",
                    expected_revision=1,
                    role_id="eval",
                    request_id="prepare",
                ),
                context,
            )
        )
        await runtime.start_execution_events()
        async with asyncio.timeout(35):
            while True:
                work = executions.queue.list(
                    __import__(
                        "pyromind_runtime.infrastructure.sql_experiment_execution",
                        fromlist=["owner_key"],
                    ).owner_key(owner)
                )
                if work and work[0]["state"] in {
                    "completed",
                    "failed",
                    "reconciliation_required",
                }:
                    assert work[0]["state"] == "completed", work
                    break
                await asyncio.sleep(0.1)
        row = executions.list(owner)[0]
        assert row["state"] == "awaiting_confirmation"
        assert row["submission"] is None  # preparing never launches platform work
        role_md = (root / "manager/pi/agent/agents/eval.md").read_text()
        assert "skills: experiment-evaluation, avi-baseline-evaluation" in role_md
        assert requests
        assert "<available_skills>" in str(requests[0]["messages"][0]["content"])
        names = {tool["function"]["name"] for tool in requests[0]["tools"]}
        assert names == {
            "read",
            "read_experiment_resource",
            "submit_experiment_evaluation",
            "get_experiment_execution",
        }
        assert "avi-baseline-evaluation" in json.dumps(requests[0]["messages"])
        assert all(
            '<active_agent name="eval"/>' in str(payload["messages"][0]["content"])
            for payload in requests
        ), [str(payload["messages"][0]["content"])[:100] for payload in requests]
        assert any(
            "preview" in str(message.get("content"))
            and "inspect" in str(message.get("content"))
            for payload in requests
            for message in payload["messages"]
            if message["role"] == "tool"
        )
        assert runtime._execution_worker is not None
        await runtime._execution_worker.close()
        from pyromind_runtime.infrastructure.sql_experiment_execution import owner_key

        control_id = executions.queue.enqueue(
            owner_key(owner),
            "role:eval",
            "control-test",
            "prepare_execution",
            {"run_id": run["id"], "owner": owner.model_dump(mode="json")},
        )
        item = executions.queue.claim(kinds=("prepare_execution",))
        assert item is not None and item["id"] == control_id
        scope = {
            "work_id": control_id,
            "generation": item["generation"],
            "run_id": run["id"],
        }
        previous_ref = work[0]["checkpoint"]["execution_ref"]
        resumed = await runtime._control_delegation(
            owner,
            "eval",
            {
                "action": "resume",
                "execution_ref": previous_ref,
                "request_id": "resume-test",
                "task": "Explain remaining limitations.",
                "scope": scope,
            },
        )
        assert resumed["execution_ref"] and resumed["execution_ref"] != previous_ref
        async with asyncio.timeout(15):
            while resumed["status"] not in {
                "succeeded",
                "failed",
                "stopped",
                "paused",
            }:
                await asyncio.sleep(0.1)
                resumed = await runtime._control_delegation(
                    owner,
                    "eval",
                    {
                        "action": "status",
                        "execution_ref": resumed["execution_ref"],
                        "scope": scope,
                    },
                )
        assert resumed["status"] in {"succeeded"}
        block_model = True
        active = await runtime._control_delegation(
            owner,
            "eval",
            {
                "action": "resume",
                "execution_ref": resumed["execution_ref"],
                "request_id": "stop-test",
                "task": "Inspect once more.",
                "scope": scope,
            },
        )
        async with asyncio.timeout(10):
            await model_started.wait()
        await runtime._control_delegation(
            owner,
            "eval",
            {
                "action": "stop",
                "execution_ref": active["execution_ref"],
                "scope": scope,
            },
        )
        async with asyncio.timeout(15):
            while True:
                state = await runtime._control_delegation(
                    owner,
                    "eval",
                    {
                        "action": "status",
                        "execution_ref": active["execution_ref"],
                        "scope": scope,
                    },
                )
                if state["status"] in {"stopped", "failed", "succeeded"}:
                    assert state["status"] == "stopped"
                    break
                await asyncio.sleep(0.1)
        executions.queue.finish(control_id, item["generation"], {"control_test": True})
    finally:
        release_model.set()
        await runtime.close()
        await server.cleanup()
