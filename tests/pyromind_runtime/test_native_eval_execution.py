"""Real Pi + pi-subagents, real SQL queue, original tool schemas, stub platform."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from aiohttp import web
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import RequestExperimentTask
from pyromind_runtime.domain.experiment_plans import PlanOwner
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


def stream(name=None, arguments=None, text="Done"):
    delta = (
        {"content": text}
        if name is None
        else {
            "tool_calls": [
                {
                    "index": 0,
                    "id": str(uuid4()),
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


@pytest.mark.parametrize("restart", [False, True])
async def test_native_eval_submits_then_resumes_for_report_and_notifies_manager(
    tmp_path, monkeypatch, restart
):
    captured, submitted, manager_requests = [], [], []
    current = {}

    async def respond(request):
        payload = await request.json()
        captured.append(payload)
        names = {item["function"]["name"] for item in payload.get("tools", [])}
        if "df_submit_evaluation" not in names:
            manager_requests.append(payload)
            return stream(text="Baseline completed. Only Eval has executed the work.")
        messages = json.dumps(payload["messages"], ensure_ascii=False)
        used = [
            call["function"]["name"]
            for message in payload["messages"]
            for call in message.get("tool_calls", [])
        ]
        if (
            '\\"phase\\": \\"report\\"' in messages
            or '"phase": "report"' in messages
            or 'phase\\": \\"report' in messages
        ):
            if "write" not in used:
                return stream(
                    "write",
                    {
                        "path": current["report"],
                        "content": "# Baseline\nOne actual sample was scored "
                        "without errors.\n",
                    },
                )
            if "record_experiment_result" not in used:
                return stream(
                    "record_experiment_result",
                    {
                        "run_id": current["id"],
                        "summary": "One sample scored, no execution errors.",
                        "report_uri": current["report"],
                    },
                )
            return stream(text="Report registered.")
        if not submitted:
            if "df_submit_evaluation" not in used:
                return stream(
                    "df_submit_evaluation",
                    {
                        "operation": "inference",
                        "input_path": "/dataset.jsonl",
                        "model_path": "/models/base",
                    },
                )
            return stream(
                "df_submit_evaluation",
                {
                    "operation": "inference",
                    "input_path": "/dataset.jsonl",
                    "model_path": "/models/base",
                    "dataset_config_path": current["root"]
                    + "/public_data/dataset.json",
                    "evaluation_config_path": current["root"]
                    + "/public_data/evaluation.json",
                },
            )
        return stream(text="Platform task submitted. Runtime will wait for it.")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    monkeypatch.setenv(
        "PYROMIND_PI_RUNTIME",
        str(
            Path(__file__).resolve().parents[2]
            / "harness-adapter/pi-runtime/dist/index.js"
        ),
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'ledger.db'}")
    plan_metadata.create_all(engine)
    metadata.create_all(engine)
    plans, store = SQLExperimentPlanStore(engine), SQLExperimentExecutionStore(engine)
    root = tmp_path / "conversations"
    root.mkdir()
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    metrics = {"total": 1, "evaluated_cases": 1, "api_or_evaluation_errors": 0}
    report = {
        "status": "succeeded",
        "evaluation_status": "validated",
        "metrics": metrics,
    }

    class Storage:
        def read(self, path):
            return {
                "metrics.json": json.dumps(metrics).encode(),
                "report.json": json.dumps(report).encode(),
                "predictions.jsonl": b'{"id":"one","prediction":"ok"}\n',
            }[path.rsplit("/", 1)[-1]]

    class Query:
        async def query(self, task_id, context):
            assert task_id == "platform-local-test" and context.user_id == "42"
            return {
                "status": "Running"
                if restart and not current.get("restarted")
                else "Succeeded"
            }

    runtime = ConversationRuntime(
        root,
        {"pi": adapter},
        default_harness_id="pi",
        experiment_plan_store=plans,
        experiment_execution_store=store,
        execution_roles=("eval",),
        external_task_query=Query(),
        evaluation_storage_factory=lambda _: Storage(),
    )
    adapter.set_plan_handler(runtime.experiment_plan_call)
    adapter.set_role_handler(runtime.execution_role_call)
    original_execute = adapter._business_tools.execute

    async def execute(name, arguments, context, **kwargs):
        if name != "df_submit_evaluation":
            return await original_execute(name, arguments, context, **kwargs)
        if "dataset_config_path" not in arguments:
            # Real original schema rejects before the platform-create guard.
            return await original_execute(name, arguments, context, **kwargs)
        await asyncio.to_thread(context.extra["managed_submission_guard"])
        submitted.append((name, arguments, context.workspace.working_dir))
        if not restart:
            stamp = datetime.now(UTC).isoformat()
            runtime.register_external_task(
                "manager",
                {
                    "task_id": "platform-local-test",
                    "kind": "workflow_run",
                    "status": "pending",
                    "managed_task_run_id": current["id"],
                    "submitted_at": stamp,
                    "updated_at": stamp,
                },
            )
        return {
            "is_error": False,
            "content": [{"type": "text", "text": "Submitted platform-local-test"}],
            "details": {
                "task_id": "platform-local-test",
                "run_id": str(uuid4()),
                "output_dir": "/evaluation/output",
                "status": "Pending",
            },
            "signals": [],
        }

    monkeypatch.setattr(adapter._business_tools, "execute", execute)
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
        row = await runtime.request_experiment_task(
            "manager",
            RequestExperimentTask.model_validate(
                {
                    "role_id": "eval",
                    "request_id": "baseline",
                    "task": {
                        "name": "自由命名的基模评测",
                        "instruction": "Run one sample baseline",
                        "inputs": {
                            "model_path": "/models/base",
                            "input_path": "/dataset.jsonl",
                        },
                    },
                }
            ),
            context,
        )
        current.update(
            id=row["id"],
            root=f"public_data/experiment-runs/{row['id']}",
            report=f"public_data/experiment-runs/{row['id']}/public_data/report.md",
        )
        await runtime.start_execution_events()
        if restart:
            async with asyncio.timeout(30):
                while not any(
                    item["kind"] == "execute_task" and item["state"] == "completed"
                    for item in store.work_items(owner)
                ):
                    await asyncio.sleep(0.1)
            await runtime.close()
            adapter = PiAdapter(root, terminal_backend="os-sandbox")
            runtime = ConversationRuntime(
                root,
                {"pi": adapter},
                default_harness_id="pi",
                experiment_plan_store=plans,
                experiment_execution_store=store,
                execution_roles=("eval",),
                external_task_query=Query(),
                evaluation_storage_factory=lambda _: Storage(),
            )
            adapter.set_plan_handler(runtime.experiment_plan_call)
            adapter.set_role_handler(runtime.execution_role_call)
            current["restarted"] = True
            # A fresh authenticated read restores credentials in memory; no
            # request credentials are saved in the queue or native transcript.
            await runtime.get_snapshot("manager", context)
            await runtime.start_execution_events()
        async with asyncio.timeout(45):
            while True:
                view = (await runtime.get_experiment_executions("manager", context))[0]
                queue = store.work_items(owner)
                if (
                    view.report_state == "ready"
                    and manager_requests
                    and any(
                        item["kind"] == "manager_turn" and item["state"] == "completed"
                        for item in queue
                    )
                    and all(item["state"] == "completed" for item in queue)
                ):
                    break
                await asyncio.sleep(0.1)
        assert len(submitted) == 1 and submitted[0][0] == "df_submit_evaluation"
        assert row["id"] in submitted[0][2]
        assert (
            view.state == "succeeded"
            and view.metrics == metrics
            and view.report_uri == current["report"]
        )
        assert plans.view(owner).current is None
        assert len(manager_requests) == 1
        if not restart:
            projected = (await runtime.get_snapshot("manager", context)).external_tasks
            assert len(projected) == 1
            assert (
                projected[0].status == "succeeded" and not projected[0].resume_pending
            )
        child_tools = {
            tool["function"]["name"]
            for payload in captured
            if payload not in manager_requests
            for tool in payload["tools"]
        }
        assert set(profile.delegated_roles[0].tools) <= child_tools
        manager_tools = {
            tool["function"]["name"] for tool in manager_requests[0]["tools"]
        }
        assert (
            "df_submit_evaluation" not in manager_tools
            and "terminal" not in manager_tools
            and "subagent" not in manager_tools
        )
        assert "/dataset.jsonl" not in json.dumps(manager_requests[0]["messages"])
        role_file = (root / "manager/pi/agent/agents/eval.md").read_text()
        assert (
            "inference-evaluation" in role_file and "df_submit_evaluation" in role_file
        )
        items = store.queue.list(store.list(owner)[0]["owner"])
        phases = [
            item
            for item in items
            if item["kind"] in {"execute_task", "generate_report"}
        ]
        assert len(phases) == 2 and all(
            item["checkpoint"].get("execution_ref") for item in phases
        )
        assert (
            phases[0]["checkpoint"]["execution_ref"]
            != phases[1]["checkpoint"]["execution_ref"]
        )
    finally:
        await runtime.close()
        await runner.cleanup()
        engine.dispose()
