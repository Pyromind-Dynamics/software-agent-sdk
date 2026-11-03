"""Real Pi + pi-subagents, real SQL queue, original tool schemas, stub platform."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from aiohttp import web
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import UserMessageCommand
from pyromind_runtime.domain.content import JsonObject, TextContent
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
from pyromind_runtime.infrastructure.sql_work_queue import work
from pyromind_runtime.ports.harness import SessionSpec
from sqlalchemy import create_engine

from . import test_standalone_experiments as execution_fixtures
from .test_evaluation_cases import FakeStorage


experiment = execution_fixtures.experiment


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


@pytest.mark.parametrize("restart", [False, "platform", "preparation"])
async def test_native_eval_submits_then_resumes_for_report_and_notifies_manager(
    tmp_path, monkeypatch, restart
):
    captured, submitted, manager_requests = [], [], []
    current = {}
    preparation = asyncio.Event()
    resume_preparation = asyncio.Event()

    async def respond(request):
        payload = await request.json()
        captured.append(payload)
        names = {item["function"]["name"] for item in payload.get("tools", [])}
        if "record_experiment_result" not in names:
            manager_requests.append(payload)
            if not any(
                message.get("role") == "tool" for message in payload["messages"]
            ):
                return stream(
                    "read_experiment_artifact",
                    {
                        "run_id": current["id"],
                        "name": "agent_report",
                    },
                )
            return stream(text="Baseline completed. Report verified.")
        if restart == "preparation" and not current.get("restarted"):
            preparation.set()
            await resume_preparation.wait()
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
    port = runner.addresses[0][1]
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

    storage = FakeStorage(
        {
            "/evaluation/output/metrics.json": json.dumps(metrics).encode(),
            "/evaluation/output/report.json": json.dumps(report).encode(),
            "/evaluation/output/predictions.jsonl": b'{"id":"one","prediction":"ok"}\n',
        }
    )

    class Query:
        async def query(self, task_id: str, context: RequestContext) -> JsonObject:
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
        evaluation_storage_factory=lambda _: storage,
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
            if restart == "preparation":
                await asyncio.wait_for(preparation.wait(), 30)
            else:
                async with asyncio.timeout(30):
                    while not any(
                        item["kind"] == "execute_task" and item["state"] == "completed"
                        for item in store.work_items(owner)
                    ):
                        await asyncio.sleep(0.1)
            await runtime.close()
            if restart == "preparation":
                # Advance the expired lease boundary without a 90-second wait.
                with engine.begin() as connection:
                    connection.execute(
                        work.update()
                        .where(work.c.state == "running")
                        .values(
                            lease_until=datetime.now(UTC).replace(tzinfo=None)
                            - timedelta(seconds=1)
                        )
                    )
            adapter = PiAdapter(root, terminal_backend="os-sandbox")
            original_execute = adapter._business_tools.execute
            monkeypatch.setattr(adapter._business_tools, "execute", execute)
            runtime = ConversationRuntime(
                root,
                {"pi": adapter},
                default_harness_id="pi",
                experiment_plan_store=plans,
                experiment_execution_store=store,
                execution_roles=("eval",),
                external_task_query=Query(),
                evaluation_storage_factory=lambda _: storage,
            )
            adapter.set_plan_handler(runtime.experiment_plan_call)
            adapter.set_role_handler(runtime.execution_role_call)
            current["restarted"] = True
            resume_preparation.set()
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
        assert len(manager_requests) == 2
        assert "One actual sample was scored" in json.dumps(
            manager_requests[-1]["messages"]
        )
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
        assert profile.delegated_roles is not None
        assert set(profile.delegated_roles[0].tools) <= child_tools
        manager_tools = {
            tool["function"]["name"] for tool in manager_requests[0]["tools"]
        }
        assert (
            "df_submit_evaluation" in manager_tools
            and "terminal" in manager_tools
            and "read" in manager_tools
            and "subagent" not in manager_tools
            and "record_experiment_result" not in manager_tools
            and "read_skill" not in manager_tools
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
        snapshot = await runtime.get_snapshot("manager", context)
        children = {
            entry.run_id: entry
            for entry in snapshot.timeline
            if entry.kind == "operation" and entry.category == "subagent"
        }
        first_ref = phases[0]["checkpoint"]["execution_ref"]
        report_ref = phases[1]["checkpoint"]["execution_ref"]
        first_details = children[first_ref].details
        report_details = children[report_ref].details
        assert isinstance(first_details, dict) and isinstance(report_details, dict)
        first_display = first_details["subagent"]
        report_display = report_details["subagent"]
        assert isinstance(first_display, dict) and isinstance(report_display, dict)
        assert first_display["task"] == report_display["task"] == "自由命名的基模评测"
        assert first_display["phase"] == "execute"
        assert report_display["phase"] == "report"
        assert report_display["previous_run_id"] == first_ref
    finally:
        await runtime.close()
        await runner.cleanup()
        engine.dispose()


async def test_manager_inspects_and_plans_without_delegating_or_submitting(
    tmp_path, monkeypatch, experiment
):
    service, _, _ = experiment
    profile = ECHOMIND_PRODUCT_PROFILE
    captured = []
    reference = {
        "goal": "Create a baseline plan without platform submission",
        "resources": [
            {"id": "bench", "name": "Evaluation data", "uri": "public_data/val.jsonl"}
        ],
        "stages": [
            {"id": "init", "name": "Init"},
            {
                "id": "inference",
                "name": "Inference",
                "placeholder": True,
                "enabled": False,
            },
        ],
        "tasks": [
            {
                "id": "baseline",
                "stage_id": "init",
                "name": "Baseline",
                "inputs": [{"name": "dataset", "resource_id": "bench"}],
            }
        ],
    }
    script = [
        ("read", {"path": ".agents/skills/experiment-planning/SKILL.md"}),
        ("terminal", {"command": "wc -l public_data/val.jsonl"}),
        ("read", {"path": "public_data/val.jsonl"}),
        ("update_experiment_context", {"context": reference, "expected_sequence": 0}),
    ]

    async def respond(request):
        payload = await request.json()
        captured.append(payload)
        if len(captured) <= len(script):
            name, arguments = script[len(captured) - 1]
            return stream(name, arguments)
        return stream(
            text="Data inspected and reference plan saved without submission."
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = server.addresses[0][1]
    root = tmp_path / "conversations"
    data = root / "manager/public_data/val.jsonl"
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    adapter = PiAdapter(root, terminal_backend="os-sandbox")
    runtime = ConversationRuntime(
        root,
        {"pi": adapter},
        default_harness_id="pi",
        experiment_plan_store=service.plans,
        experiment_execution_store=service.store,
        execution_roles=("eval",),
    )
    adapter.set_plan_handler(runtime.experiment_plan_call)
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
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_text('{"id":"sample","label":"pass"}\n')
        await runtime.start_execution_events()
        await runtime.submit_command(
            "manager",
            UserMessageCommand(
                command_id="plan-only",
                content=(
                    TextContent(
                        text=(
                            "Inspect data and create a reference plan. "
                            "Do not submit platform tasks."
                        )
                    ),
                ),
            ),
            context,
        )
        async with asyncio.timeout(30):
            while True:
                snapshot = await runtime.get_snapshot("manager", context)
                if snapshot.status == "error":
                    pytest.fail(str(snapshot.timeline))
                if len(captured) == 5 and snapshot.status == "idle":
                    break
                await asyncio.sleep(0.05)
        saved = service.plans.context(owner)
        assert saved.context.tasks[0].inputs[0].resource_id == "bench"
        assert service.store.list(owner) == []
        assert service.plans.view(owner).current is None
        results = [
            message
            for message in captured[-1]["messages"]
            if message.get("role") == "tool"
        ]
        assert len(results) == 4
        assert "sample" in str(results[2]["content"])
        assert '"sequence": 1' in str(results[3]["content"])
        assert all("request_failed" not in str(result) for result in results)
        names = {tool["function"]["name"] for tool in captured[0]["tools"]}
        assert {
            "read",
            "terminal",
            "write",
            "df_submit_evaluation",
            "read_experiment_artifact",
        } <= names
        assert not {"read_skill", "subagent", "record_experiment_result"} & names
        prompt = json.dumps(captured[0]["messages"])
        assert "inference-evaluation" in prompt and "data-processing" in prompt
    finally:
        await runtime.close()
        await server.cleanup()
