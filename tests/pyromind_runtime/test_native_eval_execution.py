"""Real Pi + pi-subagents, real SQL queue, original tool schemas, stub platform."""

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pytest
from aiohttp import web
from harness_adapter.pi_adapter import PiAdapter
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE as PREPARED_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.commands import (
    CancelCommand,
    ControlExperimentTaskCommand,
    CreateProjectCommand,
    InputResponseCommand,
    PermissionResponseCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.experiment_execution import RequestExperimentTask
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.preparation import PreparationView, UpdatePreparation
from pyromind_runtime.infrastructure.sql_experiment_execution import (
    SQLExperimentExecutionStore,
    metadata,
)
from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
    SQLExperimentPlanStore,
    metadata as plan_metadata,
)
from pyromind_runtime.infrastructure.sql_preparation import SQLPreparationStore
from pyromind_runtime.infrastructure.sql_work_queue import work
from pyromind_runtime.ports.harness import SessionSpec
from sqlalchemy import create_engine

from . import test_standalone_experiments as execution_fixtures
from .legacy_experiment_profile import EXPERIMENT_PROFILE_V4 as ECHOMIND_PRODUCT_PROFILE
from .test_evaluation_cases import FakeStorage


experiment = execution_fixtures.experiment


@pytest.mark.parametrize("delivery", ["pending", "settled", "accepted"])
async def test_manager_queue_recognizes_completion_of_steered_messages(
    experiment, tmp_path, delivery
):
    from .fake_adapter import FakeAdapter

    service, owner, _ = experiment
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)

    def events(
        command: UserMessageCommand,
    ) -> Iterator[
        tuple[Literal["message.completed", "status.changed"], str, JsonObject]
    ]:
        run_id = f"extension:{command.command_id}"
        yield (
            "message.completed",
            run_id,
            {
                "message_id": command.command_id,
                "role": "user",
                "command_id": command.command_id,
                "content": [block.model_dump(mode="json") for block in command.content],
            },
        )
        yield "status.changed", run_id, {"status": "finished"}

    class SteeringAdapter(FakeAdapter):
        async def send(self, handle, command, context) -> JsonObject:
            assert isinstance(command, UserMessageCommand)
            self.sent.append((handle.session_id, command, context))
            for kind, run_id, payload in events(command):
                self.emit(
                    handle.session_id,
                    kind,
                    payload,
                    event_id=f"{command.command_id}:{kind}",
                    run_id=run_id,
                )
            return {"accepted": True, "steered": True}

        async def notify_external_task(
            self, handle, notification, context
        ) -> JsonObject:
            self.external_task_notifications.append(
                (handle.session_id, notification, context)
            )
            self.emit(
                handle.session_id,
                "status.changed",
                {"status": "finished"},
                event_id="recovered-message:finished",
                run_id=f"callback:{notification.task_id}:succeeded",
            )
            return {"accepted": True}

    adapter = SteeringAdapter()
    runtime = ConversationRuntime(
        tmp_path / "product",
        adapter,
        experiment_plan_store=service.plans,
        experiment_execution_store=service.store,
        execution_roles=("eval",),
    )
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=owner.conversation_id,
                user_id=owner.user_id,
                profile_id=owner.profile_id,
                workspace_root=str(tmp_path / "product" / owner.conversation_id),
                tool_allowlist=("get_experiment_context",),
            ),
            context,
        )
        first = UserMessageCommand(
            command_id="first",
            content=(TextContent(text="Inspect the existing task."),),
        )
        receipt = await runtime.submit_command(owner.conversation_id, first, context)
        product = runtime._store(owner.conversation_id)
        if delivery != "pending":
            product.complete_command(receipt.model_copy(update={"status": "completed"}))
        if delivery == "settled":
            for kind, run_id, payload in events(first):
                product.append(
                    ProductEvent(
                        conversation_id=owner.conversation_id,
                        type=kind,
                        run_id=run_id,
                        payload=payload,
                    )
                )
        await runtime.submit_command(
            owner.conversation_id,
            UserMessageCommand(
                command_id="second",
                content=(TextContent(text="Continue from the existing result."),),
            ),
            context,
        )
        await runtime.start_execution_events()
        async with asyncio.timeout(5):
            while any(
                item["state"] != "completed" for item in service.store.work_items(owner)
            ):
                await asyncio.sleep(0.05)
        assert [command.command_id for _, command, _ in adapter.sent] == (
            ["first", "second"] if delivery == "pending" else ["second"]
        )
        notifications = adapter.external_task_notifications
        if delivery == "accepted":
            assert len(notifications) == 1
            notification = notifications[0][1]
            assert notification.task_id == "accepted-message:first"
            assert "Inspect the existing task." in notification.hidden_text
        else:
            assert not notifications
    finally:
        await runtime.close()


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


@pytest.mark.parametrize("scenario", ["idle", "busy", "restart", "independent"])
async def test_plugin_delivers_dataset_result_and_manager_delegates_init(
    tmp_path, monkeypatch, scenario
):
    profile = PREPARED_PROFILE
    assert profile.delegated_roles is not None
    owner = PlanOwner(
        user_id="42", conversation_id="manager", profile_id=profile.profile_id
    )
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)
    artifact = "/datasets/prepared/train.jsonl"
    dataset_id = ""
    report_path = ""
    init_inputs = []
    manager_queries = []
    foreground_replies = []
    busy_started, release_manager = asyncio.Event(), asyncio.Event()
    report_written, release_dataset = asyncio.Event(), asyncio.Event()

    async def respond(request):
        payload = await request.json()
        names = {tool["function"]["name"] for tool in payload.get("tools", [])}
        messages = payload["messages"]
        history = json.dumps(messages)
        used = {
            call["function"]["name"]
            for message in messages
            for call in message.get("tool_calls", [])
        }
        assert "record_experiment_result" not in names
        if "df_submit_evaluation" in names:
            init_inputs.append(history)
            return stream(text="Init needs an additional input.")
        if "df_submit_pipeline" in names:
            if "terminal" not in used:
                return stream("terminal", {"command": "rm -rf public_data/test-empty"})
            if "write" not in used:
                return stream(
                    "write",
                    {
                        "path": report_path,
                        "content": f"# Dataset report\nTraining file: `{artifact}`.\n",
                    },
                )
            if scenario == "restart":
                report_written.set()
                await release_dataset.wait()
            return stream(text=f"DATASET_HANDOFF\n\nReport: `{report_path}`.")
        if (
            "DATASET_HANDOFF" not in history
            and "Continue from saved dataset" not in history
        ):
            if scenario == "independent":
                foreground_replies.append(history)
                return stream(text="Manager can respond while Dataset waits.")
            busy_started.set()
            await release_manager.wait()
            return stream(text="Still working on the project.")
        if "get_experiment_execution" not in used:
            manager_queries.append(history)
            return stream("get_experiment_execution", {"run_id": dataset_id})
        if "read" not in used:
            return stream("read", {"path": report_path})
        if "request_experiment_task" not in used:
            assert artifact in history
            return stream(
                "request_experiment_task",
                {
                    "role_id": "init",
                    "request_id": "init-after-dataset",
                    "task": {
                        "name": "Baseline",
                        "instruction": "Use the verified dataset",
                        "inputs": {"input_path": artifact},
                    },
                },
            )
        return stream(text="Baseline delegated.")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    server = web.AppRunner(app)
    await server.setup()
    await web.TCPSite(server, "127.0.0.1", 0).start()
    port = server.addresses[0][1]
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
    store = SQLExperimentExecutionStore(engine)
    root = tmp_path / "conversations"
    root.mkdir()

    def open_runtime():
        adapter = PiAdapter(root, terminal_backend="os-sandbox")
        runtime = ConversationRuntime(
            root,
            {"pi": adapter},
            default_harness_id="pi",
            experiment_plan_store=SQLExperimentPlanStore(engine),
            experiment_execution_store=store,
            execution_roles=("dataset", "init"),
            release_grace_seconds=1,
        )
        adapter.set_plan_handler(runtime.experiment_plan_call)
        adapter.set_role_handler(runtime.execution_role_call)
        return runtime, adapter

    runtime, adapter = open_runtime()
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=owner.conversation_id,
                user_id=owner.user_id,
                profile_id=profile.profile_id,
                workspace_root=str(root / owner.conversation_id),
                product_instructions=profile.instructions,
                tool_allowlist=profile.development_tool_allowlist,
                delegated_roles=tuple(
                    role
                    for role in profile.delegated_roles
                    if role.name in {"dataset", "init"}
                ),
                managed_delegation=True,
                model_configuration={
                    "model": "eval-test-model",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "local-test-key",
                },
            ),
            context,
        )
        executions = runtime._experiment_executions
        assert executions is not None
        row = executions.request(
            owner,
            RequestExperimentTask.model_validate(
                {
                    "role_id": "dataset",
                    "request_id": "prepare-data",
                    "task": {
                        "name": "Dataset",
                        "instruction": "Prepare training JSONL",
                    },
                }
            ),
            ("dataset",),
            execution_policy={"artifact_contract": "data"},
        )
        dataset_id = row["id"]
        report_path = f"public_data/experiment-runs/{dataset_id}/public_data/report.md"
        await runtime.start_execution_events()
        product = runtime._store(owner.conversation_id)
        async with asyncio.timeout(25):
            while not (permissions := product.load_snapshot().pending_permissions):
                await asyncio.sleep(0.05)
        assert await adapter.has_pending_session_work(
            runtime._active[owner.conversation_id].handle
        )
        if scenario == "busy":
            await runtime.submit_command(
                owner.conversation_id,
                UserMessageCommand(
                    command_id="foreground",
                    content=(TextContent(text="Keep working on the project"),),
                ),
                context,
            )
            await asyncio.wait_for(busy_started.wait(), 10)
        if scenario == "independent":
            for index in (1, 2):
                command_id = f"foreground-{index}"
                await runtime.submit_command(
                    owner.conversation_id,
                    UserMessageCommand(
                        command_id=command_id,
                        content=(TextContent(text=f"Manager message {index}"),),
                    ),
                    context,
                )
                async with asyncio.timeout(8):
                    while not any(
                        event.type == "status.changed"
                        and event.run_id == command_id
                        and event.payload.get("status") != "running"
                        for event in product.replay()
                    ):
                        await asyncio.sleep(0.05)
            assert len(foreground_replies) == 2
            assert "Manager message 2" in foreground_replies[1]
            assert product.load_snapshot().pending_permissions
        await runtime.submit_command(
            owner.conversation_id,
            PermissionResponseCommand(
                command_id="approval",
                permission_id=permissions[0].permission_id,
                decision="allow_once",
            ),
            context,
        )
        if scenario == "restart":
            await asyncio.wait_for(report_written.wait(), 15)
            await runtime.submit_command(
                owner.conversation_id,
                CancelCommand(command_id="pause-before-handoff"),
                context,
            )
            assert product.load_snapshot().automatic_turns_suspended
        async with asyncio.timeout(25):
            while not any(
                event.type in {"operation.completed", "operation.failed"}
                and event.payload.get("name") == "dataset"
                for event in product.replay()
            ):
                await asyncio.sleep(0.05)
        release_manager.set()
        if scenario == "restart":
            async with asyncio.timeout(10):
                while store.work_items(owner)[0]["state"] != "completed":
                    await asyncio.sleep(0.05)
            await runtime.close()
            release_dataset.set()
            runtime, _ = open_runtime()
            await runtime.start_execution_events()
            snapshot = await runtime.get_snapshot(owner.conversation_id, context)
            assert snapshot.automatic_turns_suspended
            assert not busy_started.is_set() and not manager_queries
            await runtime.submit_command(
                owner.conversation_id,
                UserMessageCommand(
                    command_id="resume",
                    content=(TextContent(text="Continue from saved dataset"),),
                ),
                context,
            )
        async with asyncio.timeout(25):
            while not init_inputs or any(
                item["state"] != "completed" for item in store.work_items(owner)
            ):
                await asyncio.sleep(0.05)
        assert len(manager_queries) == len(init_inputs) == 1
        assert artifact in init_inputs[0]
        manager_turns = [
            item for item in store.work_items(owner) if item["kind"] == "manager_turn"
        ]
        expected_manager_turns = {"idle": 0, "busy": 1, "restart": 1, "independent": 2}
        assert len(manager_turns) == expected_manager_turns[scenario]
        rows = store.list(owner)
        assert [(row["role_id"], row["state"]) for row in rows] == [
            ("dataset", "needs_input" if scenario == "restart" else "reported"),
            ("init", "needs_input"),
        ]
        if scenario == "restart":
            assert rows[0]["report_state"] == "failed"
        else:
            assert rows[0]["report_state"] == "ready"
            assert report_path in rows[0]["result"]["summary"]
        assert not product.load_snapshot().pending_permissions
        assert not any(
            event.type in {"message.started", "message.completed"}
            and event.payload.get("role") == "user"
            and "Inspect subagent updates above" in str(event.payload.get("content"))
            for event in product.replay()
        )
        with pytest.raises(PermissionError):
            await runtime.get_experiment_executions(
                owner.conversation_id,
                RequestContext(user_id=owner.user_id, profile_id="other"),
            )
    finally:
        release_manager.set()
        release_dataset.set()
        await runtime.close()
        await server.cleanup()
        engine.dispose()


@pytest.mark.parametrize(
    "restart,direct_question",
    [
        (False, False),
        ("platform", False),
        ("preparation", False),
        pytest.param(False, True, id="direct-child-question"),
    ],
)
async def test_native_eval_submits_then_resumes_for_report_and_notifies_manager(
    tmp_path, monkeypatch, restart, direct_question
):
    captured, submitted, manager_requests = [], [], []
    current = {}
    followups = []
    preparation = asyncio.Event()
    resume_preparation = asyncio.Event()
    native_waiting = []

    async def respond(request):
        payload = await request.json()
        captured.append(payload)
        names = {item["function"]["name"] for item in payload.get("tools", [])}
        assert "record_experiment_result" not in names
        child_tool = "df_submit_evaluation"
        if child_tool not in names:
            if "Baseline report:" not in json.dumps(payload["messages"]):
                native_waiting.append(payload)
                return stream(text="Platform result is pending.")
            manager_requests.append(payload)
            if not any(
                message.get("role") == "tool" for message in payload["messages"]
            ):
                return stream("read", {"path": current["report"]})
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
        if "FOLLOWUP_CONTEXT_CHECK" in messages:
            assert "df_submit_evaluation" in used
            assert "write" in used
            followups.append(payload)
            return stream(text="Original child context retained. No new submission.")
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
            return stream(text=f"Baseline report: `{current['report']}`.")
        if direct_question and "ask_user" not in used:
            assert "ask_user" in names
            return stream(
                "ask_user",
                {
                    "presentation": {
                        "title": "Child needs a decision",
                        "content": [{"type": "text", "text": "No sample exists yet."}],
                    },
                    "questions": [
                        {
                            "id": "format",
                            "prompt": "Use JSONL?",
                            "options": [{"id": "yes", "label": "Use JSONL"}],
                        }
                    ],
                },
            )
        if direct_question and not submitted:
            assert any(
                message.get("role") == "tool"
                and '"option_id":"yes"' in str(message.get("content"))
                for message in payload["messages"]
            )
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
    preparation_store = SQLPreparationStore(engine)
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
        preparation_store=preparation_store,
        execution_roles=("eval", "init"),
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
    profile = PREPARED_PROFILE
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
                preparation_policy=profile.preparation_policy,
                development_tool_allowlist=profile.development_tool_allowlist,
                development_skill_allowlist=profile.development_skill_allowlist,
                command_allowlist=profile.command_allowlist,
                delegated_roles=profile.delegated_roles,
                managed_delegation=True,
                user_input_enabled=direct_question,
                model_configuration={
                    "model": "eval-test-model",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "local-test-key",
                },
            ),
            context,
        )
        runtime._store("manager").append(
            ProductEvent(
                conversation_id="manager",
                type="message.completed",
                payload={
                    "message_id": "fixture",
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Evaluate one sample from /dataset.jsonl "
                            "with /models/base using exact labels",
                        }
                    ],
                },
            )
        )
        values = {
            "goal": "Evaluate exact label correctness",
            "model": {"source": "storage", "reference": "/models/base"},
            "dataset": {"path": "/dataset.jsonl", "format": "jsonl"},
            "io_contract": {"input": "prompt", "output": "label"},
            "scoring": {"basis": "exact label match"},
            "scope": {"mode": "sample", "limit": 1},
        }
        facts = {
            key: {
                "value": value,
                "status": "resolved",
                "evidence_refs": ["message:fixture"],
            }
            for key, value in values.items()
        }
        await runtime.experiment_plan_call(
            "manager",
            "update_preparation",
            UpdatePreparation.model_validate(
                {"expected_revision": 0, "facts": facts}
            ).model_dump(mode="json"),
            context,
        )
        entry = {
            "role_id": "init",
            "request_id": "baseline",
            "stage_id": "init",
            "task": {
                "name": "自由命名的基模评测",
                "instruction": "Run one sample baseline",
                "inputs": {
                    "model_path": "/workspace/models/base",
                    "input_path": "/workspace/dataset.jsonl",
                },
            },
        }
        guide = {
            "goal": "One sample baseline",
            "stages": [{"id": "init", "kind": "init", "name": "Init"}],
            "resources": [
                {
                    "id": "data",
                    "name": "Data",
                    "storage_path": "/workspace/dataset.jsonl",
                }
            ],
            "tasks": [
                {
                    "id": "baseline",
                    "kind": "bench",
                    "name": "Baseline",
                    "stage_id": "init",
                    "inputs": [{"name": "input_path", "resource_id": "data"}],
                }
            ],
        }
        ready = await runtime.experiment_plan_call(
            "manager",
            "update_preparation",
            UpdatePreparation.model_validate(
                {
                    "expected_revision": 1,
                    "facts": facts,
                    "plan": guide,
                    "entry_task": entry,
                }
            ).model_dump(mode="json"),
            context,
        )
        receipt = await runtime.submit_command(
            "manager",
            CreateProjectCommand(
                command_id="create",
                expected_preparation_revision=PreparationView.model_validate(
                    ready
                ).revision,
            ),
            context,
        )
        row = next(
            item
            for item in store.list(owner)
            if item["id"] == receipt.response["initial_run_id"]
        )
        current.update(
            id=row["id"],
            root=f"public_data/experiment-runs/{row['id']}",
            report=f"public_data/experiment-runs/{row['id']}/public_data/report.md",
        )
        await runtime.start_execution_events()
        if direct_question:
            async with asyncio.timeout(30):
                while not (
                    snapshot := runtime._store("manager").load_snapshot()
                ).pending_inputs:
                    await asyncio.sleep(0.05)
            pending = snapshot.pending_inputs[0]
            assert pending.task_run_id == row["id"]
            assert pending.presentation is not None
            assert pending.presentation.title == "Child needs a decision"
            assert pending.checkpoint is None and submitted == []
            guidance = await runtime.submit_command(
                "manager",
                ControlExperimentTaskCommand(
                    command_id="guide-running-child",
                    run_id=row["id"],
                    action="continue",
                    message="LIVE_CONTEXT_GUIDANCE: keep the existing scope "
                    "and report actual evidence.",
                ),
                context,
            )
            assert guidance.status == "completed"
            async with asyncio.timeout(15):
                while not any(
                    item["kind"] == "steer_task" and item["state"] == "completed"
                    for item in store.work_items(owner)
                ):
                    await asyncio.sleep(0.05)
            assert len(runtime._store("manager").load_snapshot().pending_inputs) == 1
            receipt = await runtime.submit_command(
                "manager",
                InputResponseCommand.model_validate(
                    {
                        "command_id": "answer-child",
                        "request_id": pending.request_id,
                        "status": "answered",
                        "answers": [{"question_id": "format", "option_id": "yes"}],
                    }
                ),
                context,
            )
            assert receipt.status == "completed"
            assert not runtime._store("manager").load_snapshot().pending_inputs
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
                preparation_store=preparation_store,
                execution_roles=("eval", "init"),
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
                    and all(item["state"] == "completed" for item in queue)
                    and len(manager_requests) >= 2
                ):
                    break
                await asyncio.sleep(0.1)
        assert len(submitted) == 1 and submitted[0][0] == "df_submit_evaluation"
        assert row["id"] in submitted[0][2]
        assert (
            view.state == "succeeded"
            and view.metrics == metrics
            and view.report_uri is None
        )
        assert current["report"] in (view.summary or "")
        assert plans.view(owner).current is None
        assert len(manager_requests) == 2
        assert all(item["kind"] != "manager_turn" for item in queue)
        assert len(native_waiting) <= 1
        assert "One actual sample was scored" in json.dumps(
            manager_requests[-1]["messages"]
        )
        if not restart:
            async with asyncio.timeout(10):
                while True:
                    projected = (
                        await runtime.get_snapshot("manager", context)
                    ).external_tasks
                    if (
                        len(projected) == 1
                        and projected[0].status == "succeeded"
                        and not projected[0].resume_pending
                    ):
                        break
                    await asyncio.sleep(0.05)
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
        assert "request_experiment_task" in manager_tools
        assert "df_submit_evaluation" not in manager_tools
        assert {"read", "write", "edit", "terminal", "grep", "find", "ls"} <= (
            manager_tools
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
        binding = first_display["execution"]
        assert isinstance(binding, dict) and binding["task_run_id"] == row["id"]
        assert report_display["execution"] == first_display["execution"]
        assert all(
            child.execution is not None and child.execution["stage_id"] == "init"
            for child in children.values()
        )
        assert first_display["phase"] == "execute"
        assert report_display["phase"] == "report"
        assert report_display["previous_run_id"] == first_ref
        if direct_question:
            original_ref = store.native_reference(owner, row["id"])
            receipt = await runtime.submit_command(
                "manager",
                ControlExperimentTaskCommand(
                    command_id="followup-completed",
                    run_id=row["id"],
                    action="continue",
                    message="FOLLOWUP_CONTEXT_CHECK: explain the completed result "
                    "using your previous context.",
                ),
                context,
            )
            assert receipt.status == "completed"
            async with asyncio.timeout(30):
                while (
                    not followups
                    or any(
                        item["state"] != "completed" for item in store.work_items(owner)
                    )
                    or not any(
                        "Original child context retained"
                        in json.dumps(payload["messages"])
                        for payload in manager_requests
                    )
                ):
                    await asyncio.sleep(0.05)
            assert len(submitted) == 1 and len(followups) == 1
            assert "LIVE_CONTEXT_GUIDANCE" in json.dumps(followups[0]["messages"])
            assert any(
                "Original child context retained" in json.dumps(payload["messages"])
                for payload in manager_requests
            )
            assert store.native_reference(owner, row["id"]) != original_ref
            current_view = (
                await runtime.get_experiment_executions("manager", context)
            )[0]
            assert (
                current_view.state == "succeeded"
                and current_view.report_state == "ready"
            )
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
            {
                "id": "bench",
                "name": "Evaluation data",
                "storage_path": "/workspace/datasets/val.jsonl",
            }
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
        execution_roles=("eval", "init"),
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
