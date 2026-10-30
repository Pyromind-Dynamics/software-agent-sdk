from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from pydantic import ValidationError
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.experiment_plan_projects import (
    EchoMindExperimentPlanProjectBinder,
)
from pyromind_agent_server.profiles import (
    DEFAULT_PRODUCT_PROFILE,
    ECHOMIND_PRODUCT_PROFILE,
)
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.experiment_plans import ExperimentPlans
from pyromind_runtime.application.product_instructions import DATA_FLYWHEEL_INSTRUCTIONS
from pyromind_runtime.domain.commands import ConfirmExperimentPlanCommand
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_plans import (
    CreateExperimentPlan,
    PlanConflictError,
    PlanOwner,
    ReviseExperimentPlan,
    WorkflowPlan,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
    SQLExperimentPlanStore,
    metadata,
    projects,
    proposals,
    revisions,
)
from pyromind_runtime.ports.harness import SessionSpec
from sqlalchemy import create_engine, delete, event, func, select

from openhands.agent_server.echomind_projects import ProjectStore

from .fake_adapter import FakeAdapter
from .test_product_api import _app


@pytest.fixture
def plan_payload():
    return {
        "name": "图片质检实验",
        "goal": "减少漏检，同时观察误检变化",
        "resources": [{"id": "bench", "name": "固定评测集", "uri": "storage://bench"}],
        "unresolved": ["训练参数待补充"],
        "stages": [
            {"id": key, "name": name}
            for key, name in (
                ("init", "Init"),
                ("data", "Data"),
                ("train", "Train"),
                ("eval", "Eval"),
            )
        ]
        + [
            {"id": key, "name": name, "enabled": False, "placeholder": True}
            for key, name in (("inference", "Inference"), ("tracer", "Tracer"))
        ],
        "edges": [
            {"source": a, "target": b}
            for a, b in (("init", "data"), ("data", "train"), ("train", "eval"))
        ],
        "tasks": [
            {
                "id": "baseline",
                "stage_id": "init",
                "name": "基模自由名称 bench",
                "inputs": [{"name": "评测集", "resource_id": "bench"}],
                "outputs": [{"name": "baseline_report"}],
            },
            {
                "id": "analysis",
                "stage_id": "data",
                "name": "根据基线探查数据",
                "inputs": [
                    {
                        "name": "基线报告",
                        "task_id": "baseline",
                        "output_name": "baseline_report",
                    }
                ],
            },
        ],
    }


@pytest.fixture
def plan_store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'plans.sqlite'}")
    metadata.create_all(engine)
    store = SQLExperimentPlanStore(engine)
    yield store
    engine.dispose()


@pytest.fixture
def owner():
    return PlanOwner(
        user_id="42", conversation_id="conversation", profile_id="planning"
    )


def test_plan_graph_and_expected_outputs(plan_payload):
    plan = WorkflowPlan.model_validate(plan_payload)
    assert plan.stages[0].name == "Init"
    assert plan.tasks[0].name == "基模自由名称 bench"
    assert plan.tasks[1].inputs[0].output_name == "baseline_report"


def test_evaluation_only_plan_does_not_require_training_parameters(plan_payload):
    plan_payload["goal"] = "只做评测"
    plan_payload["unresolved"] = []
    plan_payload["edges"] = []
    for stage in plan_payload["stages"]:
        stage["enabled"] = stage["id"] == "eval"
    plan_payload["tasks"] = [
        {
            "id": "custom-evaluation",
            "stage_id": "eval",
            "name": "检查基模的漏检情况",
            "inputs": [{"name": "评测集", "resource_id": "bench"}],
        }
    ]
    plan = WorkflowPlan.model_validate(plan_payload)
    assert [stage.name for stage in plan.stages if stage.enabled] == ["Eval"]
    assert plan.unresolved == ()


@pytest.mark.parametrize(
    "problem",
    ["duplicate", "dangling", "cycle", "input_cycle", "output", "placeholder"],
)
def test_invalid_plan_rejected(plan_payload, problem):
    if problem == "duplicate":
        plan_payload["tasks"].append(plan_payload["tasks"][0])
    elif problem == "dangling":
        plan_payload["edges"].append({"source": "missing", "target": "data"})
    elif problem == "cycle":
        plan_payload["edges"].append({"source": "eval", "target": "init"})
    elif problem == "input_cycle":
        plan_payload["tasks"][1]["outputs"] = [{"name": "data"}]
        plan_payload["tasks"][0]["inputs"] = [
            {"name": "data", "task_id": "analysis", "output_name": "data"}
        ]
    elif problem == "output":
        plan_payload["tasks"][1]["inputs"][0]["output_name"] = "missing"
    else:
        plan_payload["stages"][-1]["enabled"] = True
    with pytest.raises(ValidationError):
        WorkflowPlan.model_validate(plan_payload)


def test_proposal_confirmation_revision_rejection_and_isolation(
    plan_store, owner, plan_payload
):
    service = ExperimentPlans(plan_store)
    draft = service.propose(
        owner, CreateExperimentPlan(plan=WorkflowPlan.model_validate(plan_payload))
    )
    view = plan_store.view(owner)
    assert view.project_id is None and view.pending_proposal == draft
    assert (
        plan_store.view(
            owner.model_copy(update={"profile_id": "other"})
        ).pending_proposal
        is None
    )
    revision = plan_store.confirm(owner, draft.proposal_id, 0, "confirm-1")
    assert revision.revision == 1
    assert plan_store.confirm(owner, draft.proposal_id, 0, "confirm-1") == revision
    assert plan_store.confirm(owner, draft.proposal_id, 0, "double-click") == revision
    updated = draft.plan.model_copy(update={"goal": "只做评测"})
    second = service.propose(
        owner, ReviseExperimentPlan(plan=updated, expected_revision=1)
    )
    assert plan_store.view(owner).current == revision
    third = service.propose(
        owner,
        ReviseExperimentPlan(
            plan=updated, expected_revision=1, source_proposal_id=second.proposal_id
        ),
    )
    with pytest.raises(PlanConflictError, match="pending"):
        plan_store.confirm(owner, second.proposal_id, 1, "stale")
    saved = plan_store.confirm(owner, third.proposal_id, 1, "confirm-2")
    assert saved.revision == 2 and saved.project_id == revision.project_id
    assert plan_store.revision(owner, 1) == revision
    fourth = service.propose(
        owner, ReviseExperimentPlan(plan=draft.plan, expected_revision=2)
    )
    plan_store.reject(owner, fourth.proposal_id)
    plan_store.reject(owner, fourth.proposal_id)
    with pytest.raises(PlanConflictError):
        plan_store.confirm(owner, fourth.proposal_id, 2, "rejected")
    assert plan_store.view(owner).current == saved
    with pytest.raises(PlanConflictError):
        service.propose(owner, ReviseExperimentPlan(plan=updated, expected_revision=1))


def test_sql_failure_rolls_back_first_confirmation(plan_store, owner, plan_payload):
    draft = ExperimentPlans(plan_store).propose(
        owner, CreateExperimentPlan(plan=WorkflowPlan.model_validate(plan_payload))
    )

    def fail_revision(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("INSERT INTO experiment_plan_revisions"):
            raise RuntimeError("database failure")

    event.listen(plan_store.engine, "before_cursor_execute", fail_revision)
    try:
        with pytest.raises(RuntimeError, match="database failure"):
            plan_store.confirm(owner, draft.proposal_id, 0, "confirmation")
    finally:
        event.remove(plan_store.engine, "before_cursor_execute", fail_revision)
    view = plan_store.view(owner)
    assert view.project_id is None and view.pending_proposal == draft
    assert plan_store.confirm(owner, draft.proposal_id, 0, "confirmation").revision == 1


async def test_runtime_plans_restore_without_harness_and_refresh_hint_failure(
    tmp_path, plan_store, plan_payload, monkeypatch
):
    context = RequestContext(
        user_id="42",
        profile_id="planning",
        cookie="private-cookie",
        authorization="private-token",
    )
    adapter = FakeAdapter("pi")
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        {"pi": adapter},
        default_harness_id="pi",
        experiment_plan_store=plan_store,
    )
    spec = SessionSpec(
        conversation_id="plan-session",
        user_id="42",
        profile_id="planning",
        workspace_root=str(tmp_path / "conversations" / "plan-session"),
        tool_allowlist=(
            "create_experiment_plan",
            "revise_experiment_plan",
            "get_experiment_plan",
        ),
        command_allowlist=(
            "user_message",
            "confirm_experiment_plan",
            "reject_experiment_plan",
        ),
    )
    await runtime.create_conversation(spec, context)
    draft = await runtime.experiment_plan_call(
        "plan-session", "create_experiment_plan", {"plan": plan_payload}, context
    )
    with pytest.raises(PermissionError):
        await runtime.get_snapshot("plan-session", replace(context, profile_id=None))
    original_append = FileProductStore.append

    def fail_notification(store, event):
        if event.type == "experiment_plan.changed":
            raise OSError("event delivery failed")
        return original_append(store, event)

    monkeypatch.setattr(FileProductStore, "append", fail_notification)
    command = ConfirmExperimentPlanCommand(
        command_id="confirm", proposal_id=str(draft["proposal_id"]), expected_revision=0
    )
    saved = await runtime.submit_command("plan-session", command, context)
    assert saved.status == "completed" and saved.response["revision"] == 1
    assert await runtime.submit_command("plan-session", command, context) == saved
    await runtime.close()
    restarted_adapter = FakeAdapter("pi")
    restarted = ConversationRuntime(
        tmp_path / "conversations",
        {"pi": restarted_adapter},
        default_harness_id="pi",
        experiment_plan_store=plan_store,
    )
    snapshot = await restarted.get_snapshot("plan-session", context)
    assert snapshot.experiment_plan is not None
    assert snapshot.experiment_plan.current_revision == 1
    assert restarted_adapter.attached == []
    for file in (tmp_path / "conversations" / "plan-session").rglob("*"):
        if file.is_file():
            assert (
                "private-cookie" not in file.read_text()
                and "private-token" not in file.read_text()
            )
    await restarted.close()


async def test_http_profile_routing_confirm_conflict_and_cross_profile_denial(
    tmp_path, plan_store, plan_payload
):
    adapter, old_adapter = FakeAdapter("pi"), FakeAdapter("openhands")
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        {"pi": adapter, "openhands": old_adapter},
        experiment_plan_store=plan_store,
    )
    app = _app(tmp_path, runtime)
    app.include_router(create_product_router(ECHOMIND_PRODUCT_PROFILE))
    headers = {"x-pyromind-debug-user-id": "42"}
    base = "/api/v2/echomind/conversations"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers
    ) as client:
        created = await client.post(base, json={"llm": {"model": "test"}})
        assert created.status_code == 201
        identity = created.json()["conversation_id"]
        context = RequestContext(
            user_id="42", profile_id=ECHOMIND_PRODUCT_PROFILE.profile_id
        )
        assert "EchoMind Manager" in adapter.created_specs[0].product_instructions
        assert (
            DATA_FLYWHEEL_INSTRUCTIONS
            not in adapter.created_specs[0].product_instructions
        )
        assert adapter.created_specs[0].tool_allowlist is not None
        assert "subagent" not in adapter.created_specs[0].tool_allowlist
        draft = await runtime.experiment_plan_call(
            identity, "create_experiment_plan", {"plan": plan_payload}, context
        )
        command_body = {
            "type": "confirm_experiment_plan",
            "command_id": "confirm",
            "proposal_id": draft["proposal_id"],
            "expected_revision": 0,
        }
        saved = await client.post(f"{base}/{identity}/commands", json=command_body)
        assert saved.status_code == 202
        assert (
            await client.post(f"{base}/{identity}/commands", json=command_body)
        ).json() == saved.json()
        snapshot = await client.get(f"{base}/{identity}/snapshot")
        assert snapshot.json()["experiment_plan"]["current_revision"] == 1
        assert (
            await client.get(f"{base}/{identity}/experiment-plans?revision=1")
        ).json()["revision"] == 1
        for suffix in ("snapshot", "events", "experiment-plans"):
            assert (
                await client.get(f"/api/v2/pyromind/conversations/{identity}/{suffix}")
            ).status_code == 404
        assert (
            await client.get(
                f"{base}/{identity}/snapshot",
                headers={"x-pyromind-debug-user-id": "other"},
            )
        ).status_code == 404
        assert (await client.get("/api/v2/pyromind/conversations")).json() == []
        assert len((await client.get(base)).json()) == 1
        forbidden = await client.post(
            f"{base}/{identity}/commands",
            json={
                "type": "rollback_workflow",
                "command_id": "no",
                "event_id": "not-allowed",
            },
        )
        assert forbidden.status_code == 404
        prior = await client.post(
            "/api/v2/pyromind/conversations", json={"llm": {"model": "test"}}
        )
        assert prior.status_code == 201
        assert (
            old_adapter.created_specs[0].product_instructions
            == DEFAULT_PRODUCT_PROFILE.instructions
        )
        assert (
            DATA_FLYWHEEL_INSTRUCTIONS
            in old_adapter.created_specs[0].product_instructions
        )
        assert (
            await client.get(f"{base}/{prior.json()['conversation_id']}/snapshot")
        ).status_code == 404
    await runtime.close()


def test_plan_migration_matches_storage_schema(tmp_path):
    from pyromind_agent_server import plan_migrations

    config = AlembicConfig()
    config.set_main_option("script_location", str(next(iter(plan_migrations.__path__))))
    config.attributes["database_url"] = f"sqlite:///{tmp_path / 'migrated.sqlite'}"
    command.upgrade(config, "head")
    command.upgrade(config, "head")
    engine = create_engine(config.attributes["database_url"])
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection, opts={"version_table": "experiment_plan_schema_version"}
        )
        assert compare_metadata(context, metadata) == []
    engine.dispose()


@pytest.mark.skipif(
    not os.getenv("EXPERIMENT_PLAN_TEST_DATABASE_URL"),
    reason="requires a dedicated MariaDB test database",
)
def test_mariadb_concurrent_first_confirmation_and_recovery(plan_payload):
    url = os.environ["EXPERIMENT_PLAN_TEST_DATABASE_URL"]
    from pyromind_agent_server import plan_migrations

    config = AlembicConfig()
    config.set_main_option("script_location", str(next(iter(plan_migrations.__path__))))
    config.attributes["database_url"] = url
    command.upgrade(config, "head")
    engines = [create_engine(url, hide_parameters=True) for _ in range(2)]
    stores = [SQLExperimentPlanStore(engine) for engine in engines]
    owner = PlanOwner(
        user_id="integration-test", conversation_id=uuid4().hex, profile_id="planning"
    )
    try:
        draft = ExperimentPlans(stores[0]).propose(
            owner, CreateExperimentPlan(plan=WorkflowPlan.model_validate(plan_payload))
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    store.confirm, owner, draft.proposal_id, 0, "same-confirmation"
                )
                for store in stores
            ]
            results = [future.result(timeout=30) for future in futures]
        assert results[0] == results[1]
        view = SQLExperimentPlanStore(engines[1]).view(owner)
        assert view.current_revision == 1 and view.pending_proposal is None
        with engines[0].connect() as connection:
            count = connection.scalar(
                select(func.count())
                .select_from(revisions)
                .where(revisions.c.project_id == view.project_id)
            )
            assert count == 1
        updated = draft.plan.model_copy(update={"goal": "验证修改后的计划"})
        next_draft = ExperimentPlans(stores[0]).propose(
            owner, ReviseExperimentPlan(plan=updated, expected_revision=1)
        )
        with pytest.raises(PlanConflictError):
            stores[1].confirm(owner, next_draft.proposal_id, 0, "stale")

        def fail_revision(connection, cursor, statement, parameters, context, many):
            if statement.startswith("INSERT INTO experiment_plan_revisions"):
                raise RuntimeError("injected database failure")

        event.listen(engines[0], "before_cursor_execute", fail_revision)
        try:
            with pytest.raises(RuntimeError, match="injected database failure"):
                stores[0].confirm(owner, next_draft.proposal_id, 1, "confirm-next")
        finally:
            event.remove(engines[0], "before_cursor_execute", fail_revision)
        assert stores[1].view(owner).current_revision == 1
        assert stores[1].view(owner).pending_proposal is not None
        assert (
            stores[1].confirm(owner, next_draft.proposal_id, 1, "confirm-next").revision
            == 2
        )
        assert stores[0].revision(owner, 1) == results[0]
    finally:
        with engines[0].begin() as connection:
            project_id = connection.scalar(
                select(projects.c.project_id).where(
                    projects.c.conversation_id == owner.conversation_id
                )
            )
            if project_id:
                connection.execute(
                    delete(revisions).where(revisions.c.project_id == project_id)
                )
                connection.execute(
                    delete(projects).where(projects.c.project_id == project_id)
                )
            from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
                _owner_key,
            )

            connection.execute(
                delete(proposals).where(proposals.c.owner_key == _owner_key(owner))
            )
        for engine in engines:
            engine.dispose()


async def test_plan_tools_through_real_pi_and_restored_session(
    tmp_path, plan_store, plan_payload, monkeypatch
):
    from pathlib import Path

    from aiohttp import web
    from harness_adapter.pi_adapter import PiAdapter
    from pyromind_runtime.domain.commands import UserMessageCommand
    from pyromind_runtime.domain.content import TextContent

    requests = []
    updated = {**plan_payload, "goal": "只做评测", "unresolved": []}

    async def respond(request):
        payload = await request.json()
        requests.append(payload)
        if payload["messages"][-1]["role"] == "tool":
            delta, reason = {"content": "计划提案已生成，请在面板确认。"}, "stop"
        else:
            modifying = len(requests) > 2
            arguments = {"plan": updated if modifying else plan_payload}
            if modifying:
                arguments["expected_revision"] = 1
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"plan-{len(requests)}",
                        "type": "function",
                        "function": {
                            "name": "revise_experiment_plan"
                            if modifying
                            else "create_experiment_plan",
                            "arguments": json.dumps(arguments),
                        },
                    }
                ]
            }
            reason = "tool_calls"
        chunks = [
            {
                "id": "local",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local-plan-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", **delta},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "local",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "local-plan-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            },
        ]
        return web.Response(
            text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            + "data: [DONE]\n\n",
            content_type="text/event-stream",
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    monkeypatch.setenv(
        "PYROMIND_PI_RUNTIME",
        str(
            Path(__file__).resolve().parents[2]
            / "harness-adapter/pi-runtime/dist/index.js"
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    root = tmp_path / "conversations"
    root.mkdir()
    profile = ECHOMIND_PRODUCT_PROFILE
    context = RequestContext(user_id="42", profile_id=profile.profile_id)

    def build_runtime():
        adapter = PiAdapter(root, terminal_backend="os-sandbox")
        runtime = ConversationRuntime(
            root,
            {"pi": adapter},
            default_harness_id="pi",
            experiment_plan_store=plan_store,
        )
        adapter.set_plan_handler(runtime.experiment_plan_call)
        return runtime, adapter

    async def wait_draft(runtime, base):
        async with asyncio.timeout(30):
            while True:
                snapshot = await runtime.get_snapshot("manager", context)
                proposal = snapshot.experiment_plan.pending_proposal
                if (
                    proposal
                    and proposal.base_revision == base
                    and snapshot.status == "idle"
                ):
                    return proposal
                if snapshot.status == "error":
                    pytest.fail(str(snapshot.timeline))
                await asyncio.sleep(0.05)

    runtime, adapter = build_runtime()
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
                development_tool_allowlist=profile.development_tool_allowlist,
                development_skill_allowlist=profile.development_skill_allowlist,
                command_allowlist=profile.command_allowlist,
                model_configuration={
                    "model": "local-plan-model",
                    "base_url": f"http://127.0.0.1:{port}/v1",
                    "api_key": "local-test-key",
                },
            ),
            context,
        )
        await runtime.submit_command(
            "manager",
            UserMessageCommand(
                command_id="start", content=(TextContent(text="设计实验计划"),)
            ),
            context,
        )
        proposal = await wait_draft(runtime, 0)
        tools = {tool["function"]["name"] for tool in requests[0]["tools"]}
        assert tools == {
            "read",
            "terminal",
            "create_experiment_plan",
            "revise_experiment_plan",
            "get_experiment_plan",
        }
        prompt = json.dumps(requests[0]["messages"], ensure_ascii=False)
        assert "experiment-planning" in prompt and "generate-workflow-dsl" not in prompt
        assert "For Pyromind workflow requests" not in prompt
        await runtime.submit_command(
            "manager",
            ConfirmExperimentPlanCommand(
                command_id="confirm",
                proposal_id=proposal.proposal_id,
                expected_revision=0,
            ),
            context,
        )
        request_count = len(requests)
        await runtime.submit_command(
            "manager",
            UserMessageCommand(
                command_id="revise",
                content=(TextContent(text="修改实验目标为只做评测"),),
            ),
            context,
        )
        proposal = await wait_draft(runtime, 1)
        assert proposal.plan.goal == "只做评测"
        updated_tools = {
            tool["function"]["name"] for tool in requests[request_count]["tools"]
        }
        assert {
            "read",
            "write",
            "edit",
            "terminal",
            "validate_workflow_dsl",
            "df_submit_pipeline",
            "sandbox_create",
            "sandbox_terminal",
        }.issubset(updated_tools)
        assert updated_tools.issubset(set(profile.development_tool_allowlist or ()))
        await runtime.close()
        runtime, _ = build_runtime()
        assert profile.tool_allowlist is not None
        persisted_config = json.loads(
            (root / "manager" / "pi" / "session.json").read_text(encoding="utf-8")
        )
        assert persisted_config["tool_allowlist"] == list(
            profile.development_tool_allowlist or ()
        )
        assert persisted_config["skill_allowlist"] == list(
            profile.development_skill_allowlist or ()
        )
        snapshot = await runtime.get_snapshot("manager", context)
        assert snapshot.experiment_plan is not None
        assert snapshot.experiment_plan.current_revision == 1
    finally:
        await runtime.close()
        await runner.cleanup()


async def test_plan_confirmation_binds_project_and_activates_development_policy(
    tmp_path, plan_store
):
    conversations = tmp_path / "conversations"
    conversations.mkdir()
    adapter = FakeAdapter("pi")
    binder = EchoMindExperimentPlanProjectBinder(conversations)
    runtime = ConversationRuntime(
        conversations,
        {"pi": adapter},
        default_harness_id="pi",
        experiment_plan_store=plan_store,
        experiment_plan_project_binder=binder,
    )
    profile = ECHOMIND_PRODUCT_PROFILE
    context = RequestContext(user_id="42", profile_id=profile.profile_id)
    plan = {
        "skill": "experiment-planning",
        "name": "AVI project",
        "goal": "Build the four-stage graph",
        "stages": [
            {"id": "dataset", "name": "Dataset"},
            {"id": "inference", "name": "Inference"},
            {"id": "evaluator", "name": "Evaluator"},
            {"id": "train", "name": "Train"},
        ],
        "edges": [
            {"source": "dataset", "target": "inference"},
            {"source": "inference", "target": "evaluator"},
            {"source": "evaluator", "target": "train"},
        ],
    }
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="manager-project",
                user_id="42",
                profile_id=profile.profile_id,
                workspace_root=str(conversations / "manager-project"),
                product_instructions=profile.instructions,
                tool_allowlist=profile.tool_allowlist,
                skill_allowlist=profile.skill_allowlist,
                development_tool_allowlist=profile.development_tool_allowlist,
                development_skill_allowlist=profile.development_skill_allowlist,
                command_allowlist=profile.command_allowlist,
            ),
            context,
        )
        draft = await runtime.experiment_plan_call(
            "manager-project",
            "create_experiment_plan",
            {"plan": plan},
            context,
        )
        command = ConfirmExperimentPlanCommand(
            command_id="confirm-project",
            proposal_id=str(draft["proposal_id"]),
            expected_revision=0,
        )
        receipt = await runtime.submit_command("manager-project", command, context)
        revision = await runtime.get_experiment_plan("manager-project", context)
        assert receipt.status == "completed"
        assert revision.project_id is not None
        project = ProjectStore.for_conversations_dir(conversations).load(
            revision.project_id, owner_user_id="42"
        )
        assert project.graph.template == "experiment-planning/v1"
        assert [node.id for node in project.graph.nodes] == [
            "dataset",
            "inference",
            "evaluator",
            "train",
        ]
        activated = adapter.session_policies[-1][1]
        assert activated.tool_allowlist == profile.development_tool_allowlist
        assert activated.skill_allowlist == profile.development_skill_allowlist

        assert (
            await runtime.submit_command("manager-project", command, context) == receipt
        )
        assert len(list((tmp_path / "projects").iterdir())) == 1
    finally:
        await runtime.close()
