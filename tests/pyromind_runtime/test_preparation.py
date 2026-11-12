from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import pytest
import sqlalchemy as sa
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.preparation import Preparations
from pyromind_runtime.domain.commands import CreateProjectCommand
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_plans import PlanConflictError, PlanOwner
from pyromind_runtime.domain.preparation import (
    PREPARATION_TOOLS,
    PreparationPolicy,
    PreparationState,
    PreparationView,
    SetPreparationPlan,
)
from pyromind_runtime.infrastructure.sql_preparation import (
    SQLPreparationStore,
    preparations,
)
from pyromind_runtime.infrastructure.sql_project_store import metadata, projects
from pyromind_runtime.ports.harness import DelegatedRole, SessionSpec
from sqlalchemy import event

from .fake_adapter import FakeAdapter
from .test_product_api import _app


@pytest.fixture
def preparation_setup(tmp_path):
    engine = sa.create_engine(
        f"sqlite:///{tmp_path / 'preparation.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    metadata.create_all(engine)
    store = SQLPreparationStore(engine)
    owner = PlanOwner(
        user_id="42", conversation_id="conversation", profile_id="prepared"
    )
    policy = PreparationPolicy.model_validate(
        {
            "allowed_roles": ["eval"],
            "workflow_nodes": [
                {
                    "id": "measure",
                    "name": "Measure",
                    "roles": ["eval"],
                    "max_instances": 1,
                    "tasks": [{"id": "score"}],
                },
                {"id": "future", "name": "Future"},
            ],
        }
    )
    store.initialize(owner)
    yield store, owner, policy
    engine.dispose()


@pytest.fixture
def ready_request():
    return SetPreparationPlan.model_validate(
        {
            "entry_task_id": "first",
            "plan": {
                "goal": "Evaluate model with the confirmed rubric",
                "constraints": ["Use the held-out samples, not training labels."],
                "resources": [
                    {
                        "id": "data",
                        "name": "Data",
                        "storage_path": "/workspace/eval.jsonl",
                    }
                ],
                "stages": [{"id": "evaluation", "kind": "measure", "name": "Eval"}],
                "tasks": [
                    {
                        "id": "first",
                        "kind": "score",
                        "name": "Evaluate",
                        "description": "Evaluate data and deliver a report.",
                        "stage_id": "evaluation",
                        "inputs": [{"name": "dataset", "resource_id": "data"}],
                    }
                ],
            },
        }
    )


def test_only_plan_submission_is_exposed():
    assert set(PREPARATION_TOOLS) == {"set_preparation_plan"}
    schema = SetPreparationPlan.model_json_schema()
    assert set(schema["properties"]) == {"plan", "entry_task_id"}
    assert {"goal", "stages", "tasks"} <= set(
        schema["$defs"]["ExperimentContext"]["required"]
    )
    assert {"kind", "stage_id"} <= set(schema["$defs"]["ExperimentTask"]["required"])


def test_submit_plan_exposes_review_state_without_checklist(
    preparation_setup, ready_request
):
    store, owner, policy = preparation_setup
    before = PreparationView.from_state(store.get(owner), policy)
    assert before.phase == "collecting"
    ready = Preparations(store).set_plan(owner, ready_request, policy, 0)
    assert ready.phase == "ready" and ready.entry_task is not None
    assert ready.entry_task.role_id == "eval"
    assert ready.entry_task.task.inputs == {"dataset": "/workspace/eval.jsonl"}
    view = PreparationView.from_state(ready, policy)
    assert view.phase == "ready" and not view.review_pending
    assert view.policy == policy
    assert not {"facts", "evidence", "missing"} & view.model_dump().keys()
    assert store.project(owner) is None
    with store.engine.connect() as conn:
        document = conn.scalar(sa.select(preparations.c.document))
        assert not {"facts", "evidence", "missing", "policy"} & document.keys()
        assert conn.scalar(sa.select(sa.func.count()).select_from(projects)) == 0


def test_new_message_withdraws_ready_and_rejects_stale_context_and_button(
    preparation_setup, ready_request
):
    store, owner, policy = preparation_setup
    ready = Preparations(store).set_plan(owner, ready_request, policy, 0)
    store.invalidate(owner, "user-edit")
    current = store.get(owner)
    assert current.review_pending and current.phase == "planning"
    assert PreparationView.from_state(current, policy).review_pending
    store.invalidate(owner, "user-edit")
    assert store.get(owner) == current
    with pytest.raises(PlanConflictError):
        store.activate(owner, ready.revision, "old-button")
    with pytest.raises(PlanConflictError):
        Preparations(store).set_plan(owner, ready_request, policy, ready.revision)
    refreshed = Preparations(store).set_plan(
        owner, ready_request, policy, current.revision
    )
    assert not refreshed.review_pending and refreshed.phase == "ready"


def test_project_creation_is_atomic_deduplicated_and_isolated(
    preparation_setup, ready_request
):
    store, owner, policy = preparation_setup
    ready = Preparations(store).set_plan(owner, ready_request, policy, 0)

    def create(index):
        return SQLPreparationStore(store.engine).activate(
            owner, ready.revision, f"button-{index}", cluster="us-west-1#pre"
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(create, range(3)))
    assert len({result.project_id for result in results}) == 1
    with store.engine.connect() as conn:
        assert conn.scalar(sa.select(sa.func.count()).select_from(projects)) == 1
        assert conn.scalar(sa.select(projects.c.cluster)) == "us-west-1#pre"
        assert not {"experiment_task_runs", "experiment_work"} & set(
            sa.inspect(conn).get_table_names()
        )
    for key in ("user_id", "profile_id", "conversation_id"):
        foreign = owner.model_copy(update={key: "foreign"})
        with pytest.raises(PermissionError):
            store.get(foreign)
        assert store.project(foreign) is None
    with pytest.raises(ValueError, match="activated"):
        Preparations(store).set_plan(owner, ready_request, policy, ready.revision)


def test_project_creation_failure_rolls_back_without_losing_review(
    preparation_setup, ready_request
):
    store, owner, policy = preparation_setup
    ready = Preparations(store).set_plan(owner, ready_request, policy, 0)

    def fail(_conn, _cursor, statement, _parameters, _context, _executemany):
        if statement.startswith("UPDATE experiment_projects"):
            raise RuntimeError("creation metadata write failed")

    event.listen(store.engine, "before_cursor_execute", fail)
    try:
        with pytest.raises(RuntimeError, match="metadata write failed"):
            store.activate(owner, ready.revision, "create")
    finally:
        event.remove(store.engine, "before_cursor_execute", fail)
    assert store.project(owner) is None
    assert store.get(owner) == ready


@pytest.mark.parametrize(
    "change,error",
    [
        ({"entry_task_id": "unknown"}, "entry_task_id"),
        ({"goal": ""}, "goal"),
        ({"stage_kind": "unknown"}, r"stages\[0\].kind"),
        ({"stage_kind": "future"}, "unimplemented"),
        ({"task_kind": "unknown"}, "allowed task kinds: score"),
        ({"stage_id": "unknown"}, "unknown stage"),
        ({"inputs": []}, "concrete inputs"),
        (
            {"inputs": [{"name": "data", "resource_id": "unknown"}]},
            "unknown resource",
        ),
        ({"storage_path": None}, "concrete inputs"),
        ({"disabled": True}, "enabled plan stage"),
        ({"credential": True}, "credentials cannot be persisted"),
    ],
)
def test_invalid_plan_does_not_save_or_create(
    preparation_setup, ready_request, change, error
):
    store, owner, policy = preparation_setup
    arguments = ready_request.model_dump(mode="json")
    plan = arguments["plan"]
    if "entry_task_id" in change:
        arguments["entry_task_id"] = change["entry_task_id"]
    for key in ("goal",):
        if key in change:
            plan[key] = change[key]
    if "stage_kind" in change:
        plan["stages"][0]["kind"] = change["stage_kind"]
    if change.get("disabled"):
        plan["stages"][0]["enabled"] = False
    for source, target in (
        ("task_kind", "kind"),
        ("stage_id", "stage_id"),
        ("inputs", "inputs"),
    ):
        if source in change:
            plan["tasks"][0][target] = change[source]
    if "storage_path" in change:
        plan["resources"][0]["storage_path"] = change["storage_path"]
    if change.get("credential"):
        plan["tasks"][0]["inputs"][0]["name"] = "authorization"
    with pytest.raises(ValueError, match=error):
        Preparations(store).set_plan(
            owner, SetPreparationPlan.model_validate(arguments), policy, 0
        )
    assert store.get(owner) == PreparationState()
    assert store.project(owner) is None


@pytest.mark.parametrize("field", ["facts", "missing", "evidence", "policy"])
def test_preparation_state_rejects_removed_checklist_fields(field):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PreparationState.model_validate({field: {}})


@pytest.mark.parametrize("field", ["checklist_id", "version", "fields"])
def test_preparation_policy_rejects_removed_checklist_fields(preparation_setup, field):
    _, _, policy = preparation_setup
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PreparationPolicy.model_validate({**policy.model_dump(), field: {}})


async def test_runtime_snapshot_restore_and_project_notification(
    preparation_setup, ready_request, tmp_path
):
    store, owner, policy = preparation_setup
    adapter = FakeAdapter()
    context = RequestContext(
        user_id=owner.user_id, profile_id=owner.profile_id, x_cluster="us-west-1#pre"
    )
    root = tmp_path / "conversations"
    runtime = ConversationRuntime(root, adapter, preparation_store=store)
    try:
        initial = await runtime.create_conversation(
            SessionSpec(
                conversation_id=owner.conversation_id,
                user_id=owner.user_id,
                profile_id=owner.profile_id,
                workspace_root=str(root),
                preparation_policy=policy,
                tool_allowlist=("read", "set_preparation_plan"),
                development_tool_allowlist=("read", "write", "subagent", "ask_user"),
                command_allowlist=("user_message", "create_project"),
                delegated_roles=(DelegatedRole(name="eval"),),
            ),
            context,
        )
        assert initial.preparation is not None
        assert initial.preparation.phase == "collecting"
        for name in ("get_preparation", "update_preparation", "inspect_resource"):
            with pytest.raises(PermissionError):
                await runtime.preparation_call(owner.conversation_id, name, {}, context)
        with pytest.raises(ValueError, match="model context"):
            await runtime.preparation_call(
                owner.conversation_id,
                "set_preparation_plan",
                ready_request.model_dump(mode="json"),
                context,
            )
        revision = await runtime.preparation_revision(owner.conversation_id, context)
        result = await runtime.preparation_call(
            owner.conversation_id,
            "set_preparation_plan",
            ready_request.model_dump(mode="json"),
            replace(context, preparation_revision=revision),
        )
        assert result == {"phase": "ready", "revision": 1}
        snapshot = await runtime.get_snapshot(owner.conversation_id, context)
        assert snapshot.preparation is not None
        assert snapshot.preparation.phase == "ready"
        assert store.project(owner) is None and not adapter.external_task_notifications
        command = CreateProjectCommand(
            command_id="create", expected_preparation_revision=1
        )
        receipt = await runtime.submit_command(owner.conversation_id, command, context)
        assert receipt.status == "completed"
        assert (
            await runtime.submit_command(owner.conversation_id, command, context)
            == receipt
        )
        notification = adapter.external_task_notifications[-1][1]
        assert notification.kind == "project_start"
        assert "workflowPLAN.json" in notification.hidden_text
        assert (
            "held-out" in notification.hidden_text
            and '"facts"' not in notification.hidden_text
        )
        assert len(adapter.external_task_notifications) == 1
    finally:
        await runtime.close()
    restored = ConversationRuntime(root, FakeAdapter(), preparation_store=store)
    try:
        snapshot = await restored.get_snapshot(owner.conversation_id, context)
        assert snapshot.project is not None
        assert snapshot.preparation is not None
        assert snapshot.preparation.phase == "active"
        listed = await restored.list_snapshots(context)
        assert listed[0].preparation == snapshot.preparation
        with pytest.raises(PermissionError):
            await restored.get_snapshot(
                owner.conversation_id, replace(context, profile_id="foreign")
            )
    finally:
        await restored.close()


async def test_http_prepare_create_and_cross_profile_denial(
    preparation_setup, tmp_path
):
    store, _, _ = preparation_setup
    adapter = FakeAdapter()
    adapter.capabilities = adapter.capabilities.model_copy(
        update={"user_input_reply": True}
    )
    runtime = ConversationRuntime(
        tmp_path / "conversations", adapter, preparation_store=store
    )
    profile = replace(ECHOMIND_PRODUCT_PROFILE, harness_id="openhands")
    app = _app(tmp_path, runtime)
    app.include_router(create_product_router(profile))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"x-pyromind-debug-user-id": "42"},
        ) as client:
            response = await client.post(
                profile.route_prefix, json={"llm": {"model": "test"}}
            )
            assert response.status_code == 201, response.text
            identity = response.json()["conversation_id"]
            view = response.json()["preparation"]
            assert view["phase"] == "collecting"
            assert not {"facts", "missing", "evidence"} & view.keys()
            foreign = await client.get(
                f"/api/v2/pyromind/conversations/{identity}/preparation"
            )
            assert foreign.status_code == 404
            assert (
                await client.get(f"{profile.route_prefix}/{identity}/project")
            ).json() is None
    finally:
        await runtime.close()
