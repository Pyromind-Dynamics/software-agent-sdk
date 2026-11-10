from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import httpx
import pytest
import sqlalchemy as sa
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.preparation import Preparations
from pyromind_runtime.application.snapshot_projector import SnapshotProjector
from pyromind_runtime.domain.commands import CreateProjectCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.experiment_plans import (
    PlanConflictError,
    PlanOwner,
)
from pyromind_runtime.domain.preparation import (
    PREPARATION_TOOLS,
    ChecklistField,
    PreparationFact,
    PreparationPolicy,
    SetPreparationPlan,
    UpdatePreparation,
)
from pyromind_runtime.domain.session_policy import SessionPolicy
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.infrastructure.sql_preparation import SQLPreparationStore
from pyromind_runtime.infrastructure.sql_project_store import (
    metadata,
    projects,
)
from pyromind_runtime.ports.harness import DelegatedRole, SessionSpec

from .fake_adapter import FakeAdapter
from .test_product_api import _app


def test_preparation_tool_schemas_require_fields_needed_for_execution():
    facts = PREPARATION_TOOLS["update_preparation"].model_json_schema()
    fact = facts["$defs"]["PreparationFact"]
    assert "value" in fact["required"]
    assert {entry["type"] for entry in fact["properties"]["value"]["anyOf"]} == {
        "string",
        "object",
        "array",
        "integer",
        "boolean",
        "null",
    }

    plan = PREPARATION_TOOLS["set_preparation_plan"].model_json_schema()["$defs"]
    assert {"stages", "tasks"} <= set(plan["ExperimentContext"]["required"])
    assert "kind" in plan["ExperimentStage"]["required"]
    assert {"kind", "stage_id"} <= set(plan["ExperimentTask"]["required"])


@pytest.mark.parametrize(
    "legacy",
    [{"mode": "replace"}, {"mode": "merge"}, {"plan": {}}, {"entry_task": {}}],
)
def test_fact_tool_rejects_removed_legacy_arguments(legacy):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PREPARATION_TOOLS["update_preparation"].model_validate(
            {"expected_revision": 0, "facts": {}, **legacy}
        )


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
    policy = PreparationPolicy(
        checklist_id="test",
        version=1,
        allowed_roles=("eval",),
        fields=(
            ChecklistField(key="dataset", label="Dataset"),
            ChecklistField(key="mapping", label="Mapping", depends_on=("dataset",)),
        ),
    )
    store.initialize(owner, policy)
    yield store, owner, policy
    engine.dispose()


@pytest.fixture
def ready_request():
    return {
        "expected_revision": 1,
        "facts": {
            key: {
                "value": value,
                "status": "resolved",
                "evidence_refs": ["message:request"],
            }
            for key, value in (
                ("dataset", "/eval.jsonl"),
                ("mapping", "prompt and answer"),
            )
        },
        "plan": {
            "goal": "Evaluate model",
            "resources": [
                {
                    "id": "data",
                    "name": "Data",
                    "storage_path": "/workspace/eval.jsonl",
                }
            ],
            "stages": [{"id": "evaluation", "name": "Eval"}],
            "tasks": [
                {
                    "id": "first",
                    "name": "Evaluate",
                    "stage_id": "evaluation",
                    "inputs": [{"name": "dataset", "resource_id": "data"}],
                }
            ],
        },
    }


def make_ready(store, owner, payload):
    service = Preparations(store)
    evidence = {"message:request"}
    service.update(
        owner, UpdatePreparation(expected_revision=0, facts=payload["facts"]), evidence
    )
    return service.set_plan(
        owner,
        SetPreparationPlan(
            expected_revision=payload["expected_revision"],
            plan=payload["plan"],
            entry_task_id="first",
            role_id="eval",
        ),
    )


def test_checklist_gates_plan_and_never_creates_project(
    preparation_setup, ready_request
):
    store, owner, _ = preparation_setup
    service = Preparations(store)
    with pytest.raises(ValueError, match="checklist"):
        service.set_plan(
            owner,
            SetPreparationPlan(
                expected_revision=0,
                plan=ready_request["plan"],
                entry_task_id="first",
                role_id="eval",
            ),
        )
    ready = make_ready(store, owner, ready_request)
    assert ready.phase == "ready" and ready.project_id is None
    assert store.project(owner) is None
    with store.engine.connect() as conn:
        assert conn.scalar(sa.select(sa.func.count()).select_from(projects)) == 0
    with pytest.raises(ValueError, match="evidence"):
        service.update(
            owner,
            UpdatePreparation(expected_revision=2, facts=ready_request["facts"]),
            set(),
        )


def test_dataset_change_invalidates_dependent_information(
    preparation_setup, ready_request
):
    store, owner, _ = preparation_setup
    ready = make_ready(store, owner, ready_request)
    changed = Preparations(store).update(
        owner,
        UpdatePreparation(
            expected_revision=ready.revision,
            facts={
                "dataset": ready.facts["dataset"].model_copy(
                    update={"value": "/other.jsonl"}
                )
            },
        ),
        {"message:request"},
    )
    assert changed.phase == "collecting" and changed.missing == ("mapping",)
    assert changed.plan is None
    with pytest.raises(PlanConflictError):
        store.activate(owner, ready.revision, "old-button")


def test_fact_merge_preserves_unrelated_fields_and_reports_reserved_keys(
    preparation_setup, ready_request
):
    store, owner, policy = preparation_setup
    owner = owner.model_copy(update={"conversation_id": "merge"})
    store.initialize(
        owner,
        policy.model_copy(
            update={
                "fields": (
                    *policy.fields,
                    ChecklistField(key="scope", label="Scope", value_type="object"),
                )
            }
        ),
    )
    service = Preparations(store)
    evidence = {"message:request"}
    first = service.update(
        owner,
        UpdatePreparation(expected_revision=0, facts=ready_request["facts"]),
        evidence,
    )
    request = UpdatePreparation(
        expected_revision=first.revision,
        facts={
            "scope": PreparationFact(
                value={"authorization": "User requested training"},
                status="resolved",
                evidence_refs=("message:request",),
            )
        },
    )
    with pytest.raises(ValueError, match=r"facts\.scope\.value\.authorization") as exc:
        service.update(owner, request, evidence)
    assert "User requested training" not in str(exc.value)
    assert store.get(owner) == first
    with pytest.raises(
        ValueError, match=r"unknown checklist field\(s\): decision_note"
    ):
        service.update(
            owner,
            request.model_copy(
                update={"facts": {"decision_note": request.facts["scope"]}}
            ),
            evidence,
        )
    assert store.get(owner) == first
    scope = request.facts["scope"].model_copy(
        update={"value": {"decision_note": "User requested training"}}
    )
    merged = service.update(
        owner, request.model_copy(update={"facts": {"scope": scope}}), evidence
    )
    assert merged.facts == {**first.facts, "scope": scope}
    cleared = service.update(
        owner,
        UpdatePreparation(
            expected_revision=merged.revision,
            facts={"scope": PreparationFact()},
        ),
        evidence,
    )
    assert cleared.facts == {**first.facts, "scope": PreparationFact()}
    assert cleared.missing == ("scope",)


def test_saving_after_inspection_preserves_new_evidence_and_revision(
    preparation_setup, ready_request
):
    store, owner, _ = preparation_setup
    ready = make_ready(store, owner, ready_request)
    assert (
        store.add_evidence(owner, "resource:first", "first preview") == ready.revision
    )
    assert store.get(owner).revision == ready.revision
    saved = store.save(owner, ready, ready.revision)
    assert saved.evidence == {"resource:first": "first preview"}


def test_new_message_withdraws_ready_and_is_idempotent(
    preparation_setup, ready_request
):
    store, owner, _ = preparation_setup
    ready = make_ready(store, owner, ready_request)
    store.invalidate(owner, "message-command")
    current = store.get(owner)
    assert current.review_pending and current.phase == "planning"
    store.invalidate(owner, "message-command")
    assert store.get(owner).revision == current.revision
    with pytest.raises(PlanConflictError):
        store.activate(owner, ready.revision, "stale")


def test_project_creation_is_atomic_and_deduplicated_across_writers(
    preparation_setup, ready_request
):
    store, owner, _ = preparation_setup
    ready = make_ready(store, owner, ready_request)

    def create(index):
        return SQLPreparationStore(store.engine).activate(
            owner,
            ready.revision,
            f"button-{index}",
            cluster="us-west-1#pre",
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
    for foreign in (
        owner.model_copy(update={"user_id": "other"}),
        owner.model_copy(update={"profile_id": "other"}),
        owner.model_copy(update={"conversation_id": "other"}),
    ):
        with pytest.raises(PermissionError):
            store.get(foreign)
        assert store.project(foreign) is None


def test_unknown_stage_role_inputs_and_credentials_are_rejected(
    preparation_setup, ready_request
):
    store, owner, _ = preparation_setup
    Preparations(store).update(
        owner,
        UpdatePreparation(expected_revision=0, facts=ready_request["facts"]),
        {"message:request"},
    )
    request = {
        "expected_revision": 1,
        "plan": ready_request["plan"],
        "entry_task_id": "first",
        "role_id": "eval",
    }
    task = ready_request["plan"]["tasks"][0]
    for invalid in (
        {**request, "role_id": "train"},
        {**request, "entry_task_id": "unknown"},
        {
            **request,
            "plan": {
                **request["plan"],
                "tasks": [{**task, "stage_id": "unknown"}],
            },
        },
        {
            **request,
            "plan": {
                **request["plan"],
                "tasks": [
                    {**task, "inputs": [{"name": "data", "resource_id": "unknown"}]}
                ],
            },
        },
        {
            **request,
            "plan": {
                **request["plan"],
                "resources": [
                    {
                        "id": "data",
                        "name": "Data",
                        "uri": "https://user:secret@example.com",
                    }
                ],
            },
        },
    ):
        with pytest.raises(ValueError):
            Preparations(store).set_plan(
                owner, SetPreparationPlan.model_validate(invalid)
            )
        assert store.get(owner).revision == 1
    assert store.get(owner).phase == "planning"


def test_stale_preparation_update_returns_latest_revision_and_facts(
    preparation_setup, ready_request
):
    store, owner, _ = preparation_setup
    first = store.get(owner)
    concurrent = first.model_copy(
        update={
            "facts": {
                **first.facts,
                "dataset": PreparationFact(
                    value="/newer.jsonl",
                    status="resolved",
                    evidence_refs=("message:request",),
                ),
            }
        }
    )
    store.save(owner, concurrent, first.revision)

    with pytest.raises(ValueError) as exc_info:
        Preparations(store).update(
            owner,
            UpdatePreparation.model_validate(
                {
                    "expected_revision": first.revision,
                    "facts": ready_request["facts"],
                }
            ),
            {"message:request"},
        )

    details = json.loads(str(exc_info.value))
    assert details["error"] == "preparation_revision_conflict"
    assert details["expected_revision"] == first.revision
    assert details["current_revision"] == first.revision + 1
    assert details["current_preparation"]["facts"]["dataset"]["value"] == "/newer.jsonl"
    assert "Merge" in details["action"]


def test_invalid_workflow_target_returns_current_enabled_targets():
    from pyromind_runtime.application.preparation import validate_workflow_role
    from pyromind_runtime.domain.experiment_plans import ExperimentContext

    policy = ECHOMIND_PRODUCT_PROFILE.preparation_policy
    assert policy is not None
    plan = ExperimentContext.model_validate(
        {
            "goal": "Run evaluation",
            "stages": [
                {
                    "id": "disabled",
                    "kind": "eval",
                    "name": "Disabled",
                    "enabled": False,
                },
                {"id": "review", "kind": "eval", "name": "Review"},
                {"id": "train", "kind": "train", "name": "Training"},
                {
                    "id": "placeholder",
                    "kind": "inference",
                    "name": "Placeholder",
                    "placeholder": True,
                    "enabled": False,
                },
            ],
            "tasks": [],
        }
    )

    with pytest.raises(ValueError) as exc_info:
        validate_workflow_role(policy, plan, "train", "eval")

    details = json.loads(str(exc_info.value))
    assert details["error"] == "invalid_execution_target"
    assert details["requested"] == {"stage_id": "train", "role_id": "eval"}
    assert details["valid_targets"] == [
        {
            "stage_id": "review",
            "stage_kind": "eval",
            "stage_name": "Review",
            "roles": ["eval"],
        },
        {
            "stage_id": "train",
            "stage_kind": "train",
            "stage_name": "Training",
            "roles": ["train"],
        },
    ]
    assert "new request_id" in details["action"]


async def test_runtime_creation_restore_permissions_and_stage_protection(
    preparation_setup, ready_request, tmp_path
):
    preparation_store, owner, policy = preparation_setup
    adapter = FakeAdapter()
    context = RequestContext(
        user_id=owner.user_id,
        profile_id=owner.profile_id,
        x_cluster="us-west-1#pre",
    )
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        adapter,
        preparation_store=preparation_store,
    )
    tools = ("read", "get_preparation", "update_preparation", "inspect_resource")
    development = ("read", "write", "edit", "terminal", "subagent", "ask_user")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id=owner.conversation_id,
            user_id=owner.user_id,
            profile_id=owner.profile_id,
            workspace_root=str(tmp_path / "conversations"),
            preparation_policy=policy,
            tool_allowlist=tools,
            development_tool_allowlist=development,
            command_allowlist=("user_message", "create_project"),
            delegated_roles=(
                DelegatedRole(
                    name="eval",
                    description="Evaluate",
                    prompt="Evaluate within the supplied scope",
                ),
            ),
        ),
        context,
    )
    for tool in ("update_experiment_context", "request_experiment_task"):
        with pytest.raises(PermissionError):
            await runtime.preparation_call(owner.conversation_id, tool, {}, context)
    ready = make_ready(preparation_store, owner, ready_request)
    command = CreateProjectCommand(
        command_id="create", expected_preparation_revision=ready.revision
    )
    receipt = await asyncio.wait_for(
        runtime.submit_command(owner.conversation_id, command, context), 5
    )
    assert receipt.status == "completed"
    project = preparation_store.project(owner)
    assert project is not None
    assert project.cluster == "us-west-1#pre"
    assert project.initial_run_id is None
    notification = adapter.external_task_notifications[-1][1]
    assert notification.kind == "project_start"
    assert "/eval.jsonl" in notification.hidden_text
    assert "workflowPLAN.json" in notification.hidden_text
    assert adapter.session_policies[-1][1].tool_allowlist == development
    assert (
        await runtime.submit_command(owner.conversation_id, command, context) == receipt
    )
    policies = len(adapter.session_policies)
    another = await runtime.submit_command(
        owner.conversation_id,
        CreateProjectCommand(
            command_id="another-click", expected_preparation_revision=ready.revision
        ),
        context,
    )
    assert another.response["project_id"] == receipt.response["project_id"]
    assert len(adapter.session_policies) == policies
    assert {notice.task_id for _, notice, _ in adapter.external_task_notifications} == {
        "project-start:" + project.project_id
    }
    snapshot = await runtime.get_snapshot(owner.conversation_id, context)
    assert snapshot.project is not None and snapshot.preparation is not None
    assert snapshot.preparation.phase == "active"
    await runtime.close()
    restored = ConversationRuntime(
        tmp_path / "conversations",
        FakeAdapter(),
        preparation_store=preparation_store,
    )
    result = await restored.get_snapshot(owner.conversation_id, context)
    assert result.project is not None
    assert result.project.project_id == snapshot.project.project_id
    assert result.project.initial_run_id is None
    listed = await restored.list_snapshots(context)
    assert listed[0].project == result.project
    with pytest.raises(PermissionError):
        await restored.get_snapshot(
            owner.conversation_id, RequestContext(user_id=owner.user_id)
        )
    await restored.close()


async def test_http_prepare_create_and_cross_profile_denial(
    preparation_setup, ready_request, tmp_path
):
    store, _, _ = preparation_setup
    adapter = FakeAdapter()
    adapter.capabilities = adapter.capabilities.model_copy(
        update={"user_input_reply": True}
    )
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        adapter,
        preparation_store=store,
    )
    profile = replace(ECHOMIND_PRODUCT_PROFILE, harness_id="openhands")
    app = _app(tmp_path, runtime)
    app.include_router(create_product_router(profile))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"x-pyromind-debug-user-id": "42"},
    ) as client:
        response = await client.post(
            profile.route_prefix, json={"llm": {"model": "test"}}
        )
        assert response.status_code == 201, response.text
        snapshot = response.json()
        assert snapshot["preparation"]["phase"] == "collecting"
        assert snapshot.get("project") is None
        identity = snapshot["conversation_id"]
        denied = await client.post(
            f"{profile.route_prefix}/{identity}/commands",
            json={
                "type": "confirm_experiment_plan",
                "command_id": "legacy-bypass",
                "proposal_id": "none",
                "expected_revision": 0,
            },
        )
        assert denied.status_code == 422
        before = await client.get(f"{profile.route_prefix}/{identity}/project")
        assert before.status_code == 200 and before.json() is None
        foreign = await client.get(
            f"/api/v2/pyromind/conversations/{identity}/preparation"
        )
        assert foreign.status_code == 404
    await runtime.close()


def test_adapter_binding_projects_started_progress_and_report_rounds():
    frame = {
        "eventId": "child:start",
        "sessionId": "parent",
        "runId": "native-child",
        "kind": "subagent.started",
        "payload": {
            "details": {
                "subagent": {
                    "version": 1,
                    "execution": {
                        "project_id": "project",
                        "task_run_id": "business-run",
                        "stage_id": "evaluation",
                        "plan_sequence": 1,
                    },
                }
            }
        },
    }
    (event,) = translate_runner_event(frame)
    binding = event.payload["execution"]
    assert isinstance(binding, dict) and binding["task_run_id"] == "business-run"
    assert event.type == "operation.started"
    from pyromind_runtime.domain.capabilities import HarnessCapabilities
    from pyromind_runtime.domain.snapshot import ConversationSnapshot

    snapshot = SnapshotProjector().reduce(
        ConversationSnapshot(
            conversation_id="parent", capabilities=HarnessCapabilities()
        ),
        ProductEvent(
            conversation_id="parent",
            seq=1,
            type=event.type,
            payload=event.payload,
            run_id=event.run_id,
        ),
    )
    operation = snapshot.timeline[0]
    assert operation.kind == "operation" and operation.execution is not None
    assert operation.execution["stage_id"] == "evaluation"
    assert snapshot.status == "idle"


async def test_inspection_and_terminal_provide_checklist_evidence_without_project(
    preparation_setup, ready_request, tmp_path
):
    store, owner, policy = preparation_setup
    from unittest.mock import Mock

    inspector = Mock()
    inspector.inspect.return_value = {
        "content": "prompt and answer",
        "scope": "file prefix",
        "truncated": True,
    }
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        FakeAdapter(),
        preparation_store=store,
        resource_inspector=inspector,
    )
    context = RequestContext(
        user_id=owner.user_id, profile_id=owner.profile_id, cookie="auth_token=private"
    )
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=owner.conversation_id,
                user_id=owner.user_id,
                profile_id=owner.profile_id,
                workspace_root=str(tmp_path / "conversations"),
                initial_message=(TextContent(text="Evaluate /eval.jsonl"),),
                tool_allowlist=(
                    "read",
                    "get_preparation",
                    "update_preparation",
                    "inspect_resource",
                ),
                preparation_policy=policy,
                delegated_roles=(
                    DelegatedRole(
                        name="eval",
                        description="Evaluate",
                        prompt="Evaluate within the supplied scope",
                    ),
                ),
            ),
            context,
        )
        before = await runtime.get_preparation(owner.conversation_id, context)
        inspected, second = await asyncio.gather(
            *(
                runtime.preparation_call(
                    owner.conversation_id, "inspect_resource", {"path": path}, context
                )
                for path in ("/eval.jsonl", "/model/config.json")
            )
        )
        product = FileProductStore(tmp_path / "conversations" / owner.conversation_id)
        for operation_id, succeeded in (
            ("count-rows", True),
            ("failed-probe", False),
        ):
            product.append(
                ProductEvent(
                    conversation_id=owner.conversation_id,
                    type="operation.completed" if succeeded else "operation.failed",
                    payload={"operation_id": operation_id, "name": "terminal"},
                )
            )
        view = await runtime.get_preparation(owner.conversation_id, context)
        reference = inspected["evidence_ref"]
        assert isinstance(reference, str)
        assert second["evidence_ref"] in view.evidence
        assert (
            view.revision
            == inspected["revision"]
            == second["revision"]
            == before.revision
        )
        assert reference in view.evidence and "message:initial-message" in view.evidence
        assert "truncated=True" in view.evidence[reference]
        assert view.evidence["operation:count-rows"] == "terminal"
        assert "operation:failed-probe" not in view.evidence
        facts = {
            key: {**fact, "evidence_refs": [reference, "operation:count-rows"]}
            for key, fact in ready_request["facts"].items()
        }
        request = UpdatePreparation.model_validate(
            {"expected_revision": before.revision, "facts": facts}
        )
        result = await runtime.preparation_call(
            owner.conversation_id,
            "update_preparation",
            request.model_dump(mode="json"),
            context,
        )
        assert result["phase"] == "planning"
        assert store.project(owner) is None
        assert inspector.inspect.call_args.args[1].cookie == context.cookie
        assert "private" not in str(store.get(owner).model_dump())
    finally:
        await runtime.close()


def test_checklist_dependencies_are_order_independent_and_acyclic(
    preparation_setup, ready_request
):
    store, owner, policy = preparation_setup
    from pyromind_runtime.domain.preparation import ChecklistField

    with pytest.raises(ValueError, match="cyclic"):
        policy.model_validate(
            {
                **policy.model_dump(),
                "fields": [
                    {"key": "a", "label": "A", "depends_on": ["b"]},
                    {"key": "b", "label": "B", "depends_on": ["a"]},
                ],
            }
        )
    other = owner.model_copy(update={"conversation_id": "reverse-dependencies"})
    policy = policy.model_copy(
        update={
            "fields": (
                ChecklistField(key="scoring", label="Scoring", depends_on=("mapping",)),
                *reversed(policy.fields),
            )
        }
    )
    store.initialize(other, policy)
    facts = {
        **ready_request["facts"],
        "scoring": {
            "value": "exact match",
            "status": "resolved",
            "evidence_refs": ["message:request"],
        },
    }
    service = Preparations(store)
    first = service.update(
        other,
        UpdatePreparation.model_validate({"expected_revision": 0, "facts": facts}),
        {"message:request"},
    )
    facts["dataset"] = {**facts["dataset"], "value": "/other.jsonl"}
    changed = service.update(
        other,
        UpdatePreparation.model_validate(
            {"expected_revision": first.revision, "facts": facts}
        ),
        {"message:request"},
    )
    assert set(changed.missing) == {"mapping", "scoring"}


@pytest.mark.parametrize(
    "location",
    [{"storage_path": "/workspace/eval.jsonl"}, {"uri": "/eval.jsonl"}],
)
def test_plan_submission_reuses_collected_facts_and_resolves_entry_resources(
    preparation_setup, ready_request, location
):
    store, owner, _ = preparation_setup
    service = Preparations(store)
    collected = service.update(
        owner,
        UpdatePreparation(expected_revision=0, facts=ready_request["facts"]),
        {"message:request"},
    )
    plan = {
        **ready_request["plan"],
        "resources": [{"id": "data", "name": "Data", **location}],
    }
    ready = service.set_plan(
        owner,
        SetPreparationPlan.model_validate(
            {
                "expected_revision": collected.revision,
                "plan": plan,
                "entry_task_id": "first",
                "role_id": "eval",
            }
        ),
    )
    assert ready.phase == "ready" and ready.facts == collected.facts
    assert ready.entry_task is not None and ready.entry_task.task.inputs == {
        "dataset": "/workspace/eval.jsonl"
    }
    assert ready.entry_task.task.name == "Evaluate"
    assert store.project(owner) is None


def test_plan_retry_requires_complete_arguments_and_stage_edges(
    preparation_setup, ready_request
):
    from pydantic import ValidationError

    store, owner, _ = preparation_setup
    ready = make_ready(store, owner, ready_request)
    assert ready.plan is not None
    plan = ready.plan.model_dump(mode="json")
    plan["stages"].append({"id": "review", "name": "Review"})
    plan["tasks"].append(
        {"id": "second", "stage_id": "review", "name": "Explain results"}
    )
    arguments = {
        "expected_revision": ready.revision,
        "entry_task_id": "first",
        "role_id": "eval",
        "plan": plan,
    }
    plan["edges"] = [{"source": "first", "target": "second"}]
    with pytest.raises(ValueError, match=r"plan.stages\[\].id.*evaluation.*review"):
        SetPreparationPlan.model_validate(arguments)
    plan["edges"] = [{"source": "evaluation", "target": "review"}]
    with pytest.raises(ValueError, match=r"Entry task 'first'.*inputs.*resource_id"):
        Preparations(store).set_plan(
            owner,
            SetPreparationPlan.model_validate(
                {
                    **arguments,
                    "plan": {**plan, "tasks": [{**plan["tasks"][0], "inputs": []}]},
                }
            ),
        )
    with pytest.raises(ValidationError, match="role_id"):
        SetPreparationPlan.model_validate(
            {key: value for key, value in arguments.items() if key != "role_id"}
        )
    assert store.get(owner) == ready
    saved = Preparations(store).set_plan(
        owner, SetPreparationPlan.model_validate(arguments)
    )
    assert saved.phase == "ready"
    assert saved.entry_task is not None and saved.plan is not None
    assert saved.entry_task.role_id == "eval"
    assert saved.plan.edges[0].source == "evaluation"
    assert store.project(owner) is None


@pytest.mark.parametrize("phase", ["legacy", "collecting", "active"])
async def test_preparation_capability_upgrade_preserves_contract_and_project(
    preparation_setup, ready_request, tmp_path, phase
):
    preparations, owner, policy = preparation_setup
    adapter = FakeAdapter()
    root = tmp_path / "conversations"
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)
    runtime = ConversationRuntime(
        root,
        adapter,
        preparation_store=preparations,
    )
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=owner.conversation_id,
                user_id=owner.user_id,
                profile_id=owner.profile_id,
                workspace_root=str(root),
                tool_allowlist=("read",),
                preparation_policy=policy if phase != "legacy" else None,
                delegated_roles=(
                    DelegatedRole(
                        name="eval", description="Evaluate", prompt="Evaluate"
                    ),
                ),
            ),
            context,
        )
        if phase == "active":
            ready = make_ready(preparations, owner, ready_request)
            preparations.activate(owner, ready.revision, "create")
        before = preparations.get(owner)
        project = preparations.project(owner)
        tools = ("read", "write", "edit", "terminal", "grep", "find", "ls")
        development = (*tools, "subagent")
        configuration = SessionPolicy(
            configuration_version=7,
            preparation_policy=policy.model_copy(update={"version": 2}),
            tool_allowlist=tools,
            development_tool_allowlist=development,
            product_instructions="Read the child's Markdown report.",
            delegated_roles=(
                {"name": "eval", "prompt": "Return a Markdown report."},
                {"name": "new-role", "prompt": "A role outside the saved policy."},
            ),
        )
        foreign = replace(context, profile_id="another-profile")
        with pytest.raises(PermissionError):
            await runtime.apply_session_configuration(
                owner.conversation_id, configuration, foreign
            )
        assert await runtime.apply_session_configuration(
            owner.conversation_id, configuration, context
        )
        store = FileProductStore(root / owner.conversation_id)
        if phase == "legacy":
            assert not adapter.session_policies
            assert store.access_policy()["tool_allowlist"] == ("read",)
        else:
            expected = development if phase == "active" else tools
            assert store.access_policy()["tool_allowlist"] == expected
            assert adapter.session_policies[-1][1].tool_allowlist == expected
            refreshed = adapter.session_policies[-1][1]
            assert refreshed.product_instructions == configuration.product_instructions
            assert configuration.delegated_roles is not None
            assert refreshed.delegated_roles == configuration.delegated_roles[:1]
            assert store.access_policy()["delegated_roles"] == ("eval",)
            assert store.preparation_policy() == policy
            assert store.development_policy().tool_allowlist == development
        assert preparations.get(owner) == before
        assert preparations.project(owner) == project
    finally:
        await runtime.close()


def test_workflow_catalog_rejects_generated_node_and_task_types(
    preparation_setup, ready_request
):
    store, owner, policy = preparation_setup
    owner = owner.model_copy(update={"conversation_id": "catalog"})
    policy = PreparationPolicy.model_validate(
        {
            **policy.model_dump(),
            "workflow_nodes": [
                {
                    "id": "measure",
                    "name": "Measure",
                    "roles": ["eval"],
                    "tasks": [{"id": "score", "display_name": "Baseline"}],
                },
                {"id": "future", "name": "Future"},
            ],
        }
    )
    store.initialize(owner, policy)
    service = Preparations(store)
    collected = service.update(
        owner,
        UpdatePreparation(expected_revision=0, facts=ready_request["facts"]),
        {"message:request"},
    )
    plan = ready_request["plan"]
    plan["stages"][0]["kind"] = "measure"
    plan["tasks"][0]["kind"] = "score"
    request = SetPreparationPlan.model_validate(
        {
            "expected_revision": collected.revision,
            "plan": plan,
            "entry_task_id": "first",
            "role_id": "eval",
        }
    )
    for invalid, error in (
        (
            request.plan.model_copy(
                update={
                    "stages": (
                        request.plan.stages[0].model_copy(update={"kind": "invented"}),
                    )
                }
            ),
            r"plan\.stages\[0\]\.kind",
        ),
        (
            request.plan.model_copy(
                update={
                    "tasks": (
                        request.plan.tasks[0].model_copy(update={"kind": "invented"}),
                    )
                }
            ),
            r"plan\.tasks\[0\]\.kind.*allowed task kinds: score",
        ),
        (
            request.plan.model_copy(
                update={
                    "stages": (
                        request.plan.stages[0].model_copy(update={"kind": "future"}),
                    )
                }
            ),
            "unimplemented catalog nodes",
        ),
    ):
        with pytest.raises(ValueError, match=error):
            service.set_plan(owner, request.model_copy(update={"plan": invalid}))
        assert store.get(owner).revision == collected.revision
        assert store.project(owner) is None
    ready = service.set_plan(owner, request)
    assert ready.phase == "ready"
    assert ready.plan is not None
    assert ready.plan.stages[0].kind == "measure"
    assert ready.plan.tasks[0].kind == "score"
    assert ready.plan.stages[0].id == "evaluation"


def test_catalog_is_profile_owned_and_old_plans_remain_compatible():
    from pyromind_runtime.application.preparation import validate_workflow_catalog
    from pyromind_runtime.domain.experiment_plans import ExperimentContext

    policy = ECHOMIND_PRODUCT_PROFILE.preparation_policy
    assert policy is not None
    nodes = {node.id: node for node in policy.workflow_nodes}
    assert set(nodes) == {
        "init",
        "dataset",
        "train",
        "eval",
        "inference",
        "tracer",
        "rawdata",
    }
    assert nodes["eval"].roles == ("eval",)
    assert {task.id for task in nodes["init"].tasks} == {"bench", "analysis"}
    assert {task.id for task in nodes["eval"].tasks} == {"checkpoint_eval", "analysis"}
    assert nodes["train"].roles == ("train",)
    assert nodes["dataset"].roles == ("dataset",)
    assert nodes["init"].roles == ("init",)
    assert nodes["eval"].roles == ("eval",)
    assert not nodes["inference"].roles
    context = ExperimentContext.model_validate(
        {
            "goal": "Baseline",
            "stages": [
                {"id": "arbitrary-instance", "name": "Renamable", "kind": "init"}
            ],
            "tasks": [
                {
                    "id": "baseline",
                    "kind": "bench",
                    "name": "Run baseline",
                    "stage_id": "arbitrary-instance",
                }
            ],
        }
    )
    validate_workflow_catalog(policy, context)
    assert context.stages[0].name == "Renamable"
    assert context.stages[0].kind == "init"
    combined = context.model_copy(
        update={
            "tasks": (
                *context.tasks,
                context.tasks[0].model_copy(
                    update={"id": "read-baseline", "kind": "analysis"}
                ),
            )
        }
    )
    validate_workflow_catalog(policy, combined)
    assert len(combined.stages) == 1 and len(combined.tasks) == 2
    with pytest.raises(ValueError, match="instance limit"):
        validate_workflow_catalog(
            policy,
            context.model_copy(
                update={
                    "stages": (
                        *context.stages,
                        context.stages[0].model_copy(update={"id": "extra"}),
                    ),
                }
            ),
        )
    legacy = ExperimentContext.model_validate(
        {"goal": "Legacy", "stages": [{"id": "legacy", "name": "Legacy"}]}
    )
    validate_workflow_catalog(policy.model_copy(update={"workflow_nodes": ()}), legacy)
    assert "kind" not in legacy.stages[0].model_dump()
