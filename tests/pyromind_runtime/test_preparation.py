from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import UUID

import httpx
import pytest
import sqlalchemy as sa
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.experiment_plan_projects import (
    EchoMindPreparationProjectMaterializer,
)
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.preparation import Preparations
from pyromind_runtime.application.snapshot_projector import SnapshotProjector
from pyromind_runtime.domain.commands import CreateProjectCommand
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.experiment_plans import (
    PlanConflictError,
    PlanOwner,
    UpdateExperimentContext,
)
from pyromind_runtime.domain.preparation import (
    ChecklistField,
    PreparationFact,
    PreparationPolicy,
    SetPreparationPlan,
    UpdatePreparation,
    WorkflowNodeDefinition,
    WorkflowTaskDefinition,
)
from pyromind_runtime.domain.session_policy import SessionPolicy
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.infrastructure.sql_experiment_execution import (
    SQLExperimentExecutionStore,
    runs,
)
from pyromind_runtime.infrastructure.sql_experiment_plan_store import (
    SQLExperimentPlanStore,
    metadata,
    projects,
)
from pyromind_runtime.infrastructure.sql_preparation import SQLPreparationStore
from pyromind_runtime.infrastructure.sql_work_queue import (
    metadata as queue_metadata,
    work,
)
from pyromind_runtime.ports.harness import DelegatedRole, SessionSpec
from pyromind_runtime.ports.storage import EvaluationStorageUnavailableError

from openhands.agent_server.echomind_projects import ProjectStore

from .fake_adapter import FakeAdapter
from .test_product_api import _app


@pytest.fixture
def preparation_setup(tmp_path):
    engine = sa.create_engine(
        f"sqlite:///{tmp_path / 'preparation.sqlite'}",
        connect_args={"check_same_thread": False},
    )
    metadata.create_all(engine)
    queue_metadata.create_all(engine)
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
            "resources": [{"id": "data", "name": "Data", "uri": "/eval.jsonl"}],
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
        "entry_task": {
            "role_id": "eval",
            "request_id": "first",
            "stage_id": "evaluation",
            "task": {
                "name": "Evaluate",
                "instruction": "Use the confirmed rubric",
                "inputs": {"dataset": "/eval.jsonl", "model": "/model", "limit": 2},
            },
        },
    }


def make_ready(store, owner, payload):
    service = Preparations(store)
    evidence = {"message:request"}
    service.update(
        owner, UpdatePreparation(expected_revision=0, facts=payload["facts"]), evidence
    )
    return service.update(owner, UpdatePreparation.model_validate(payload), evidence)


async def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


def test_checklist_gates_plan_and_never_creates_project(
    preparation_setup, ready_request
):
    store, owner, _ = preparation_setup
    service = Preparations(store)
    request = {**ready_request, "expected_revision": 0, "facts": {}}
    with pytest.raises(ValueError, match="checklist"):
        service.update(owner, UpdatePreparation.model_validate(request), set())
    ready = make_ready(store, owner, ready_request)
    assert ready.phase == "ready" and ready.project_id is None
    assert store.project(owner) is None
    with store.engine.connect() as conn:
        assert conn.scalar(sa.select(sa.func.count()).select_from(projects)) == 0
        assert conn.scalar(sa.select(sa.func.count()).select_from(runs)) == 0
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
    facts = dict(ready.facts)
    facts["dataset"] = facts["dataset"].model_copy(update={"value": "/other.jsonl"})
    changed = Preparations(store).update(
        owner,
        UpdatePreparation(expected_revision=ready.revision, facts=facts),
        {"message:request"},
    )
    assert changed.phase == "collecting" and changed.missing == ("mapping",)
    assert changed.plan is None
    with pytest.raises(PlanConflictError):
        store.activate(owner, ready.revision, "old-button")


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
            owner, ready.revision, f"button-{index}"
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(create, range(3)))
    assert len({result.project_id for result in results}) == 1
    assert len({result.initial_run_id for result in results}) == 1
    with store.engine.connect() as conn:
        assert conn.scalar(sa.select(sa.func.count()).select_from(projects)) == 1
        assert conn.scalar(sa.select(sa.func.count()).select_from(runs)) == 1
        assert conn.scalar(sa.select(sa.func.count()).select_from(work)) == 1
    run = store.executions.list(owner)[0]
    assert run["task_spec"]["stage_id"] == "evaluation"
    assert run["task_spec"]["plan_sequence"] == 1
    assert run["task_spec"]["stage_snapshot"]["name"] == "Eval"
    assert run["task_spec"]["preparation_facts"]["dataset"] == "/eval.jsonl"
    for foreign in (
        owner.model_copy(update={"user_id": "other"}),
        owner.model_copy(update={"profile_id": "other"}),
        owner.model_copy(update={"conversation_id": "other"}),
    ):
        with pytest.raises(PermissionError):
            store.get(foreign)
        assert store.project(foreign) is None


def test_queue_failure_rolls_back_project_and_execution(
    preparation_setup, ready_request, monkeypatch
):
    store, owner, _ = preparation_setup
    ready = make_ready(store, owner, ready_request)

    def fail(*args, **kwargs):
        raise RuntimeError("queue write failed")

    monkeypatch.setattr(store.executions.queue, "put", fail)
    with pytest.raises(RuntimeError, match="queue write"):
        store.activate(owner, ready.revision, "create")
    assert store.project(owner) is None
    assert store.get(owner).phase == "ready"
    with store.engine.connect() as conn:
        assert conn.scalar(sa.select(sa.func.count()).select_from(runs)) == 0


def test_unknown_stage_role_inputs_and_credentials_are_rejected(
    preparation_setup, ready_request
):
    store, owner, _ = preparation_setup
    Preparations(store).update(
        owner,
        UpdatePreparation(expected_revision=0, facts=ready_request["facts"]),
        {"message:request"},
    )
    for entry in (
        {**ready_request["entry_task"], "stage_id": "unknown"},
        {**ready_request["entry_task"], "role_id": "train"},
        {**ready_request["entry_task"], "request_id": "unknown"},
        {
            **ready_request["entry_task"],
            "task": {
                "name": "Invalid",
                "instruction": "Invalid",
                "inputs": {"dataset": "/another.jsonl"},
            },
        },
        {
            **ready_request["entry_task"],
            "task": {
                "name": "Invalid",
                "instruction": "Invalid",
                "inputs": {
                    "dataset": "/eval.jsonl",
                    "authorization": "Bearer forbidden",
                },
            },
        },
    ):
        with pytest.raises(ValueError):
            Preparations(store).update(
                owner,
                UpdatePreparation.model_validate(
                    {**ready_request, "entry_task": entry}
                ),
                {"message:request"},
            )
    assert store.get(owner).phase == "planning"


async def test_runtime_creation_restore_permissions_and_stage_protection(
    preparation_setup, ready_request, tmp_path
):
    preparation_store, owner, policy = preparation_setup
    adapter = FakeAdapter()
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)
    plans = SQLExperimentPlanStore(preparation_store.engine)
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        adapter,
        experiment_plan_store=plans,
        preparation_store=preparation_store,
    )
    tools = ("read", "get_preparation", "update_preparation", "inspect_resource")
    development = tools + (
        "get_experiment_context",
        "update_experiment_context",
        "request_experiment_task",
        "get_experiment_execution",
        "control_experiment_task",
    )
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
            await runtime.experiment_plan_call(owner.conversation_id, tool, {}, context)
    ready = make_ready(preparation_store, owner, ready_request)
    command = CreateProjectCommand(
        command_id="create", expected_preparation_revision=ready.revision
    )
    receipt = await asyncio.wait_for(
        runtime.submit_command(owner.conversation_id, command, context), 5
    )
    assert receipt.status == "completed"
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
    snapshot = await runtime.get_snapshot(owner.conversation_id, context)
    assert snapshot.project is not None and snapshot.preparation is not None
    assert snapshot.preparation.phase == "active"
    await runtime.close()
    restored = ConversationRuntime(
        tmp_path / "conversations",
        FakeAdapter(),
        experiment_plan_store=SQLExperimentPlanStore(preparation_store.engine),
        preparation_store=preparation_store,
    )
    result = await restored.get_snapshot(owner.conversation_id, context)
    assert result.project is not None
    assert result.project.project_id == snapshot.project.project_id
    assert result.project.initial_run_id == receipt.response["initial_run_id"]
    listed = await restored.list_snapshots(context)
    assert listed[0].project == result.project
    with pytest.raises(PermissionError):
        await restored.get_snapshot(
            owner.conversation_id, RequestContext(user_id=owner.user_id)
        )
    await restored.close()


async def test_create_project_materializes_working_graph_without_starting_task(
    preparation_setup, tmp_path
):
    store, _, _ = preparation_setup
    owner = PlanOwner(
        user_id="42",
        conversation_id="graph-merge",
        profile_id="prepared",
    )
    policy = PreparationPolicy(
        checklist_id="graph-merge",
        version=1,
        allowed_roles=("eval",),
        create_project_starts_entry_task=False,
        fields=(ChecklistField(key="dataset", label="Dataset"),),
        workflow_nodes=(
            WorkflowNodeDefinition(
                id="init",
                name="Init",
                roles=("eval",),
                tasks=(WorkflowTaskDefinition(id="bench", description="Baseline"),),
            ),
            WorkflowNodeDefinition(
                id="dataset",
                name="Dataset",
                tasks=(
                    WorkflowTaskDefinition(
                        id="clean_label",
                        description="Prepare the dataset",
                    ),
                ),
            ),
        ),
    )
    store.initialize(owner, policy)
    service = Preparations(store)
    facts = {
        "dataset": PreparationFact(
            value="/datasets/pcb_avi_eval_3",
            status="resolved",
            evidence_refs=("message:request",),
        )
    }
    collected = service.update(
        owner,
        UpdatePreparation(expected_revision=0, facts=facts),
        {"message:request"},
    )
    plan = {
        "goal": "Prepare an AVI dataset and then train a model",
        "resources": [
            {
                "id": "source",
                "name": "Source dataset",
                "uri": "/datasets/pcb_avi_eval_3",
            }
        ],
        "stages": [
            {"id": "init", "kind": "init", "name": "Init"},
            {"id": "dataset", "kind": "dataset", "name": "Dataset"},
        ],
        "tasks": [
            {
                "id": "bench",
                "kind": "bench",
                "name": "Baseline",
                "stage_id": "init",
                "inputs": [{"name": "dataset", "resource_id": "source"}],
            },
            {
                "id": "clean_label",
                "kind": "clean_label",
                "name": "Prepare dataset",
                "stage_id": "dataset",
                "inputs": [{"name": "dataset", "resource_id": "source"}],
            },
        ],
        "edges": [{"source": "init", "target": "dataset"}],
    }
    service.set_plan(
        owner,
        SetPreparationPlan.model_validate(
            {
                "expected_revision": collected.revision,
                "plan": plan,
            }
        ),
    )
    context = RequestContext(user_id="42", profile_id="prepared")
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        FakeAdapter(),
        experiment_plan_store=SQLExperimentPlanStore(store.engine),
        preparation_store=store,
        preparation_project_materializer=EchoMindPreparationProjectMaterializer(
            tmp_path / "conversations"
        ),
    )
    await runtime.create_conversation(
        SessionSpec(
            conversation_id=owner.conversation_id,
            user_id=owner.user_id,
            profile_id=owner.profile_id,
            workspace_root=str(tmp_path / "conversations"),
            preparation_policy=policy,
            tool_allowlist=("read",),
            development_tool_allowlist=("read", "get_project_graph"),
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
    try:
        receipt = await runtime.submit_command(
            owner.conversation_id,
            CreateProjectCommand(
                command_id="create-project",
                expected_preparation_revision=store.get(owner).revision,
            ),
            context,
        )
        assert receipt.response["initial_run_id"] is None
        assert store.executions.list(owner) == []

        project_id = receipt.response["project_id"]
        assert isinstance(project_id, str)
        project_store = ProjectStore.for_conversations_dir(tmp_path / "conversations")
        graph = project_store.get_graph(project_id, owner_user_id=owner.user_id)
        assert graph.project_id == project_id
        assert {node.id for node in graph.nodes} == {"dataset"}
        dataset_node = graph.nodes[0]
        assert dataset_node.execution_role == "dataset"
        assert "execution_role" not in dataset_node.config
        assert "sample_validation" not in dataset_node.config

        graph_path = tmp_path / "projects" / str(UUID(project_id)) / "graph.json"
        graph_path.unlink()
        repaired = await runtime.get_project(owner.conversation_id, context)
        assert repaired is not None and repaired.project_id == project_id
        assert graph_path.is_file()
        assert (
            project_store.get_graph(project_id, owner_user_id=owner.user_id).project_id
            == project_id
        )
    finally:
        await runtime.close()


@pytest.mark.parametrize("explicit_entry", [True, False])
async def test_create_project_enqueues_project_created_manager_turn_once(
    preparation_setup, ready_request, tmp_path, explicit_entry
):
    store, _, _ = preparation_setup
    owner = PlanOwner(
        user_id="42",
        conversation_id="project-created-turn",
        profile_id="prepared",
    )
    policy = PreparationPolicy(
        checklist_id="project-created-turn",
        version=1,
        allowed_roles=("eval",),
        create_project_starts_entry_task=False,
        fields=(
            ChecklistField(key="dataset", label="Dataset"),
            ChecklistField(key="mapping", label="Mapping"),
        ),
    )
    store.initialize(owner, policy)
    payload = ready_request if explicit_entry else {**ready_request, "entry_task": None}
    ready = make_ready(store, owner, payload)
    adapter = FakeAdapter()
    execution_store = SQLExperimentExecutionStore(store.engine)
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        adapter,
        experiment_plan_store=SQLExperimentPlanStore(store.engine),
        experiment_execution_store=execution_store,
        execution_roles=("eval",),
        preparation_store=store,
        preparation_project_materializer=EchoMindPreparationProjectMaterializer(
            tmp_path / "conversations"
        ),
    )
    context = RequestContext(user_id=owner.user_id, profile_id=owner.profile_id)
    await runtime.create_conversation(
        SessionSpec(
            conversation_id=owner.conversation_id,
            user_id=owner.user_id,
            profile_id=owner.profile_id,
            workspace_root=str(tmp_path / "conversations"),
            preparation_policy=policy,
            tool_allowlist=("read",),
            development_tool_allowlist=("read", "get_experiment_context"),
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
    command = CreateProjectCommand(
        command_id="create-project",
        expected_preparation_revision=ready.revision,
    )
    try:
        receipt = await runtime.submit_command(owner.conversation_id, command, context)
        project_id = receipt.response["project_id"]
        assert isinstance(project_id, str)
        assert receipt.response["initial_run_id"] is None

        with store.engine.connect() as conn:
            rows = (
                conn.execute(sa.select(work).where(work.c.kind == "manager_turn"))
                .mappings()
                .all()
            )
        assert len(rows) == 1
        payload = rows[0]["payload"]
        notice_id = "project-created:" + project_id
        assert payload["kind"] == "project_created"
        assert payload["notice_id"] == notice_id
        assert payload["result"] == {
            "project_id": project_id,
            "graph_id": project_id,
            "node_id": "evaluation",
            "execution_role": "eval",
            "request_id": notice_id,
        }

        assert (
            await runtime.submit_command(owner.conversation_id, command, context)
            == receipt
        )
        with store.engine.connect() as conn:
            assert (
                conn.scalar(
                    sa.select(sa.func.count())
                    .select_from(work)
                    .where(work.c.kind == "manager_turn")
                )
                == 1
            )

        await runtime.start_execution_events()
        await _wait_for(lambda: adapter.external_task_notifications)
        _, notification, _ = adapter.external_task_notifications[0]
        assert notification.task_id == notice_id
        assert notification.kind == "project_created"
        assert notification.visible_text is None
        assert "open_project_stage" in notification.hidden_text
        assert "Never create a Dataset pipeline" in notification.hidden_text
        assert '"node_id": "evaluation"' in notification.hidden_text

        adapter.emit(
            owner.conversation_id,
            "status.changed",
            {"status": "finished"},
            run_id=f"callback:{notice_id}:succeeded",
            event_id="project-created-turn-finished",
        )

        def manager_turn_completed() -> bool:
            with store.engine.connect() as conn:
                return (
                    conn.scalar(
                        sa.select(work.c.state).where(work.c.kind == "manager_turn")
                    )
                    == "completed"
                )

        await _wait_for(manager_turn_completed)
    finally:
        await runtime.close()


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
        experiment_plan_store=SQLExperimentPlanStore(store.engine),
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
        assert denied.status_code == 404
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


def test_stage_mutation_and_delegation_preserve_frozen_bindings(
    preparation_setup, ready_request
):
    from pyromind_runtime.application.experiment_execution import ExperimentExecutions
    from pyromind_runtime.domain.experiment_execution import RequestExperimentTask

    store, owner, _ = preparation_setup
    ready = make_ready(store, owner, ready_request)
    project = store.activate(owner, ready.revision, "create")
    plans = SQLExperimentPlanStore(store.engine)
    view = plans.context(owner)
    with pytest.raises(ValueError, match="unfinished"):
        plans.update_context(
            owner,
            UpdateExperimentContext(
                context=view.context.model_copy(update={"stages": (), "tasks": ()}),
                expected_sequence=view.sequence,
            ),
        )
    renamed = view.context.model_copy(
        update={
            "stages": (view.context.stages[0].model_copy(update={"name": "Renamed"}),)
        }
    )
    plans.update_context(
        owner, UpdateExperimentContext(context=renamed, expected_sequence=view.sequence)
    )
    assert (
        store.executions.list(owner)[0]["task_spec"]["stage_snapshot"]["name"] == "Eval"
    )
    service = ExperimentExecutions(plans, store.executions)
    request = RequestExperimentTask.model_validate(
        {**ready_request["entry_task"], "request_id": "second"}
    )
    second = service.request(owner, request, ("eval",), True)
    assert (
        second["project_id"] == project.project_id
        and second["task_spec"]["plan_sequence"] == 2
    )
    renamed_again = renamed.model_copy(
        update={
            "stages": (renamed.stages[0].model_copy(update={"name": "Third name"}),)
        }
    )
    plans.update_context(
        owner, UpdateExperimentContext(context=renamed_again, expected_sequence=2)
    )
    assert service.request(owner, request, ("eval",), True)["id"] == second["id"]
    for invalid in (
        request.model_copy(update={"stage_id": None}),
        request.model_copy(update={"role_id": "train"}),
    ):
        with pytest.raises((ValueError, PermissionError)):
            service.request(owner, invalid, ("eval",), True)


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
        experiment_plan_store=SQLExperimentPlanStore(store.engine),
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
        inspected = await runtime.experiment_plan_call(
            owner.conversation_id, "inspect_resource", {"path": "/eval.jsonl"}, context
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
        assert reference in view.evidence and "message:initial-message" in view.evidence
        assert "truncated=True" in view.evidence[reference]
        assert view.evidence["operation:count-rows"] == "terminal"
        assert "operation:failed-probe" not in view.evidence
        facts = {
            key: {**fact, "evidence_refs": [reference, "operation:count-rows"]}
            for key, fact in ready_request["facts"].items()
        }
        request = UpdatePreparation.model_validate(
            {"expected_revision": view.revision, "facts": facts}
        )
        result = await runtime.experiment_plan_call(
            owner.conversation_id,
            "update_preparation",
            request.model_dump(mode="json"),
            context,
        )
        assert result["phase"] == "planning"
    finally:
        await runtime.close()
        assert store.project(owner) is None
        assert inspector.inspect.call_args.args[1].cookie == context.cookie
        assert "private" not in str(store.get(owner).model_dump())


async def test_storage_inspection_failure_does_not_use_plan_error_or_sandbox(
    preparation_setup, tmp_path
):
    store, owner, policy = preparation_setup
    from unittest.mock import Mock

    inspector = Mock()
    inspector.inspect.side_effect = EvaluationStorageUnavailableError(
        "storage auth rejected"
    )
    adapter = FakeAdapter()
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        adapter,
        experiment_plan_store=SQLExperimentPlanStore(store.engine),
        preparation_store=store,
        resource_inspector=inspector,
    )
    context = RequestContext(
        user_id=owner.user_id,
        profile_id=owner.profile_id,
        cookie="auth_token=expired",
    )
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id=owner.conversation_id,
                user_id=owner.user_id,
                profile_id=owner.profile_id,
                workspace_root=str(tmp_path / "conversations"),
                tool_allowlist=("inspect_resource",),
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
        before = store.get(owner)
        created_sessions = len(adapter.created_specs)

        with pytest.raises(ProductRuntimeError) as error:
            await runtime.experiment_plan_call(
                owner.conversation_id,
                "inspect_resource",
                {"path": "/datasets"},
                context,
            )

        assert error.value.code == "storage_unavailable"
        assert "Plan database" not in error.value.message
        assert len(adapter.created_specs) == created_sessions
        assert store.get(owner).revision == before.revision
    finally:
        await runtime.close()


def test_report_work_keeps_stage_active_until_report_finishes(
    preparation_setup, ready_request
):
    store, owner, _ = preparation_setup
    ready = make_ready(store, owner, ready_request)
    project = store.activate(owner, ready.revision, "create")
    with store.engine.begin() as connection:
        connection.execute(
            runs.update()
            .where(runs.c.id == project.initial_run_id)
            .values(state="succeeded", report_state="running", active_key=None)
        )
    plans = SQLExperimentPlanStore(store.engine)
    current = plans.context(owner)
    request = UpdateExperimentContext(
        context=current.context.model_copy(update={"stages": (), "tasks": ()}),
        expected_sequence=current.sequence,
    )
    with pytest.raises(ValueError, match="unfinished"):
        plans.update_context(owner, request)
    with store.engine.begin() as connection:
        connection.execute(
            runs.update()
            .where(runs.c.id == project.initial_run_id)
            .values(report_state="ready")
        )
    plans.update_context(owner, request)
    assert (
        store.executions.list(owner)[0]["task_spec"]["stage_snapshot"]["id"]
        == "evaluation"
    )


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


def test_plan_submission_reuses_collected_facts_and_resolves_entry_resources(
    preparation_setup, ready_request
):
    from pyromind_runtime.domain.preparation import SetPreparationPlan

    store, owner, _ = preparation_setup
    service = Preparations(store)
    collected = service.update(
        owner,
        UpdatePreparation(expected_revision=0, facts=ready_request["facts"]),
        {"message:request"},
    )
    ready = service.set_plan(
        owner,
        SetPreparationPlan.model_validate(
            {
                "expected_revision": collected.revision,
                "plan": ready_request["plan"],
                "entry_task_id": "first",
                "role_id": "eval",
            }
        ),
    )
    assert ready.phase == "ready" and ready.facts == collected.facts
    assert ready.entry_task is not None and ready.entry_task.task.inputs == {
        "dataset": "/eval.jsonl"
    }
    assert ready.entry_task.task.name == "Evaluate"
    assert store.project(owner) is None


def test_plan_retry_requires_complete_arguments_and_stage_edges(
    preparation_setup, ready_request
):
    from pyromind_runtime.domain.preparation import SetPreparationPlan

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
    with pytest.raises(ValueError, match="entry_task_id and role_id"):
        Preparations(store).set_plan(
            owner,
            SetPreparationPlan.model_validate(
                {key: value for key, value in arguments.items() if key != "role_id"}
            ),
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
        experiment_plan_store=SQLExperimentPlanStore(preparations.engine),
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
        development = (*tools, "get_experiment_context")
        configuration = SessionPolicy(
            configuration_version=7,
            preparation_policy=policy.model_copy(update={"version": 2}),
            tool_allowlist=tools,
            development_tool_allowlist=development,
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
            assert adapter.session_policies[-1][1].product_instructions is None
            assert store.preparation_policy() == policy
            assert store.development_policy().tool_allowlist == development
        assert preparations.get(owner) == before
        assert preparations.project(owner) == project
    finally:
        await runtime.close()


def test_workflow_catalog_rejects_generated_node_and_task_types(
    preparation_setup, ready_request
):
    from pyromind_runtime.domain.preparation import SetPreparationPlan

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
    for invalid in (
        request.plan.model_copy(
            update={
                "stages": (
                    request.plan.stages[0].model_copy(update={"kind": "invented"}),
                )
            }
        ),
        request.plan.model_copy(
            update={
                "tasks": (
                    request.plan.tasks[0].model_copy(update={"kind": "invented"}),
                )
            }
        ),
        request.plan.model_copy(
            update={
                "stages": (
                    request.plan.stages[0].model_copy(update={"kind": "future"}),
                )
            }
        ),
    ):
        with pytest.raises(ValueError):
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
    assert set(nodes) == {"dataset", "train", "inference", "evaluator"}
    assert {task.id for task in nodes["dataset"].tasks} == {"prepare_dataset"}
    assert {task.id for task in nodes["train"].tasks} == {"train_model"}
    assert {task.id for task in nodes["inference"].tasks} == {"run_inference"}
    assert {task.id for task in nodes["evaluator"].tasks} == {"evaluate"}
    assert nodes["dataset"].roles == ("dataset",)
    assert nodes["evaluator"].roles == ("eval",)
    context = ExperimentContext.model_validate(
        {
            "goal": "Dataset",
            "stages": [
                {"id": "arbitrary-instance", "name": "Renamable", "kind": "dataset"}
            ],
            "tasks": [
                {
                    "id": "prepare",
                    "kind": "prepare_dataset",
                    "name": "Prepare data",
                    "stage_id": "arbitrary-instance",
                }
            ],
        }
    )
    validate_workflow_catalog(policy, context)
    assert context.stages[0].name == "Renamable"
    assert context.stages[0].kind == "dataset"
    combined = context.model_copy(
        update={
            "tasks": (
                *context.tasks,
                context.tasks[0].model_copy(
                    update={"id": "prepare-again", "kind": "prepare_dataset"}
                ),
            )
        }
    )
    validate_workflow_catalog(policy, combined)
    assert len(combined.stages) == 1 and len(combined.tasks) == 2
    limited_policy = policy.model_copy(
        update={
            "workflow_nodes": tuple(
                node.model_copy(update={"max_instances": 1})
                if node.id == "dataset"
                else node
                for node in policy.workflow_nodes
            )
        }
    )
    with pytest.raises(ValueError, match="instance limit"):
        validate_workflow_catalog(
            limited_policy,
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
