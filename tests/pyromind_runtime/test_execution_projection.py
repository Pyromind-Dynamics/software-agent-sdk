from dataclasses import replace

import httpx
import pytest
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.profiles import ConversationProfile
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.experiment_plans import ExperimentPlans
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_execution import RequestExperimentTask
from pyromind_runtime.domain.experiment_plans import (
    CreateExperimentPlan,
    PlanOwner,
    WorkflowPlan,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
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

from .fake_adapter import FakeAdapter
from .test_product_api import _app


@pytest.fixture
async def execution_runtime(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'execution.sqlite'}")
    metadata.create_all(engine)
    plan_metadata.create_all(engine)
    plans = SQLExperimentPlanStore(engine)
    executions = SQLExperimentExecutionStore(engine)
    adapter = FakeAdapter()
    runtime = ConversationRuntime(
        tmp_path / "conversations",
        adapter,
        experiment_plan_store=plans,
        experiment_execution_store=executions,
        execution_roles=("evaluator",),
    )
    context = RequestContext(user_id="42", profile_id="planning")
    await runtime.create_conversation(
        SessionSpec(
            conversation_id="session",
            user_id="42",
            profile_id="planning",
            workspace_root=str(tmp_path / "conversations/session"),
            tool_allowlist=(
                "get_experiment_plan",
                "request_experiment_task",
                "get_experiment_execution",
            ),
            command_allowlist=(
                "user_message",
                "confirm_experiment_execution",
                "reject_experiment_execution",
            ),
        ),
        context,
    )
    owner = PlanOwner(user_id="42", profile_id="planning", conversation_id="session")
    plan = WorkflowPlan.model_validate(
        {
            "name": "Experiment",
            "goal": "Measure baseline",
            "stages": [{"id": "first", "name": "First stage"}],
            "tasks": [{"id": "free", "stage_id": "first", "name": "Any task name"}],
        }
    )
    draft = ExperimentPlans(plans).propose(owner, CreateExperimentPlan(plan=plan))
    plans.confirm(owner, draft.proposal_id, 0, "confirm-plan")
    yield runtime, context, owner, plans, executions
    await runtime.close()


async def test_execution_snapshot_and_outbox_recover_after_delivery_failure(
    execution_runtime, monkeypatch
):
    runtime, context, owner, plans, executions = execution_runtime
    run = await runtime.request_experiment_task(
        "session",
        RequestExperimentTask(
            task_id="free",
            expected_revision=1,
            role_id="evaluator",
            request_id="prepare",
        ),
        context,
    )
    snapshot = await runtime.get_snapshot("session", context)
    view = snapshot.experiment_executions[0]
    assert view.run_id == run["id"] and view.task_name == "Any task name"
    assert view.stage_id == "first" and view.metrics is None
    original = FileProductStore.append

    def fail(store, event):
        if event.type == "experiment_execution.changed":
            raise OSError("simulated event store outage")
        return original(store, event)

    monkeypatch.setattr(FileProductStore, "append", fail)
    with pytest.raises(OSError):
        await runtime.flush_execution_events()
    assert len(executions.pending_deliveries()) == 1
    monkeypatch.setattr(FileProductStore, "append", original)
    await runtime.flush_execution_events()
    assert executions.pending_deliveries() == []
    before = (await runtime.get_snapshot("session", context)).through_seq
    await runtime.flush_execution_events()
    assert (await runtime.get_snapshot("session", context)).through_seq == before
    with pytest.raises(PermissionError):
        await runtime.get_experiment_executions(
            "session", replace(context, profile_id="other")
        )


async def test_execution_read_api_is_scoped_and_does_not_expose_storage_internals(
    execution_runtime, tmp_path
):
    runtime, context, owner, plans, executions = execution_runtime
    await runtime.request_experiment_task(
        "session",
        RequestExperimentTask(
            task_id="free",
            expected_revision=1,
            role_id="evaluator",
            request_id="prepare",
        ),
        context,
    )
    app = _app(tmp_path, runtime)
    app.include_router(
        create_product_router(
            ConversationProfile(
                route_prefix="/api/v2/plans/conversations", profile_id="planning"
            )
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"x-pyromind-debug-user-id": "42"},
    ) as client:
        response = await client.get(
            "/api/v2/plans/conversations/session/experiment-executions"
        )
        assert response.status_code == 200
        row = response.json()[0]
        assert row["state"] == "queued" and row["metrics"] is None
        assert "owner" not in row and "request_key" not in row
        assert (
            await client.get(
                "/api/v2/pyromind/conversations/session/experiment-executions"
            )
        ).status_code == 404
        foreign = await client.get(
            "/api/v2/plans/conversations/session/experiment-executions",
            headers={"x-pyromind-debug-user-id": "other"},
        )
        assert foreign.status_code == 404
