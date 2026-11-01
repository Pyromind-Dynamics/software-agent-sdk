import hashlib
from dataclasses import replace

import httpx
import pytest
from pyromind_agent_server.api.router import create_product_router
from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_plans import PlanOwner
from pyromind_runtime.domain.session_policy import SessionPolicy

from .fake_adapter import FakeAdapter
from .test_product_api import _app
from .test_standalone_experiments import experiment, request, claim  # noqa: F401


async def test_new_profile_has_no_graph_gate_and_artifacts_remain_isolated(tmp_path, experiment):
    service, _, engine = experiment
    adapter = FakeAdapter("pi")
    runtime = ConversationRuntime(tmp_path / "conversations", {"pi": adapter}, default_harness_id="pi",
        experiment_plan_store=service.plans, experiment_execution_store=service.store, execution_roles=("eval",))
    app = _app(tmp_path, runtime)
    app.include_router(create_product_router(ECHOMIND_PRODUCT_PROFILE))
    base = ECHOMIND_PRODUCT_PROFILE.route_prefix
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", headers={"x-pyromind-debug-user-id": "42"}) as client:
            created = await client.post(base, json={"llm": {"model": "test"}})
            assert created.status_code == 201
            identity = created.json()["conversation_id"]
            owner = PlanOwner(user_id="42", conversation_id=identity, profile_id=ECHOMIND_PRODUCT_PROFILE.profile_id)
            context = RequestContext(user_id="42", profile_id=owner.profile_id)
            spec = adapter.created_specs[0]
            assert set(spec.tool_allowlist) == {"read_skill", "get_experiment_context", "update_experiment_context", "request_experiment_task", "get_experiment_execution", "control_experiment_task"}
            assert "df_submit_evaluation" in spec.delegated_roles[0].tools
            assert "update_experiment_context" not in spec.delegated_roles[0].tools
            run = await runtime.request_experiment_task(identity, request(intent="prepare"), context)
            assert run["revision"] is None and service.plans.view(owner).current is None
            snapshot = await client.get(f"{base}/{identity}/snapshot")
            assert snapshot.status_code == 200 and len(snapshot.json()["experiment_executions"]) == 1
            scope = claim(service, owner, run)
            body = b"Prepared without GPU execution."
            path = f"public_data/experiment-runs/{run['id']}/report.md"
            destination = tmp_path / "conversations" / identity / path
            destination.parent.mkdir(parents=True)
            destination.write_bytes(body)
            service.store.record_result(owner, run["id"], scope, {"summary": "Prepared", "report_uri": path}, {"uri": path, "sha256": hashlib.sha256(body).hexdigest()})
            route = f"{base}/{identity}/experiment-executions/{run['id']}/artifacts/agent_report"
            assert (await client.get(route)).json()["content"] == body.decode()
            assert (await client.get(route, headers={"x-pyromind-debug-user-id": "99"})).status_code == 404
            assert (await client.get(route.replace(base, "/api/v2/pyromind/conversations"))).status_code == 404
            destination.write_text("changed")
            assert (await client.get(route)).status_code == 409
            service.store.queue.finish(scope["work_id"], scope["generation"], {})
            await runtime.flush_execution_events()
            await runtime.flush_execution_events()
            items = service.store.queue.list(run["owner"])
            assert len([item for item in items if item["kind"] == "manager_turn"]) == 1
            with pytest.raises(PermissionError):
                await runtime.experiment_plan_call(identity, "create_experiment_plan", {}, context)
    finally:
        await runtime.close()


def test_upgrade_retires_unstarted_legacy_work_but_preserves_old_proposal(experiment):
    service, owner, _ = experiment
    context = service.plans.context(owner, create=True)
    old = service.store.request(owner, context.project_id, 4, "baseline", "eval", "legacy")
    service.store.propose(owner, old["id"], {"configuration": {"model": "old"}})
    assert service.store.retire_legacy_work(owner)
    assert service.store.list(owner)[0]["state"] == "awaiting_confirmation"
    assert service.store.queue.claim() is None
    new = service.request(owner, request(), ("eval",))
    scope = claim(service, owner, new)
    assert not service.store.retire_legacy_work(owner)
    service.store.queue.finish(scope["work_id"], scope["generation"], {})
