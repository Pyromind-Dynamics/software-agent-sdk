from dataclasses import replace

import pytest
from pyromind_agent_server.profiles import (
    DEFAULT_PRODUCT_PROFILE,
    ECHOMIND_PRODUCT_PROFILE,
)
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.experiment_plans import (
    ExperimentContext,
    UpdateExperimentContext,
)
from pyromind_runtime.domain.session_policy import SessionPolicy
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec

from . import (
    test_experiment_plans as plan_fixtures,
    test_standalone_experiments as execution_fixtures,
)
from .fake_adapter import FakeAdapter
from .test_conversation_runtime import _wait_for_status


plan_payload = plan_fixtures.plan_payload
experiment = execution_fixtures.experiment


def test_reference_plan_preserves_task_resource_and_upstream_references(
    experiment, plan_payload
):
    service, owner, _ = experiment
    plan_payload["edges"].append({"source": "eval", "target": "init"})
    context = ExperimentContext.model_validate(
        {
            key: plan_payload[key]
            for key in ("goal", "resources", "stages", "tasks", "edges")
        }
    )
    saved = service.plans.update_context(
        owner, UpdateExperimentContext(context=context)
    )
    restored = service.plans.context(owner)
    assert restored == saved
    assert restored.context.tasks[0].inputs[0].resource_id == "bench"
    assert restored.context.tasks[1].inputs[0].output_name == "baseline_report"
    assert service.plans.view(owner).current is None
    assert service.store.list(owner) == []
    assert ExperimentContext.model_validate({"goal": "Old context"}).tasks == ()
    with pytest.raises(ValueError, match="context changed"):
        service.plans.update_context(owner, UpdateExperimentContext(context=context))


@pytest.mark.parametrize(
    "invalid", ["duplicate", "stage", "resource", "output", "placeholder"]
)
def test_reference_plan_rejects_invalid_task_contracts(plan_payload, invalid):
    if invalid == "duplicate":
        plan_payload["tasks"].append(plan_payload["tasks"][0])
    elif invalid == "stage":
        plan_payload["tasks"][0]["stage_id"] = "absent"
    elif invalid == "resource":
        plan_payload["tasks"][0]["inputs"][0]["resource_id"] = "absent"
    elif invalid == "output":
        plan_payload["tasks"][1]["inputs"][0]["output_name"] = "absent"
    else:
        plan_payload["stages"][-1]["enabled"] = True
    with pytest.raises(ValueError):
        ExperimentContext.model_validate(
            {
                key: plan_payload[key]
                for key in ("goal", "resources", "stages", "tasks", "edges")
            }
        )


async def test_idle_upgrade_restores_manager_capabilities_without_executing(
    tmp_path, experiment
):
    profile = ECHOMIND_PRODUCT_PROFILE
    context = RequestContext(user_id="42", profile_id=profile.profile_id)
    root = tmp_path / "conversations"
    adapter = FakeAdapter("pi")
    service, _, _ = experiment
    runtime = ConversationRuntime(
        root,
        {"pi": adapter},
        default_harness_id="pi",
        experiment_plan_store=service.plans,
        experiment_execution_store=service.store,
        execution_roles=("eval",),
    )
    policy = SessionPolicy(
        configuration_version=profile.configuration_version,
        tool_allowlist=profile.tool_allowlist,
        skill_allowlist=profile.skill_allowlist,
        product_instructions=profile.instructions,
        delegated_roles=tuple(
            role.model_dump(mode="json") for role in profile.delegated_roles or ()
        ),
        managed_delegation=profile.managed_delegation,
        command_allowlist=profile.command_allowlist,
    )
    try:
        await runtime.create_conversation(
            SessionSpec(
                conversation_id="old-manager",
                user_id="42",
                profile_id=profile.profile_id,
                workspace_root=str(root / "old-manager"),
                tool_allowlist=("read_skill", "get_experiment_context"),
                skill_allowlist=("experiment-planning",),
                development_tool_allowlist=("read_skill", "get_project_graph"),
                development_skill_allowlist=("experiment-planning",),
            ),
            context,
        )
        store = FileProductStore(root / "old-manager")
        adapter.emit(
            "old-manager", "status.changed", {"status": "running"}, event_id="busy"
        )
        await _wait_for_status(store, "running")
        assert not await runtime.apply_session_configuration(
            "old-manager", policy, context
        )
        assert not adapter.session_policies
        adapter.emit(
            "old-manager", "status.changed", {"status": "idle"}, event_id="idle"
        )
        await _wait_for_status(store, "idle")
        assert await runtime.apply_session_configuration("old-manager", policy, context)
        assert store.configuration_version() == 4
        assert store.access_policy()["tool_allowlist"] == profile.tool_allowlist
        assert store.access_policy()["skill_allowlist"] == profile.skill_allowlist
        activated = store.activate_development_policy()
        assert activated.tool_allowlist == ("read_skill", "get_project_graph")
        assert activated.skill_allowlist == ("experiment-planning",)
        assert len(adapter.session_policies) == 1
        assert await runtime.apply_session_configuration("old-manager", policy, context)
        assert len(adapter.session_policies) == 1
        assert not adapter.sent
        assert (await runtime.get_snapshot("old-manager", context)).status == "idle"
        with pytest.raises(PermissionError):
            await runtime.apply_session_configuration(
                "old-manager", policy, replace(context, user_id="99")
            )
        with pytest.raises(PermissionError):
            await runtime.apply_session_configuration(
                "old-manager", policy, replace(context, profile_id="other")
            )
        assert DEFAULT_PRODUCT_PROFILE.tool_allowlist is None
        assert DEFAULT_PRODUCT_PROFILE.delegated_roles is None
        assert not DEFAULT_PRODUCT_PROFILE.managed_delegation
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "path", ["relative", "dot", "absolute", "escape", "another-task"]
)
async def test_report_registration_normalizes_only_task_local_paths(experiment, path):
    service, owner, _ = experiment
    run = service.request(
        owner, execution_fixtures.request(intent="prepare"), ("eval",)
    )
    scope = execution_fixtures.claim(service, owner, run)
    canonical = f"public_data/experiment-runs/{run['id']}/public_data/report.md"
    report_uri = {
        "relative": canonical,
        "dot": "./" + canonical,
        "absolute": "/" + canonical,
        "escape": canonical.replace("public_data/report", "../report"),
        "another-task": canonical.replace(run["id"], "other"),
    }[path]
    reads = []

    async def read(_owner, uri):
        reads.append(uri)
        return b"Prepared report"

    async def execute(*_):
        pytest.fail("report registration must not submit platform work")

    call = service.role_call(
        owner,
        "eval",
        "record_experiment_result",
        {
            "run_id": run["id"],
            "summary": "Prepared",
            "report_uri": report_uri,
        },
        scope,
        execute,
        read,
    )
    if path in {"relative", "dot"}:
        await call
        assert reads == [canonical]
        assert service.list(owner)[0].report_uri == canonical
    else:
        with pytest.raises(PermissionError, match="workspace"):
            await call
        assert not reads
