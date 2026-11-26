"""Route direct SDK tasks to conversations and Projects from trusted callbacks."""

import asyncio
import json
from dataclasses import replace

import pytest

from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.preparation import Preparations
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.preparation import PreparationPolicy
from pyromind_runtime.domain.session_policy import SessionPolicy
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.infrastructure.file_product_store import FileProductStore
from pyromind_runtime.ports.harness import SessionSpec

from .fake_adapter import FakeAdapter
from .test_preparation import (
    preparation_setup as preparation_setup,
    ready_request as ready_request,
)
from .test_workflow_execution import _runtime


@pytest.fixture
async def sdk_project_runtime(preparation_setup, ready_request, tmp_path):
    preparations, owner, original = preparation_setup
    policy = PreparationPolicy(
        workflow_nodes=tuple(
            node.model_copy(update={"roles": ()}) for node in original.workflow_nodes
        )
    )
    Preparations(preparations).set_plan(owner, ready_request, policy, 0)
    project = preparations.activate(owner, 1, "create", cluster="test-cluster")
    runtime = ConversationRuntime(
        tmp_path / "conversations", FakeAdapter(), preparation_store=preparations
    )
    context = RequestContext(
        user_id=owner.user_id, profile_id=owner.profile_id, x_cluster="other-cluster"
    )
    await runtime.create_conversation(
        SessionSpec(
            conversation_id=owner.conversation_id,
            user_id=owner.user_id,
            profile_id=owner.profile_id,
            workspace_root=str(tmp_path / "conversations" / owner.conversation_id),
            preparation_policy=policy,
            delegated_roles=(),
        ),
        context,
    )
    try:
        yield runtime, context, project
    finally:
        await runtime.close()


def correlation(environment, kind="workflow_run"):
    return TaskCorrelation.model_validate_json(
        json.loads(environment["PYROMIND_TASK_CONTEXTS"])[kind]
    )


async def test_sdk_context_is_stable_without_a_submission_ledger_and_survives_restart(
    tmp_path,
):
    runtime, adapter, _, context = await _runtime(tmp_path)
    scope = "workflow-conversation"
    store = FileProductStore(tmp_path / scope)
    registry = WorkflowExternalTaskRegistry(tmp_path)
    runtime._external_tasks = registry
    before = store.load_snapshot().through_seq
    environment = await runtime.platform_sdk_context(scope, context)
    address = correlation(environment)
    assert address.signature is not None and address.execution is None
    assert json.loads(environment["PYROMIND_PROJECT_EXECUTIONS"]) == []
    assert store.load_snapshot().through_seq == before
    assert store.load_snapshot().external_tasks == ()
    assert not store.commands_path.read_text()
    assert not list((tmp_path / scope).rglob("*correlations.key"))
    try:
        await runtime.deliver_external_task_status(
            scope,
            task_id="sdk-direct",
            status="Pending",
            correlation=address,
            trusted_platform=True,
        )
        assert not adapter.external_task_notifications
        for status in ("Running", "Pending"):
            await runtime.deliver_external_task_status(
                scope,
                task_id="sdk-direct",
                status=status,
                correlation=address,
                trusted_platform=True,
            )
        assert store.load_snapshot().external_tasks[0].status == "running"
        for _ in range(2):
            await runtime.deliver_external_task_status(
                scope,
                task_id="sdk-direct",
                status="Succeeded",
                correlation=address,
                trusted_platform=True,
            )
        assert len(adapter.external_task_notifications) == 1
        await runtime.deliver_external_task_status(
            scope,
            task_id="sdk-direct",
            status="Running",
            correlation=address,
            trusted_platform=True,
        )
        assert store.load_snapshot().external_tasks[0].status == "succeeded"
        registered = registry.resolve(scope, "sdk-direct")
        assert registered is not None and registered["status"] == "succeeded"
    finally:
        await runtime.close()
    recovered = ConversationRuntime(tmp_path, FakeAdapter())
    try:
        refreshed = await recovered.platform_sdk_context(scope, context)
        assert (
            refreshed["PYROMIND_TASK_CONTEXTS"] == environment["PYROMIND_TASK_CONTEXTS"]
        )
        assert (
            json.loads(refreshed["PYROMIND_TASK_ASSOCIATIONS"])[0]["task_id"]
            == "sdk-direct"
        )
        assert store.verify_task_correlation(address)
    finally:
        await recovered.close()


async def test_untrusted_early_observation_cannot_win_over_the_platform_result(
    tmp_path,
):
    runtime, adapter, _, context = await _runtime(tmp_path)
    scope = "workflow-conversation"
    address = correlation(await runtime.platform_sdk_context(scope, context))
    try:
        observed = await runtime.deliver_external_task_status(
            scope, task_id="sdk-task", status="Succeeded", correlation=address
        )
        assert observed.type == "external_task.observed"
        assert not adapter.external_task_notifications
        await runtime.deliver_external_task_status(
            scope,
            task_id="sdk-task",
            status="Failed",
            correlation=address,
            trusted_platform=True,
        )
        await asyncio.gather(*tuple(runtime._observation_replays))
        task = FileProductStore(tmp_path / scope).load_snapshot().external_tasks[0]
        assert task.status == "failed"
        assert len(adapter.external_task_notifications) == 1
        with pytest.raises(PermissionError, match="trusted platform"):
            await runtime.deliver_external_task_status(
                scope, task_id="sdk-task", status="Succeeded"
            )
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "change", [{"cluster": "foreign"}, {"task_kind": "workflow_debug"}]
)
async def test_tampered_sdk_addresses_cannot_adopt_unknown_tasks(tmp_path, change):
    runtime, _, _, context = await _runtime(tmp_path)
    scope = "workflow-conversation"
    address = correlation(await runtime.platform_sdk_context(scope, context))
    try:
        with pytest.raises(PermissionError, match="signature"):
            await runtime.deliver_external_task_status(
                scope,
                task_id="forged",
                status="Succeeded",
                correlation=address.model_copy(update=change),
                trusted_platform=True,
            )
        assert FileProductStore(tmp_path / scope).load_snapshot().external_tasks == ()
        with pytest.raises(PermissionError):
            await runtime.platform_sdk_context(scope, replace(context, user_id="other"))
    finally:
        await runtime.close()


async def test_project_callbacks_bind_all_shards_and_preserve_a_newer_attempt(
    sdk_project_runtime,
):
    runtime, context, project = sdk_project_runtime
    scope = project.conversation_id
    environment = await runtime.platform_sdk_context(scope, context)
    execution = json.loads(environment["PYROMIND_PROJECT_EXECUTIONS"])[0]
    first = TaskCorrelation.model_validate_json(
        execution["out_ids"]["data_preparation"]
    )
    assert first.cluster == "test-cluster"
    assert first.execution is not None and first.execution.attempt == 1
    for identity in ("shard-1", "shard-2"):
        await runtime.deliver_external_task_status(
            scope,
            task_id=identity,
            status="Pending",
            correlation=first,
            trusted_platform=True,
        )
    await runtime.bind_plan_submission(scope, first.execution.task_id, 2, context)
    second_execution = json.loads(
        (await runtime.platform_sdk_context(scope, context))[
            "PYROMIND_PROJECT_EXECUTIONS"
        ]
    )[0]
    second = TaskCorrelation.model_validate_json(
        second_execution["out_ids"]["data_preparation"]
    )
    assert second.execution is not None and second.execution.attempt == 2
    await runtime.deliver_external_task_status(
        scope,
        task_id="new-attempt",
        status="Pending",
        correlation=second,
        trusted_platform=True,
    )
    for identity in ("shard-1", "shard-2"):
        await runtime.deliver_external_task_status(
            scope,
            task_id=identity,
            status="Succeeded",
            correlation=first,
            trusted_platform=True,
        )
    snapshot = await runtime.get_snapshot(scope, context)
    runs = {run.plan_attempt: run for run in snapshot.experiment_executions or ()}
    assert runs[1].platform_task_id == "shard-1"
    assert runs[1].platform_task_ids == ("shard-1", "shard-2")
    assert runs[1].state == "waiting_analysis"
    assert runs[2].platform_task_ids == ("new-attempt",)
    assert runs[2].state == "waiting_platform"
    assert all(task.run_id is None for task in snapshot.external_tasks)
    associations = json.loads(
        (await runtime.platform_sdk_context(scope, context))[
            "PYROMIND_TASK_ASSOCIATIONS"
        ]
    )
    assert {
        row["task_id"] for row in associations if row["run_id"] == runs[1].run_id
    } == {"shard-1", "shard-2"}
    store = FileProductStore(runtime.conversation_root / scope)
    current = next(
        run
        for run in store.load_snapshot().experiment_executions or ()
        if run.plan_attempt == 2
    )
    runtime._record_execution(store, current.model_copy(update={"state": "succeeded"}))
    stale = current.model_copy(update={"platform_task_ids": ("another-shard",)})
    runtime._record_execution(store, stale, binding_only=True)
    saved = next(
        run
        for run in store.load_snapshot().experiment_executions or ()
        if run.plan_attempt == 2
    )
    assert saved.state == "succeeded"
    assert saved.platform_task_ids == ("new-attempt", "another-shard")


async def test_sdk_context_rejects_historical_role_orchestration(sdk_project_runtime):
    runtime, context, project = sdk_project_runtime
    store = FileProductStore(runtime.conversation_root / project.conversation_id)
    policy = store.preparation_policy()
    assert policy is not None
    store.apply_configuration(
        SessionPolicy(
            preparation_policy=policy.model_copy(update={"allowed_roles": ("eval",)})
        )
    )
    with pytest.raises(PermissionError, match="single Main"):
        await runtime.platform_sdk_context(project.conversation_id, context)


async def test_rejected_project_fork_cannot_change_the_sdk_execution_cluster(
    sdk_project_runtime,
):
    runtime, context, project = sdk_project_runtime
    scope = project.conversation_id
    await runtime._read_store(scope, context)
    before = runtime._active[scope].context
    assert before.x_cluster == project.cluster
    with pytest.raises(ValueError, match="cannot be forked"):
        await runtime.fork_conversation(
            scope,
            event_id="unused",
            title=None,
            context=replace(context, x_cluster="foreign-cluster"),
        )
    assert runtime._active[scope].context == before
    signed = correlation(await runtime.platform_sdk_context(scope, before))
    assert signed.cluster == before.x_cluster


async def test_all_issued_sdk_task_kinds_can_join_the_persisted_task_contract(tmp_path):
    runtime, _, _, context = await _runtime(tmp_path)
    scope = "workflow-conversation"
    addresses = json.loads(
        (await runtime.platform_sdk_context(scope, context))["PYROMIND_TASK_CONTEXTS"]
    )
    try:
        for kind, address in addresses.items():
            await runtime.deliver_external_task_status(
                scope,
                task_id="sdk-" + kind,
                status="Pending",
                correlation=TaskCorrelation.model_validate_json(address),
                trusted_platform=True,
            )
        tasks = FileProductStore(tmp_path / scope).load_snapshot().external_tasks
        assert len(tasks) == len(addresses)
        for task in tasks:
            assert task.correlation is not None
            expected = (
                "data_preparation"
                if task.correlation.task_kind == "evaluation"
                else task.correlation.task_kind
            )
            assert task.kind == expected
    finally:
        await runtime.close()
