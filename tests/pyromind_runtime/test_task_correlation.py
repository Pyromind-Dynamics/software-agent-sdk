import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from pyromind_runtime.domain.task_correlation import (
    TaskCorrelation,
    TaskOrigin,
    parse_task_correlation,
)
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from .test_workflow_execution import _runtime


@pytest.mark.parametrize(
    "origin", [TaskOrigin(), TaskOrigin(type="subagent", subagent_id="child-1")]
)
def test_json_return_address_is_flat_and_versioned(origin):
    correlation = TaskCorrelation(
        conversation_id="c", task_kind="workflow_debug", **origin.model_dump()
    )
    assert parse_task_correlation(correlation.to_out_id()) == correlation
    assert '"target"' not in correlation.to_out_id()
    assert ('"subagent_id"' in correlation.to_out_id()) == (origin.type == "subagent")


def test_project_execution_ids_round_trip_without_changing_legacy_addresses():
    correlation = TaskCorrelation(
        conversation_id="c",
        task_kind="workflow_run",
        project_id="project-1",
        stage_run_id="stage-run-1",
        task_run_id="task-run-1",
    )

    assert parse_task_correlation(correlation.to_out_id()) == correlation
    legacy = TaskCorrelation(conversation_id="c", task_kind="workflow_run")
    assert parse_task_correlation(legacy.to_out_id()) == legacy
    assert legacy.project_id is None
    assert legacy.stage_run_id is None
    assert legacy.task_run_id is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        "null",
        "[]",
        "{}",
        "agent1#c",
        "agent1#debug#c",
        '{"version":2,"source":"agent","conversation_id":"c","type":"main","task_kind":"workflow_run"}',
    ],
)
def test_legacy_foreign_and_incomplete_return_addresses_are_rejected(value):
    assert parse_task_correlation(value) is None


def test_only_child_origins_have_instance_ids():
    for fields in (
        {"type": "main", "subagent_id": "child"},
        {"type": "subagent"},
        {"type": "subagent", "subagent_id": ""},
    ):
        with pytest.raises(ValidationError):
            TaskOrigin.model_validate(fields)


def register(runtime, task_id, correlation):
    now = datetime.now(UTC).isoformat()
    runtime.register_external_task(
        "workflow-conversation",
        {
            "task_id": task_id,
            "kind": correlation.task_kind,
            "correlation": correlation.model_dump(mode="json", exclude_none=True),
            "status": "running",
            "submitted_at": now,
            "updated_at": now,
        },
    )


async def test_main_and_same_role_instances_receive_only_their_own_callbacks(tmp_path):
    runtime, adapter, _, _ = await _runtime(tmp_path)
    try:
        origins = [
            TaskOrigin(),
            TaskOrigin(type="subagent", subagent_id="worker-1"),
            TaskOrigin(type="subagent", subagent_id="worker-2"),
        ]
        correlations = [
            TaskCorrelation(
                conversation_id="workflow-conversation",
                task_kind="workflow_run",
                **origin.model_dump(),
            )
            for origin in origins
        ]
        for index, correlation in enumerate(correlations):
            register(runtime, str(index), correlation)
        await asyncio.gather(
            *(
                runtime.deliver_external_task_status(
                    "workflow-conversation",
                    task_id=str(index),
                    status="Succeeded",
                    correlation=correlation,
                )
                for index, correlation in enumerate(correlations)
            )
        )
        assert {
            notice.task_id: TaskCorrelation.model_validate(notice.correlation).origin
            for _, notice, _ in adapter.external_task_notifications
        } == {str(index): origin for index, origin in enumerate(origins)}
        await runtime.deliver_external_task_status(
            "workflow-conversation",
            task_id="1",
            status="Succeeded",
            correlation=correlations[1],
        )
        assert len(adapter.external_task_notifications) == 3
        forged = correlations[1].model_copy(update={"subagent_id": "worker-2"})
        with pytest.raises(PermissionError, match="origin"):
            await runtime.deliver_external_task_status(
                "workflow-conversation",
                task_id="1",
                status="Succeeded",
                correlation=forged,
            )
        with pytest.raises(PermissionError, match="conversation"):
            await runtime.deliver_external_task_status(
                "workflow-conversation",
                task_id="1",
                status="Succeeded",
                correlation=correlations[1].model_copy(
                    update={"conversation_id": "other"}
                ),
            )
    finally:
        await runtime.close()


async def test_early_callback_is_durable_and_replayed_after_registration(tmp_path):
    runtime, adapter, _, _ = await _runtime(tmp_path)
    correlation = TaskCorrelation(
        conversation_id="workflow-conversation",
        task_kind="workflow_debug",
        type="subagent",
        subagent_id="worker",
    )
    try:
        observed = await runtime.deliver_external_task_status(
            "workflow-conversation",
            task_id="early",
            status="Failed",
            error_summary="node failed",
            correlation=correlation,
        )
        assert observed.type == "external_task.observed"
        assert adapter.external_task_notifications == []
        assert any(
            event.type == "external_task.observed"
            for event in FileProductStore(tmp_path / "workflow-conversation").replay()
        )
        register(runtime, "early", correlation)
        await asyncio.gather(*tuple(runtime._observation_replays))
        assert len(adapter.external_task_notifications) == 1
        notice = adapter.external_task_notifications[0][1]
        assert notice.correlation == correlation
        assert notice.error_summary == "node failed"
        assert notice.reset_attempt_budget is False
    finally:
        await runtime.close()


async def test_unavailable_child_stays_pending_until_acknowledged(
    tmp_path, monkeypatch
):
    runtime, adapter, _, _ = await _runtime(tmp_path)
    correlation = TaskCorrelation(
        conversation_id="workflow-conversation",
        task_kind="workflow_run",
        type="subagent",
        subagent_id="worker",
    )
    register(runtime, "deferred", correlation)
    original = adapter.notify_external_task

    async def unavailable(*args):
        return {"accepted": False, "reason": "subagent_unavailable"}

    try:
        monkeypatch.setattr(adapter, "notify_external_task", unavailable)
        await runtime.deliver_external_task_status(
            "workflow-conversation",
            task_id="deferred",
            status="Succeeded",
            correlation=correlation,
        )
        store = FileProductStore(tmp_path / "workflow-conversation")
        assert store.load_snapshot().external_tasks[0].resume_pending
        monkeypatch.setattr(adapter, "notify_external_task", original)
        await runtime._resume_notifications(
            runtime._active["workflow-conversation"], store
        )
        assert store.load_snapshot().external_tasks[0].resume_pending is False
        assert adapter.external_task_notifications[0][1].correlation == correlation
    finally:
        await runtime.close()


async def test_confirmation_keeps_the_child_origin_and_rejects_other_profiles(tmp_path):
    from dataclasses import replace

    from pyromind_runtime.domain.commands import PermissionResponseCommand

    runtime, _, submission, context = await _runtime(tmp_path)
    origin = TaskOrigin(type="subagent", subagent_id="worker")
    try:
        with pytest.raises(PermissionError):
            await runtime.prepare_workflow_execution(
                "workflow-conversation",
                "/workflow.py",
                replace(context, profile_id="another-profile"),
                origin=origin,
            )
        await runtime.prepare_workflow_execution(
            "workflow-conversation",
            "/workflow.py",
            context,
            origin=origin,
            request_id="child-request",
        )
        await runtime.submit_command(
            "workflow-conversation",
            PermissionResponseCommand(
                command_id="approve",
                permission_id="workflow-run:child-request",
                decision="allow_once",
            ),
            context,
        )
        assert submission.correlations[0] is not None
        assert submission.correlations[0].origin == origin
        assert (
            FileProductStore(tmp_path / "workflow-conversation")
            .load_snapshot()
            .external_tasks[0]
            .correlation
            == submission.correlations[0]
        )
    finally:
        await runtime.close()


async def test_suspended_callback_is_saved_and_recipient_recovers_after_restart(
    tmp_path,
):
    from pyromind_runtime.application.conversation_runtime import ConversationRuntime
    from pyromind_runtime.domain.events import ProductEvent

    from .fake_adapter import FakeAdapter

    runtime, adapter, _, context = await _runtime(tmp_path)
    correlation = TaskCorrelation(
        conversation_id="workflow-conversation",
        task_kind="workflow_run",
        type="subagent",
        subagent_id="worker",
    )
    register(runtime, "paused", correlation)
    store = FileProductStore(tmp_path / "workflow-conversation")
    store.append(
        ProductEvent(
            conversation_id="workflow-conversation",
            type="execution.policy.updated",
            payload={"automatic_turns_suspended": True},
        )
    )
    await runtime.deliver_external_task_status(
        "workflow-conversation",
        task_id="paused",
        status="Succeeded",
        correlation=correlation,
    )
    assert adapter.external_task_notifications == []
    assert store.load_snapshot().external_tasks[0].resume_pending
    await runtime.close()
    recovered = FakeAdapter("pi")
    runtime = ConversationRuntime(tmp_path, recovered, default_harness_id="pi")
    try:
        await runtime.get_snapshot("workflow-conversation", context)
        assert recovered.external_task_notifications == []
        store.append(
            ProductEvent(
                conversation_id="workflow-conversation",
                type="execution.policy.updated",
                payload={"automatic_turns_suspended": False},
            )
        )
        await runtime.get_snapshot("workflow-conversation", context)
        assert len(recovered.external_task_notifications) == 1
        assert recovered.external_task_notifications[0][1].correlation == correlation
    finally:
        await runtime.close()
