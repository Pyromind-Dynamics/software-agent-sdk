from __future__ import annotations

import asyncio
import json

import pytest
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from pydantic import ValidationError
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.application.snapshot_projector import (
    SnapshotProjector,
    project_stage_activity,
)
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.content import TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.experiment_plans import ExperimentContext
from pyromind_runtime.domain.preparation import (
    PreparationPolicy,
    ProjectView,
    WorkflowNodeDefinition,
)
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    ExternalTaskState,
    OperationActivity,
    TimelineOperation,
)
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from .fake_adapter import FakeAdapter


def activity():
    return {
        "items": [
            {
                "kind": "message",
                "item_id": "child-message",
                "started_seq": 1,
                "role": "assistant",
                "content": [{"type": "text", "text": "**Reading samples**"}],
                "status": "completed",
            },
            {
                "kind": "operation",
                "item_id": "child-tool",
                "started_seq": 2,
                "name": "read",
                "status": "completed",
                "arguments": {"path": "/eval.jsonl"},
                "output": [{"type": "text", "text": "Four samples"}],
            },
        ],
        "truncated": False,
    }


@pytest.mark.parametrize("status", ["running", "completed", "failed"])
@pytest.mark.parametrize(
    "platform_status",
    [None, "pending", "running", "waiting", "succeeded", "failed", "stopped"],
)
def test_role_activity_projects_into_existing_stage_columns_without_business_success(
    status,
    platform_status,
):
    policy = PreparationPolicy(
        allowed_roles=("worker",),
        workflow_nodes=(
            WorkflowNodeDefinition(id="baseline", name="Baseline", roles=("worker",)),
        ),
    )
    operation = TimelineOperation.model_validate(
        {
            "item_id": "child-operation",
            "started_seq": 1,
            "name": "worker",
            "category": "subagent",
            "status": status,
            "activity": activity(),
            "details": {"subagent": {"session_key": "child-session"}},
        }
    )
    project = ProjectView(
        project_id="project",
        conversation_id="chat",
        plan_sequence=1,
        creation_command_id="create",
        plan=ExperimentContext.model_validate(
            {
                "stages": [
                    {"id": "business-stage", "kind": "baseline", "name": "Measure"}
                ]
            }
        ),
    )
    snapshot = ConversationSnapshot(
        conversation_id="chat",
        capabilities=HarnessCapabilities(),
        project=project,
        timeline=(operation,),
        external_tasks=(
            ExternalTaskState(
                task_id="platform-task",
                kind="workflow_run",
                status=platform_status,
                submitted_at="2026-11-11T03:34:13Z",
                updated_at="2026-11-11T03:34:13Z",
                correlation=TaskCorrelation(
                    conversation_id="chat",
                    type="subagent",
                    subagent_id="child-session",
                    task_kind="workflow_run",
                ),
            ),
        )
        if platform_status
        else (),
    )
    projected = project_stage_activity(snapshot, policy)
    child = projected.timeline[0]
    assert isinstance(child, TimelineOperation)
    assert child.execution == {
        "project_id": "project",
        "task_run_id": "child-operation",
        "stage_id": "business-stage",
        "plan_sequence": 1,
    }
    assert child.activity == operation.activity
    assert projected.experiment_executions is not None
    execution = projected.experiment_executions[0]
    assert child.execution is not None
    assert execution.run_id == child.execution["task_run_id"]
    expected = {
        "running": "preparing",
        "completed": "agent_completed",
        "failed": "failed",
    }[status]
    if platform_status in {"pending", "running", "waiting"}:
        expected = "waiting_platform"
    elif platform_status in {"failed", "stopped"}:
        expected = "failed" if platform_status == "failed" else "cancelled"
    elif platform_status == "succeeded" and status == "completed":
        expected = "waiting_analysis"
    elif platform_status == "succeeded" and status == "running":
        expected = "analyzing"
    assert execution.state == expected
    assert execution.platform_task_id == ("platform-task" if platform_status else None)
    assert execution.report_state != "ready"
    assert operation.execution is None
    assert snapshot.experiment_executions is None
    assert project_stage_activity(projected, policy) == projected
    ambiguous = project.model_copy(
        update={
            "plan": project.plan.model_copy(
                update={
                    "stages": (
                        *project.plan.stages,
                        project.plan.stages[0].model_copy(update={"id": "second"}),
                    )
                }
            )
        }
    )
    unmapped = project_stage_activity(
        snapshot.model_copy(update={"project": ambiguous}), policy
    )
    assert unmapped.timeline == snapshot.timeline
    assert not unmapped.experiment_executions


@pytest.mark.parametrize("link", ["session", "previous"])
@pytest.mark.parametrize("delivery", ["none", "published"])
def test_continuations_share_one_stage_execution_and_preserve_all_rounds(
    link, delivery
):
    session_key = "same-context" if link == "session" else None
    report = "public_data/subagents/first/report.md"

    policy = PreparationPolicy(
        allowed_roles=("worker",),
        workflow_nodes=(
            WorkflowNodeDefinition(id="baseline", name="Baseline", roles=("worker",)),
        ),
    )
    first = TimelineOperation(
        item_id="subagent:first",
        run_id="first",
        name="worker",
        started_seq=1,
        category="subagent",
        status="completed",
        activity=OperationActivity.model_validate(activity()),
        details={
            "subagent": {
                "session_key": session_key,
                "report_uri": report if delivery == "published" else None,
            }
        },
    )
    second = first.model_copy(
        update={
            "item_id": "subagent:second",
            "run_id": "second",
            "started_seq": 4,
            "status": "completed",
            "details": {
                "subagent": {
                    "session_key": session_key,
                    "previous_run_id": "first" if link == "previous" else None,
                }
            },
        }
    )
    tool_operations = []
    if delivery in {"read", "none"}:
        tool_operations.append(
            TimelineOperation(
                item_id="read-report",
                name="read",
                started_seq=2,
                category="tool",
                status="completed" if delivery == "read" else "failed",
                arguments={"path": report},
            )
        )
    if link == "tool_result":
        tool_operations.append(
            TimelineOperation(
                item_id="resume",
                name="subagent",
                started_seq=3,
                category="tool",
                status="completed",
                arguments={"action": "resume", "id": "first"},
                details={"runId": "second"},
            )
        )
    fresh = second.model_copy(
        update={
            "item_id": "subagent:fresh",
            "run_id": "fresh",
            "started_seq": 5,
            "status": "running",
            "details": {"subagent": {"session_key": "different-context"}},
        }
    )
    snapshot = ConversationSnapshot(
        conversation_id="chat",
        capabilities=HarnessCapabilities(),
        project=ProjectView(
            project_id="project",
            conversation_id="chat",
            plan_sequence=1,
            creation_command_id="create",
            plan=ExperimentContext.model_validate(
                {
                    "stages": [{"id": "stage", "kind": "baseline", "name": "Measure"}],
                }
            ),
        ),
        timeline=(first, *tool_operations, second, fresh),
    )
    result = project_stage_activity(snapshot, policy)
    assert result.experiment_executions is not None
    continued, independent = result.experiment_executions
    assert continued.run_id == first.item_id
    assert continued.state == ("agent_completed" if delivery == "none" else "reported")
    assert continued.report_uri == (None if delivery == "none" else report)
    assert continued.report_state == ("not_started" if delivery == "none" else "ready")
    assert independent.run_id == fresh.item_id
    assert independent.state == "preparing"
    assert len(result.timeline) == len(snapshot.timeline)
    for item in (result.timeline[0], result.timeline[-2]):
        assert isinstance(item, TimelineOperation)
        assert item.execution is not None
        assert item.execution["task_run_id"] == first.item_id
        assert item.activity == first.activity


def test_adapter_translates_native_activity_into_shared_models():
    event = translate_runner_event(
        {
            "eventId": "native",
            "sessionId": "chat",
            "runId": "native-child",
            "kind": "subagent.progress",
            "payload": {"activity": activity()},
        }
    )[0]
    assert event.type == "operation.progress"
    assert event.payload["activity"] == OperationActivity.model_validate(
        activity()
    ).model_dump(mode="json")
    assert "sessionFile" not in json.dumps(event.payload)
    operation = TimelineOperation.model_validate(
        {"item_id": "old", "started_seq": 1, "name": "read"}
    )
    assert "activity" not in operation.model_dump()
    for items in (
        [{**activity()["items"][0], "role": "system"}],
        [{**activity()["items"][1], "category": "subagent"}],
        [{**activity()["items"][1], "activity": {"items": []}}],
    ):
        with pytest.raises(ValidationError):
            OperationActivity.model_validate({"items": items})


def test_projection_persists_child_activity_without_reopening_completed_work():
    projector = SnapshotProjector()
    snapshot = ConversationSnapshot(
        conversation_id="chat", capabilities=HarnessCapabilities()
    )
    for seq, kind, values in (
        (1, "operation.started", {"name": "eval", "category": "subagent"}),
        (2, "operation.completed", {"output": [{"type": "text", "text": "Done"}]}),
        (3, "operation.progress", {"activity": activity()}),
    ):
        snapshot = projector.reduce(
            snapshot,
            ProductEvent.model_validate(
                {
                    "conversation_id": "chat",
                    "seq": seq,
                    "type": kind,
                    "payload": {"operation_id": "child", **values},
                }
            ),
        )
    operation = snapshot.timeline[0]
    assert isinstance(operation, TimelineOperation)
    assert operation.status == "completed"
    assert operation.completed_seq == 2
    assert isinstance(operation.output[0], TextContent)
    assert operation.output[0].text == "Done"
    assert operation.activity is not None
    restored = ConversationSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored == snapshot


async def test_read_recovers_activity_once_without_attaching_or_starting_work(
    tmp_path,
):
    class Reader(FakeAdapter):
        def __init__(self):
            super().__init__()
            self.reads = 0

        async def read_operation_activities(self, handle, operation_ids, context):
            self.reads += 1
            assert operation_ids == ("child",)
            return {
                "child": OperationActivity.model_validate(activity()),
                "foreign": OperationActivity(items=()),
            }

    directory = tmp_path / "chat"
    directory.mkdir()
    store = FileProductStore(directory)
    store.create(
        ConversationSnapshot(
            conversation_id="chat", capabilities=HarnessCapabilities()
        ),
        user_id="42",
        profile_id="profile",
    )
    store.append(
        ProductEvent(
            conversation_id="chat",
            type="operation.started",
            payload={"operation_id": "child", "name": "eval", "category": "subagent"},
        )
    )
    store.append(
        ProductEvent(
            conversation_id="chat",
            type="operation.completed",
            payload={"operation_id": "child", "output": []},
        )
    )
    adapter = Reader()
    runtime = ConversationRuntime(tmp_path, adapter)
    context = RequestContext(user_id="42", profile_id="profile")
    for denied in (
        RequestContext(user_id="other", profile_id="profile"),
        RequestContext(user_id="42", profile_id="other"),
    ):
        with pytest.raises(PermissionError):
            await runtime.get_snapshot("chat", denied)
    assert adapter.reads == 0
    first = await runtime.get_snapshot("chat", context)
    assert first.timeline[0].kind == "operation" and first.timeline[0].activity is None
    await asyncio.wait_for(runtime._read_recoveries["chat"], timeout=3)
    second = await runtime.get_snapshot("chat", context)
    assert second.through_seq == first.through_seq + 1
    assert adapter.reads == 1 and adapter.attached == []
    assert len(first.timeline) == 1
    operation = second.timeline[0]
    assert isinstance(operation, TimelineOperation)
    assert operation.status == "completed" and operation.activity is not None
    await runtime.close()
