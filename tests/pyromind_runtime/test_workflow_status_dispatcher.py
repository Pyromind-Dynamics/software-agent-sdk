from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from pyromind_agent_server.external_task_registry import WorkflowExternalTaskRegistry
from pyromind_agent_server.workflow_status_dispatcher import WorkflowStatusDispatcher
from pyromind_common.tools.data_preparation.platform_submit import (
    TASK_ASSOCIATION_DIRNAME as PREPARATION_TASK_DIR,
    DataPreparationTaskAssociation,
    DataPreparationTaskStore,
)
from pyromind_common.tools.pyromind_cleaning.task_store import (
    TASK_ASSOCIATION_DIRNAME as CLEANING_TASK_DIR,
    DatasetCleaningTaskAssociation,
    DatasetCleaningTaskStore,
)
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.domain.task_correlation import TaskCorrelation
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from .fake_adapter import FakeAdapter


class _ExternalTasks:
    def __init__(self) -> None:
        self.updated: list[tuple[str, str, str]] = []

    def owner(self, task_id: str) -> str | None:
        return "conversation-2" if task_id == "task-2" else None

    def resolve(self, conversation_id: str, task_id: str):
        if (conversation_id, task_id) != ("conversation-2", "task-2"):
            return None
        return {
            "task_id": "task-2",
            "kind": "data_cleaning",
            "run_id": "run-2",
            "status": "running",
            "output_dir": "/outputs/run-2",
            "submitted_at": "2026-08-24T00:00:00+00:00",
            "updated_at": "2026-08-24T00:00:00+00:00",
            "resume_pending": False,
        }

    def update_status(self, conversation_id: str, task_id: str, status: str) -> None:
        self.updated.append((conversation_id, task_id, status))

    def register(self, conversation_id: str, payload) -> None:
        return None


def test_external_task_registry_resolves_and_updates_owned_task(tmp_path) -> None:
    root = tmp_path / "conversations"
    store = DatasetCleaningTaskStore(root / CLEANING_TASK_DIR)
    store.save(
        DatasetCleaningTaskAssociation(
            task_id="task-1",
            conversation_id="conversation-1",
            run_id="run-1",
            output_dir="/outputs/run-1",
            input_path="/input.jsonl",
            script_path="/pipeline.py",
            status="Running",
            submitted_at=datetime(2026, 8, 24, tzinfo=UTC),
            updated_at=datetime(2026, 8, 24, tzinfo=UTC),
        )
    )
    registry = WorkflowExternalTaskRegistry(root)

    payload = registry.resolve("conversation-1", "task-1")
    registry.update_status("conversation-1", "task-1", "succeeded")

    assert payload is not None
    assert payload["kind"] == "data_cleaning"
    assert registry.owner("task-1") == "conversation-1"
    assert registry.resolve("another-conversation", "task-1") is None
    updated = store.get("task-1")
    assert updated is not None
    assert updated.status == "Succeeded"


def test_external_task_registry_canonicalizes_preparation_owner(tmp_path) -> None:
    root = tmp_path / "conversations"
    store = DataPreparationTaskStore(root / PREPARATION_TASK_DIR)
    store.save(
        DataPreparationTaskAssociation(
            task_id="task-preparation",
            conversation_id="4207b60c-6423-49f6-b4c4-2a50598b1323",
            run_id="run-1",
            output_dir="/outputs/run-1",
            input_path="/input.jsonl",
            script_path="public_data/data-preparation/pipeline.py",
            status="Pending",
        )
    )

    registry = WorkflowExternalTaskRegistry(root)

    assert registry.owner("task-preparation") == ("4207b60c642349f6b4c42a50598b1323")
    assert (
        registry.resolve("4207b60c642349f6b4c42a50598b1323", "task-preparation")
        is not None
    )


async def test_dispatcher_resolves_uuid_preparation_owner_without_callback_id(
    tmp_path,
) -> None:
    root = tmp_path / "conversations"
    conversation_id = "4207b60c642349f6b4c42a50598b1323"
    conversation = root / conversation_id
    conversation.mkdir(parents=True)
    product_store = FileProductStore(conversation)
    product_store.create(
        ConversationSnapshot(
            conversation_id=conversation_id,
            capabilities=HarnessCapabilities(cancel=True),
        ),
        user_id="42",
        harness_id="pi",
    )
    task_store = DataPreparationTaskStore(root / PREPARATION_TASK_DIR)
    task_store.save(
        DataPreparationTaskAssociation(
            task_id="8262",
            conversation_id="4207b60c-6423-49f6-b4c4-2a50598b1323",
            run_id="run-1",
            output_dir="/outputs/run-1",
            input_path="/input.jsonl",
            script_path="public_data/data-preparation/pipeline.py",
            status="Pending",
        )
    )
    registry = WorkflowExternalTaskRegistry(root)
    runtime = ConversationRuntime(
        root,
        {"pi": FakeAdapter("pi")},
        default_harness_id="pi",
        external_tasks=registry,
    )

    result = await WorkflowStatusDispatcher(runtime).dispatch(
        task_id="8262",
        status="Succeeded",
    )

    assert result.conversation_id == conversation_id
    assert product_store.load_snapshot().external_tasks[0].status == "succeeded"
    association = task_store.get("8262")
    assert association is not None
    assert association.status == "Succeeded"


async def test_dispatcher_routes_pi_callback_and_store_dedupes(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversation = conversations / "conversation-1"
    conversation.mkdir(parents=True)
    store = FileProductStore(conversation)
    store.create(
        ConversationSnapshot(
            conversation_id="conversation-1",
            capabilities=HarnessCapabilities(cancel=True),
        ),
        user_id="42",
        harness_id="pi",
    )
    task = {
        "task_id": "task-1",
        "kind": "data_preparation",
        "run_id": "run-1",
        "status": "running",
        "output_dir": "/agent/conversation-1/data_preparation/run-1",
        "submitted_at": "2026-08-24T00:00:00+00:00",
        "updated_at": "2026-08-24T00:00:00+00:00",
        "resume_pending": False,
    }
    store.append(
        ProductEvent(
            conversation_id="conversation-1",
            type="external_task.submitted",
            payload=task,
        )
    )
    runtime = ConversationRuntime(
        conversations,
        {"pi": FakeAdapter()},
        default_harness_id="pi",
    )
    dispatcher = WorkflowStatusDispatcher(runtime)
    first = await dispatcher.dispatch(
        task_id="task-1",
        status="Succeeded",
        conversation_id="conversation-1",
    )
    second = await dispatcher.dispatch(
        task_id="task-1",
        status="Succeeded",
        conversation_id="conversation-1",
    )

    assert first.outcome == second.outcome == "delivered_async"
    events = store.replay()
    assert [event.type for event in events] == [
        "external_task.submitted",
        "external_task.completed",
    ]


async def test_only_trusted_platform_can_adopt_and_update_signed_sdk_tasks(tmp_path):
    conversation_id = "signed-sdk-conversation"
    conversation = tmp_path / conversation_id
    conversation.mkdir()
    store = FileProductStore(conversation)
    store.create(
        ConversationSnapshot(
            conversation_id=conversation_id, capabilities=HarnessCapabilities()
        ),
        user_id="42",
        harness_id="pi",
    )
    correlation = store.issue_task_correlation(
        TaskCorrelation(conversation_id=conversation_id, task_kind="workflow_run")
    )
    runtime = ConversationRuntime(tmp_path, FakeAdapter())
    dispatcher = WorkflowStatusDispatcher(runtime)
    arguments = {
        "task_id": "actual-sdk-task",
        "correlation": correlation,
        "auto_run": False,
    }
    try:
        await dispatcher.dispatch(status="Succeeded", **arguments)
        assert store.load_snapshot().external_tasks == ()

        await dispatcher.dispatch(status="Running", trusted_platform=True, **arguments)
        assert store.load_snapshot().external_tasks[0].status == "running"

        with pytest.raises(PermissionError, match="trusted platform"):
            await dispatcher.dispatch(status="Succeeded", **arguments)
        assert store.load_snapshot().external_tasks[0].status == "running"

        await dispatcher.dispatch(
            status="Succeeded", trusted_platform=True, **arguments
        )
        assert store.load_snapshot().external_tasks[0].status == "succeeded"
    finally:
        await runtime.close()


async def test_dispatcher_routes_pi_through_same_runtime_path(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversation = conversations / "conversation-2"
    conversation.mkdir(parents=True)
    store = FileProductStore(conversation)
    store.create(
        ConversationSnapshot(
            conversation_id="conversation-2",
            capabilities=HarnessCapabilities(cancel=True),
        ),
        user_id="42",
        harness_id="pi",
    )
    adapter = FakeAdapter("pi")
    external_tasks = _ExternalTasks()
    runtime = ConversationRuntime(
        conversations,
        {"pi": adapter},
        default_harness_id="pi",
        external_tasks=external_tasks,
    )
    result = await WorkflowStatusDispatcher(runtime).dispatch(
        task_id="task-2",
        status="Succeeded",
    )

    assert result.outcome == "delivered_async"
    deferred = store.load_snapshot().external_tasks[0]
    assert deferred.status == "succeeded"
    assert deferred.resume_pending is True
    assert adapter.external_task_notifications == []

    await runtime.get_snapshot("conversation-2", RequestContext(user_id="42"))
    await asyncio.wait_for(runtime._read_recoveries["conversation-2"], 3)

    assert store.load_snapshot().external_tasks[0].resume_pending is False
    assert adapter.external_task_notifications[0][0] == "conversation-2"
    assert adapter.external_task_notifications[0][1].task_id == "task-2"
    assert external_tasks.updated == [("conversation-2", "task-2", "succeeded")]

    await WorkflowStatusDispatcher(runtime).dispatch(
        task_id="task-2",
        status="Failed",
        conversation_id="conversation-2",
    )
    assert store.load_snapshot().external_tasks[0].status == "succeeded"
    assert len(adapter.external_task_notifications) == 1
    assert external_tasks.updated == [
        ("conversation-2", "task-2", "succeeded"),
        ("conversation-2", "task-2", "succeeded"),
    ]
    await runtime.close()


async def test_dispatcher_routes_graph_task_before_conversation_runtime(
    tmp_path,
) -> None:
    runtime = ConversationRuntime(tmp_path / "conversations", FakeAdapter())
    calls: list[tuple[str, str, str, str | None]] = []

    async def graph_task_handler(
        *, task_id: str, status: str, conversation_id: str, task_kind: str | None
    ) -> str | None:
        calls.append((task_id, status, conversation_id, task_kind))
        return conversation_id if conversation_id == "graph-conversation" else None

    dispatcher = WorkflowStatusDispatcher(
        runtime,
        graph_task_handler=graph_task_handler,
    )
    result = await dispatcher.dispatch(
        task_id="graph-task-1",
        status="Succeeded",
        correlation=TaskCorrelation(
            conversation_id="graph-conversation", task_kind="echomind_train"
        ),
    )

    assert result.outcome == "delivered_async"
    assert result.conversation_id == "graph-conversation"
    assert calls == [
        ("graph-task-1", "Succeeded", "graph-conversation", "echomind_train")
    ]
    mismatched = await dispatcher.dispatch(
        task_id="graph-task-1",
        status="Succeeded",
        correlation=TaskCorrelation(
            conversation_id="unrelated-conversation", task_kind="echomind_train"
        ),
    )
    assert mismatched.outcome == "unknown_task"
    await runtime.close()


async def test_dispatcher_rejects_callback_for_conflicting_owner(tmp_path) -> None:
    conversations = tmp_path / "conversations"
    conversation = conversations / "conversation-2"
    conversation.mkdir(parents=True)
    store = FileProductStore(conversation)
    store.create(
        ConversationSnapshot(
            conversation_id="conversation-2",
            capabilities=HarnessCapabilities(cancel=True),
        ),
        user_id="42",
        harness_id="pi",
    )
    runtime = ConversationRuntime(
        conversations,
        {"pi": FakeAdapter("pi")},
        default_harness_id="pi",
        external_tasks=_ExternalTasks(),
    )

    result = await WorkflowStatusDispatcher(runtime).dispatch(
        task_id="task-2",
        status="Succeeded",
        conversation_id="another-conversation",
    )

    assert result.outcome == "unknown_task"
    assert store.load_snapshot().external_tasks == ()


def test_external_task_registry_cannot_transfer_ownership_or_change_the_recipient(
    tmp_path,
):
    import pytest

    from pyromind_runtime.domain.task_correlation import TaskCorrelation

    registry = WorkflowExternalTaskRegistry(tmp_path)
    correlation = TaskCorrelation(
        conversation_id="c",
        task_kind="workflow_run",
        type="subagent",
        subagent_id="worker",
    )
    payload = {"task_id": "task", "correlation": correlation.model_dump(mode="json")}
    registry.register("c", payload)
    with pytest.raises(PermissionError, match="conversation"):
        registry.register("other", payload)
    with pytest.raises(PermissionError, match="origin"):
        registry.register(
            "c",
            {
                **payload,
                "correlation": {**payload["correlation"], "subagent_id": "sibling"},
            },
        )
    registry.register("c", {"task_id": "task", "status": "running"})
    restored = registry.resolve("c", "task")
    assert restored is not None
    assert restored["correlation"] == payload["correlation"]


async def test_unknown_product_callback_never_falls_back_to_legacy_session(tmp_path):
    runtime = ConversationRuntime(
        tmp_path,
        {"pi": FakeAdapter("pi")},
        default_harness_id="pi",
    )
    try:
        result = await WorkflowStatusDispatcher(runtime).dispatch(
            task_id="unknown",
            status="Succeeded",
            conversation_id="missing",
        )
        assert result.outcome == "unknown_conversation"
        assert not (tmp_path / "missing" / "metadata.json").exists()
    finally:
        await runtime.close()


def test_retired_production_workflow_tool_cannot_be_resolved():
    from typing import Any, cast

    import pytest

    from pyromind_common.sdk.tool import Tool
    from pyromind_common.sdk.tool.registry import resolve_tool

    with pytest.raises(KeyError, match="run_workflow"):
        resolve_tool(Tool(name="run_workflow"), cast(Any, None))
