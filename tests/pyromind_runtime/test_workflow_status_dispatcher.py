from __future__ import annotations

from pyromind_agent_server.workflow_status_dispatcher import WorkflowStatusDispatcher
from pyromind_runtime.application.conversation_runtime import ConversationRuntime
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.infrastructure.file_product_store import FileProductStore

from .fake_adapter import FakeAdapter


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
