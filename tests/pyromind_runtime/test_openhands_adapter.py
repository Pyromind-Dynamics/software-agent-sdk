from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast
from uuid import UUID, uuid4

from harness_adapter.openhands_adapter import OpenHandsAdapter
from harness_adapter.openhands_adapter.event_translator import (
    TranslationState,
    translate_event,
)
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import HarnessEvent
from pyromind_runtime.domain.snapshot import WorkflowState
from pyromind_runtime.ports.harness import (
    ExternalTaskNotification,
    ForkSpec,
    ProductCheckpoint,
    SessionHandle,
)

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.pub_sub import Subscriber
from openhands.sdk.event import (
    ConversationStateUpdateEvent,
    Event,
    MessageEvent,
    ObservationEvent,
    StreamingDeltaEvent,
)
from openhands.sdk.llm import Message, TextContent
from openhands.tools.data_preparation.platform_submit import (
    DfSubmitPipelineObservation,
)
from openhands.tools.pyromind_cleaning.definition import (
    RunDatasetCleaningObservation,
)
from openhands.tools.update_plan import PlanStep, UpdatePlanObservation
from openhands.tools.workflow_debug import WorkflowDebugObservation


def _message(
    source: Literal["agent", "user", "environment", "hook"],
    text: str,
    *,
    event_id: str,
) -> MessageEvent:
    return MessageEvent(
        id=event_id,
        source=source,
        llm_message=Message(
            role="user" if source == "user" else "assistant",
            content=[TextContent(text=text)],
        ),
    )


class _EventService:
    def __init__(self, conversation_dir: Path | None = None) -> None:
        self.subscriber: Subscriber[Event] | None = None
        self.subscriber_id = uuid4()
        self.search_count = 0
        self.unsubscribed: list[UUID] = []
        self.removed_tasks: list[str] = []
        self.notification: dict[str, object] | None = None
        self.conversation_dir = conversation_dir or Path()

    async def subscribe_to_events(self, subscriber: Subscriber[Event]) -> UUID:
        self.subscriber = subscriber
        return self.subscriber_id

    async def search_events(self, **_kwargs):
        self.search_count += 1
        assert self.subscriber is not None
        await self.subscriber(_message("agent", "live", event_id="live-message"))
        return SimpleNamespace(
            items=[_message("user", "history", event_id="historic-message")],
            next_page_id=None,
        )

    async def unsubscribe_from_events(self, subscriber_id: UUID) -> bool:
        self.unsubscribed.append(subscriber_id)
        return True

    async def remove_active_long_task(self, task_id: str):
        self.removed_tasks.append(task_id)
        return SimpleNamespace(status="Running")

    async def send_internal_context(
        self,
        content,
        *,
        run=False,
        visible=False,
        extended_content=None,
    ):
        self.notification = {
            "content": content,
            "run": run,
            "visible": visible,
            "extended_content": extended_content,
        }


class _ConversationService:
    def __init__(self, conversation_id: UUID, event_service: _EventService) -> None:
        self.conversation_id = conversation_id
        self.event_service = event_service
        self.fork_target: tuple[UUID, _EventService] | None = None
        self.fork_calls: list[dict[str, object]] = []

    async def get_event_service(
        self, conversation_id: UUID, *, user_id: str | None = None
    ):
        if conversation_id == self.conversation_id and user_id == "42":
            return self.event_service
        if self.fork_target is not None and conversation_id == self.fork_target[0]:
            return self.fork_target[1]
        return None

    async def fork_conversation_at_event(self, conversation_id: UUID, **kwargs):
        self.fork_calls.append({"conversation_id": conversation_id, **kwargs})
        return SimpleNamespace(id=kwargs["fork_id"]), "workflow-v1"


async def test_adapter_subscribes_before_history_and_drains_live_buffer() -> None:
    conversation_id = uuid4()
    event_service = _EventService()
    service = _ConversationService(conversation_id, event_service)
    adapter = OpenHandsAdapter(lambda: cast(ConversationService, service))
    handle = await adapter.attach_session(
        conversation_id.hex, RequestContext(user_id="42")
    )
    events = cast(AsyncGenerator[HarnessEvent], adapter.subscribe(handle))

    received = [await anext(events) for _ in range(5)]
    assert [event.type for event in received] == [
        "message.started",
        "message.completed",
        "message.started",
        "message.completed",
        "history.synced",
    ]
    historic_content = cast(list[dict[str, str]], received[0].payload["content"])
    live_content = cast(list[dict[str, str]], received[2].payload["content"])
    assert historic_content[0]["text"] == "history"
    assert live_content[0]["text"] == "live"
    await events.aclose()
    await adapter.close(handle)
    assert event_service.unsubscribed == [event_service.subscriber_id]


async def test_adapter_notifies_openhands_through_formal_harness_port() -> None:
    conversation_id = uuid4()
    event_service = _EventService()
    service = _ConversationService(conversation_id, event_service)
    adapter = OpenHandsAdapter(lambda: cast(ConversationService, service))
    handle: SessionHandle = await adapter.attach_session(
        conversation_id.hex, RequestContext(user_id="42")
    )

    result = await adapter.notify_external_task(
        handle,
        ExternalTaskNotification(
            task_id="task-1",
            kind="data_cleaning",
            run_id="run-1",
            status="succeeded",
            output_dir="/outputs/run-1",
            visible_text="Task succeeded",
            hidden_text="<system_reminder>continue</system_reminder>",
        ),
        RequestContext(user_id="42"),
    )

    assert result == {"accepted": True}
    assert event_service.removed_tasks == ["task-1"]
    assert event_service.notification is not None
    assert event_service.notification["run"] is True
    assert event_service.notification["visible"] is True
    await adapter.close(handle)


async def test_adapter_fork_delegates_native_checkpoint_without_backfill(
    tmp_path,
) -> None:
    source_id = uuid4()
    target_id = uuid4()
    source_events = _EventService(tmp_path / source_id.hex)
    target_dir = tmp_path / target_id.hex
    (target_dir / "product").mkdir(parents=True)
    (target_dir / "product" / "copied.json").write_text("source product")
    target_events = _EventService(target_dir)
    service = _ConversationService(source_id, source_events)
    service.fork_target = (target_id, target_events)
    adapter = OpenHandsAdapter(lambda: cast(ConversationService, service))
    context = RequestContext(user_id="42")
    source = await adapter.attach_session(source_id.hex, context)
    checkpoint = ProductCheckpoint(
        event_id="product-workflow-v1",
        through_seq=2,
        adapter_checkpoint_ref="native-workflow-event",
        workflow=WorkflowState(
            resource_id="pyromind_workflow",
            version="v1",
            dsl="workflow = InputNode()",
            canvas=None,
        ),
    )

    target = await adapter.fork(
        source,
        ForkSpec(
            source_conversation_id=source_id.hex,
            target_conversation_id=target_id.hex,
            event_id=checkpoint.event_id,
            title="Forked",
        ),
        checkpoint,
        context,
    )

    assert service.fork_calls == [
        {
            "conversation_id": source_id,
            "event_id": "native-workflow-event",
            "fork_id": target_id,
            "title": "Forked",
            "tags": {"pyromind_app": "true"},
            "user_id": "42",
        }
    ]
    assert not (target_dir / "product").exists()
    assert target_events.search_count == 0
    assert (await anext(adapter.subscribe(target))).type == "history.synced"
    await adapter.close(target)
    await adapter.close(source)


def test_translator_ignores_empty_streaming_delta() -> None:
    state = TranslationState(session_id="conversation-1")

    assert translate_event(state, StreamingDeltaEvent(content=None)) == ()
    assert translate_event(state, StreamingDeltaEvent(content="")) == ()
    assert state.streaming_message_id is None

    translated = translate_event(state, StreamingDeltaEvent(content="hello"))

    assert [event.type for event in translated] == [
        "message.started",
        "message.delta",
    ]
    assert translated[1].payload["text"] == "hello"


def test_translator_preserves_workflow_and_usage_from_old_events() -> None:
    state = TranslationState(session_id="conversation-1")
    workflow = translate_event(
        state,
        ConversationStateUpdateEvent(
            id="workflow-event",
            key="pyromind_workflow",
            value={
                "workflow": "workflow = InputNode()",
                "xyflow": {"nodes": []},
            },
        ),
    )
    usage = translate_event(
        state,
        ConversationStateUpdateEvent(
            id="usage-event",
            key="stats",
            value={
                "usage_to_metrics": {
                    "main": {
                        "accumulated_cost": 0.25,
                        "accumulated_token_usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "cache_read_tokens": 3,
                        },
                    }
                }
            },
        ),
    )

    assert workflow[0].type == "workflow.updated"
    assert workflow[0].payload == {
        "resource_id": "pyromind_workflow",
        "version": "workflow-event",
        "dsl": "workflow = InputNode()",
        "canvas": {"nodes": []},
    }
    assert usage[0].type == "usage.updated"
    assert usage[0].payload == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cached_tokens": 3,
        "cost_usd": 0.25,
    }


def test_translator_preserves_structured_update_plan_steps() -> None:
    translated = translate_event(
        TranslationState(session_id="conversation-1"),
        ObservationEvent(
            id="plan-observation",
            source="environment",
            action_id="plan-action",
            tool_name="update_plan",
            tool_call_id="plan-call",
            observation=UpdatePlanObservation.from_text(
                "Plan updated",
                explanation="Continue",
                plan=[
                    PlanStep(step="Inspect input", status="completed"),
                    PlanStep(step="Build output", status="in_progress"),
                ],
            ),
        ),
    )

    assert translated[1].type == "plan.updated"
    assert translated[1].payload == {
        "steps": [
            {"step": "Inspect input", "status": "completed"},
            {"step": "Build output", "status": "in_progress"},
        ],
        "explanation": "Continue",
    }


def test_translator_projects_openhands_external_task_lifecycle() -> None:
    state = TranslationState(session_id="conversation-1")
    submitted = translate_event(
        state,
        ObservationEvent(
            id="submit-observation",
            source="environment",
            action_id="submit-action",
            tool_name="df_submit_pipeline",
            tool_call_id="submit-call",
            observation=DfSubmitPipelineObservation.from_text(
                "submitted",
                status="Running",
                task_id="task-1",
                run_id="run-1",
                output_dir="/outputs/run-1",
            ),
        ),
    )
    stopped = translate_event(
        state,
        ConversationStateUpdateEvent(
            id="stopped-state",
            key="active_long_tasks",
            value=[
                {
                    "task_id": "task-1",
                    "kind": "data_preparation",
                    "status": "Stopped",
                }
            ],
        ),
    )

    assert [event.type for event in submitted] == [
        "operation.completed",
        "external_task.submitted",
    ]
    assert submitted[1].payload["kind"] == "data_preparation"
    assert submitted[1].payload["run_id"] == "run-1"
    assert [event.type for event in stopped] == ["external_task.completed"]
    assert stopped[0].payload["status"] == "stopped"


def test_translator_projects_openhands_data_cleaning_submission() -> None:
    translated = translate_event(
        TranslationState(session_id="conversation-1"),
        ObservationEvent(
            id="cleaning-observation",
            source="environment",
            action_id="cleaning-action",
            tool_name="run_dataset_cleaning",
            tool_call_id="cleaning-call",
            observation=RunDatasetCleaningObservation.from_text(
                "submitted",
                status="Pending",
                task_id="task-2",
                run_id="run-2",
                output_dir="/outputs/run-2",
            ),
        ),
    )

    assert translated[1].type == "external_task.submitted"
    assert translated[1].payload["kind"] == "data_cleaning"


def test_translator_projects_workflow_debug_without_output_directory() -> None:
    translated = translate_event(
        TranslationState(session_id="conversation-1"),
        ObservationEvent(
            id="debug-observation",
            source="environment",
            action_id="debug-action",
            tool_name="workflow_debug",
            tool_call_id="debug-call",
            observation=WorkflowDebugObservation.from_text(
                "submitted",
                status="Pending",
                task_id="debug-task",
                attempt=2,
                max_attempts=10,
                keep_ui_lock=True,
            ),
        ),
    )

    assert translated[1].type == "external_task.submitted"
    assert translated[1].payload == {
        "task_id": "debug-task",
        "kind": "workflow_debug",
        "run_id": "debug-task",
        "status": "pending",
        "output_dir": None,
        "attempt": 2,
        "max_attempts": 10,
        "keep_ui_lock": True,
        "submitted_at": translated[1].occurred_at.isoformat(),
        "updated_at": translated[1].occurred_at.isoformat(),
        "resume_pending": False,
    }
