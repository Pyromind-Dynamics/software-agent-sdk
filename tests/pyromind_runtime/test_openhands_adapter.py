from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

from harness_adapter.openhands_adapter import OpenHandsAdapter
from harness_adapter.openhands_adapter.event_translator import (
    TranslationState,
    translate_event,
)
from pyromind_runtime.domain.context import RequestContext

from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.pub_sub import Subscriber
from openhands.sdk.event import (
    ConversationStateUpdateEvent,
    Event,
    MessageEvent,
    ObservationEvent,
)
from openhands.sdk.llm import Message, TextContent
from openhands.tools.update_plan import PlanStep, UpdatePlanObservation


def _message(source: str, text: str, *, event_id: str) -> MessageEvent:
    return MessageEvent(
        id=event_id,
        source=source,
        llm_message=Message(
            role="user" if source == "user" else "assistant",
            content=[TextContent(text=text)],
        ),
    )


class _EventService:
    def __init__(self) -> None:
        self.subscriber: Subscriber[Event] | None = None
        self.subscriber_id = uuid4()
        self.search_count = 0
        self.unsubscribed: list[UUID] = []

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


class _ConversationService:
    def __init__(self, conversation_id: UUID, event_service: _EventService) -> None:
        self.conversation_id = conversation_id
        self.event_service = event_service

    async def get_event_service(
        self, conversation_id: UUID, *, user_id: str | None = None
    ):
        if conversation_id == self.conversation_id and user_id == "42":
            return self.event_service
        return None


async def test_adapter_subscribes_before_history_and_drains_live_buffer() -> None:
    conversation_id = uuid4()
    event_service = _EventService()
    service = _ConversationService(conversation_id, event_service)
    adapter = OpenHandsAdapter(lambda: cast(ConversationService, service))
    handle = await adapter.attach_session(
        conversation_id.hex, RequestContext(user_id="42")
    )
    events = adapter.subscribe(handle)

    received = [await anext(events) for _ in range(5)]
    assert [event.type for event in received] == [
        "message.started",
        "message.completed",
        "message.started",
        "message.completed",
        "history.synced",
    ]
    assert received[0].payload["content"][0]["text"] == "history"
    assert received[2].payload["content"][0]["text"] == "live"
    await events.aclose()
    await adapter.close(handle)
    assert event_service.unsubscribed == [event_service.subscriber_id]


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
