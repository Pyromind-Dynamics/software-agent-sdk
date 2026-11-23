from __future__ import annotations

from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    OperationActivity,
    TimelineMessage,
    TimelineOperation,
)


def _visible_activity(activity: OperationActivity) -> OperationActivity:
    return activity.model_copy(
        update={
            "items": tuple(
                item.model_copy(update={"thought": ()})
                if isinstance(item, TimelineOperation)
                else item
                for item in activity.items
                if not isinstance(item, TimelineMessage) or item.role == "user"
            )
        }
    )


def execution_snapshot(snapshot: ConversationSnapshot) -> ConversationSnapshot:
    return snapshot.model_copy(
        update={
            "timeline": tuple(
                item.model_copy(
                    update={
                        "thought": (),
                        "activity": _visible_activity(item.activity)
                        if item.activity is not None
                        else None,
                    }
                )
                if isinstance(item, TimelineOperation)
                else item
                for item in snapshot.timeline
                if not isinstance(item, TimelineMessage) or item.role == "user"
            )
        }
    )


def execution_event(event: ProductEvent) -> ProductEvent:
    payload = event.payload
    if event.type == "message.delta" or (
        event.type in {"message.started", "message.completed"}
        and payload.get("role") != "user"
    ):
        # The view still advances through every event for SSE cursor continuity.
        return event.model_copy(update={"payload": {}})
    if event.type.startswith("operation.") and (
        "thought" in payload or "activity" in payload
    ):
        public = payload.copy()
        public.pop("thought", None)
        activity = public.get("activity")
        if activity is not None:
            public["activity"] = _visible_activity(
                OperationActivity.model_validate(activity)
            ).model_dump(mode="json")
        return event.model_copy(update={"payload": public})
    return event
