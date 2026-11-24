from __future__ import annotations

from pyromind_runtime.domain.content import JsonObject, TextContent
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
    timeline = []
    for item in snapshot.timeline:
        if isinstance(item, TimelineOperation):
            item = item.model_copy(
                update={
                    "thought": (),
                    "activity": _visible_activity(item.activity)
                    if item.activity
                    else None,
                }
            )
        elif isinstance(item, TimelineMessage) and item.role != "user":
            if (
                snapshot.project
                or item.task_id
                or item.role != "assistant"
                or not item.preparation_reply
            ):
                continue
            item = item.model_copy(
                update={
                    "content": (TextContent(text=item.preparation_reply),),
                    "preparation_reply": None,
                }
            )
        timeline.append(item)
    return snapshot.model_copy(update={"timeline": tuple(timeline)})


def execution_event(event: ProductEvent) -> ProductEvent:
    payload = event.payload
    if event.type == "message.delta" or (
        event.type in {"message.started", "message.completed"}
        and payload.get("role") != "user"
    ):
        if not event.task_id:
            key = (
                "preparation_reply_delta"
                if event.type == "message.delta"
                else "preparation_reply"
            )
            reply = payload.get(key)
            if isinstance(reply, str) and event.type != "message.started":
                visible: JsonObject = {
                    "message_id": payload["message_id"],
                    "role": "assistant",
                }
                if event.type == "message.delta":
                    visible["text"] = reply
                else:
                    visible["content"] = [{"type": "text", "text": reply}]
                return event.model_copy(update={"payload": visible})
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
