from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot, TimelineMessage


def latest_origin(snapshot: ConversationSnapshot) -> str | None:
    latest = max(
        (
            run
            for run in (snapshot.task_runs or {}).values()
            if run.task_id in snapshot.task_origins
        ),
        key=lambda run: run.seq,
        default=None,
    )
    return snapshot.task_origins[latest.task_id] if latest else None


def visible_chat_snapshot(snapshot: ConversationSnapshot) -> ConversationSnapshot:
    return snapshot.model_copy(
        update={
            "timeline": tuple(
                item
                for item in snapshot.timeline
                if not (
                    isinstance(item, TimelineMessage)
                    and item.execution_notice
                    and (item.role != "assistant" or item.status != "completed")
                )
            )
        }
    )


def visible_chat_event(event: ProductEvent) -> ProductEvent:
    if event.payload.get("execution_notice"):
        # Preserve the replay sequence without exposing a historical relay.
        return event.model_copy(
            update={"type": "chat.notification.skipped", "payload": {}}
        )
    return event
