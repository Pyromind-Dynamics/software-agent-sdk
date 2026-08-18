from __future__ import annotations

from pyromind_runtime.domain.events import HarnessEvent, ProductEvent


class ProductEventProjector:
    """Remove harness identity while retaining a stable source correlation."""

    def project(
        self,
        conversation_id: str,
        event: HarnessEvent,
    ) -> ProductEvent | None:
        if event.type == "history.synced":
            return None
        return ProductEvent(
            event_id=event.event_id,
            conversation_id=conversation_id,
            occurred_at=event.occurred_at,
            type=event.type,
            run_id=event.run_id,
            payload=event.payload,
            source_event_id=event.source_event_id,
        )
