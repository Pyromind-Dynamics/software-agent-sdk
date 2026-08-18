from __future__ import annotations

from pyromind_runtime.contracts.content import JsonObject
from pyromind_runtime.contracts.events import HarnessEvent, ProductEvent


_DIRECT_EVENT_TYPES = frozenset(
    {
        "run.started",
        "run.completed",
        "run.failed",
        "message.started",
        "message.delta",
        "message.completed",
        "permission.requested",
        "permission.resolved",
        "usage.updated",
    }
)
_TOOL_EVENT_TYPES = {
    "tool.started": "operation.started",
    "tool.progress": "operation.progress",
    "tool.completed": "operation.completed",
    "tool.failed": "operation.failed",
}


class ProductProjectionError(RuntimeError):
    pass


class ProductEventProjector:
    def project(
        self,
        conversation_id: str,
        event: HarnessEvent,
    ) -> tuple[ProductEvent, ...]:
        if event.type == "resource.updated":
            return ()
        if event.type in _DIRECT_EVENT_TYPES:
            return (self._event(conversation_id, event, event.type, event.payload),)

        product_type = _TOOL_EVENT_TYPES[event.type]
        payload = dict(event.payload)
        tool_call_id = self._required_string(payload, "tool_call_id")
        payload["operation_id"] = tool_call_id
        payload.pop("tool_call_id")
        if "tool_name" in payload:
            payload["name"] = payload.pop("tool_name")
        return (self._event(conversation_id, event, product_type, payload),)

    @staticmethod
    def _event(
        conversation_id: str,
        source: HarnessEvent,
        event_type: str,
        payload: JsonObject,
    ) -> ProductEvent:
        return ProductEvent.model_validate(
            {
                "event_id": source.event_id,
                "conversation_id": conversation_id,
                "occurred_at": source.occurred_at,
                "type": event_type,
                "run_id": source.run_id,
                "payload": payload,
            }
        )

    @staticmethod
    def _required_string(payload: JsonObject, key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ProductProjectionError(f"{key} must be a non-empty string")
        return value
