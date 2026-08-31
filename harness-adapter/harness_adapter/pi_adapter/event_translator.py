from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from pyromind_runtime.domain.events import HarnessEvent, HarnessEventType


def translate_runner_event(frame: dict[str, Any]) -> tuple[HarnessEvent, ...]:
    kind = frame.get("kind")
    payload = frame.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    common = {
        "event_id": str(frame.get("eventId")),
        "session_id": str(frame.get("sessionId")),
        "run_id": str(frame.get("runId")),
        "occurred_at": _occurred_at(frame.get("occurredAt")),
    }
    if kind == "agent.started":
        return (
            HarnessEvent(
                **common, type="status.changed", payload={"status": "running"}
            ),
        )
    if kind == "run.finished":
        outcome = payload.get("outcome")
        if not isinstance(outcome, dict):
            outcome = {}
        status = outcome.get("status")
        if status == "completed":
            return (
                HarnessEvent(
                    **common, type="status.changed", payload={"status": "idle"}
                ),
            )
        if status in {"cancelled", "suspended"}:
            return (
                HarnessEvent(
                    **common, type="status.changed", payload={"status": "paused"}
                ),
            )
        message = outcome.get("message")
        error_code = outcome.get("error_code")
        if status != "failed":
            error_code = "unknown_pi_outcome"
            message = f"Unknown Pi run outcome: {status!r}"
        return (
            HarnessEvent(
                **common,
                type="notice.raised",
                payload={
                    "severity": "error",
                    "code": str(error_code or "pi_runner_failed"),
                    "message": message
                    if isinstance(message, str)
                    else "Pi runner failed",
                },
            ),
            HarnessEvent(
                **{**common, "event_id": f"{common['event_id']}:status"},
                type="status.changed",
                payload={"status": "error"},
            ),
        )
    if not isinstance(kind, str):
        return ()
    event_type = {
        "message.started": "message.started",
        "message.delta": "message.delta",
        "message.completed": "message.completed",
        "tool.started": "operation.started",
        "tool.progress": "operation.progress",
        "tool.completed": "operation.completed",
        "tool.failed": "operation.failed",
        "usage.updated": "usage.updated",
    }.get(kind)
    if event_type is None:
        return ()
    if kind == "tool.started":
        payload = {
            "operation_id": payload.get("tool_call_id"),
            "name": payload.get("tool_name") or "tool",
            "category": "tool",
            "arguments": payload.get("arguments", {}),
            "thought": [],
        }
    elif isinstance(kind, str) and kind.startswith("tool."):
        payload = {
            "operation_id": payload.get("tool_call_id"),
            "name": payload.get("tool_name") or "tool",
            "output": payload.get("content", []),
            "details": payload.get("details")
            if isinstance(payload.get("details"), dict)
            or payload.get("details") is None
            else None,
            **(
                {"error_code": payload.get("error_code") or "tool_execution_failed"}
                if kind == "tool.failed"
                else {}
            ),
        }
    return (
        HarnessEvent(
            **common,
            type=cast(HarnessEventType, event_type),
            payload=payload,
        ),
    )


def _occurred_at(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now().astimezone()
