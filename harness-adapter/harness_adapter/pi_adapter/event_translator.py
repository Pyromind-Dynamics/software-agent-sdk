from __future__ import annotations

import base64
import binascii
import logging
import re
from datetime import datetime
from typing import Any, cast

from pyromind_runtime.domain.events import HarnessEvent, HarnessEventType


logger = logging.getLogger(__name__)

_OMITTED_CONTENT_TEXT = "[Unsupported content omitted: invalid Pi content block.]"
_IMAGE_MIME_TYPE = re.compile(r"^image/[^;,\s]+$", re.IGNORECASE)


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
    if kind in {"message.started", "message.completed"}:
        payload = {**payload, "content": _content(payload.get("content"), kind=kind)}
    elif kind == "tool.started":
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
            "output": _content(payload.get("content"), kind=kind),
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


def _content(value: Any, *, kind: str) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        if value is not None:
            _log_omitted(kind, "content_not_array", None)
        return []
    output: list[dict[str, Any]] = []
    for block in value:
        if not isinstance(block, dict):
            output.append(_omitted_content(kind, "block_not_object", None))
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            output.append({"type": "text", "text": block["text"]})
            continue
        if block_type == "image_url":
            url = block.get("url")
            if isinstance(url, str) and url:
                output.append({"type": "image_url", "url": url})
            else:
                output.append(_omitted_content(kind, "invalid_image_url", block_type))
            continue
        if block_type == "image":
            image_urls = block.get("image_urls")
            if isinstance(image_urls, (list, tuple)):
                if image_urls and all(
                    isinstance(url, str) and bool(url) for url in image_urls
                ):
                    output.append({"type": "image", "image_urls": list(image_urls)})
                else:
                    output.append(
                        _omitted_content(kind, "invalid_image_urls", block_type)
                    )
                continue
            data = block.get("data")
            mime_type = block.get("mime_type")
            if not isinstance(mime_type, str):
                mime_type = block.get("mimeType")
            if (
                isinstance(data, str)
                and data
                and isinstance(mime_type, str)
                and _IMAGE_MIME_TYPE.fullmatch(mime_type)
                and _is_base64(data)
            ):
                output.append(
                    {
                        "type": "image",
                        "image_urls": [f"data:{mime_type};base64,{data}"],
                    }
                )
            else:
                output.append(
                    _omitted_content(kind, "invalid_inline_image", block_type)
                )
            continue
        output.append(_omitted_content(kind, "unsupported_block_type", block_type))
    return output


def _is_base64(value: str) -> bool:
    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def _omitted_content(kind: str, reason: str, block_type: Any) -> dict[str, str]:
    _log_omitted(kind, reason, block_type)
    return {"type": "text", "text": _OMITTED_CONTENT_TEXT}


def _log_omitted(kind: str, reason: str, block_type: Any) -> None:
    safe_type = block_type if isinstance(block_type, str) else type(block_type).__name__
    logger.warning(
        "pi.content_omitted event_kind=%s block_type=%s reason=%s",
        kind,
        safe_type,
        reason,
    )


def _occurred_at(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now().astimezone()
