from __future__ import annotations

import ast
from typing import Literal, cast

from pydantic import JsonValue, TypeAdapter, ValidationError

from pyromind_runtime.domain.content import ContentBlock, JsonObject, TextContent
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    PendingPermission,
    PlanState,
    TimelineCompaction,
    TimelineMessage,
    TimelineNotice,
    TimelineOperation,
    TimelineWorkflow,
    UsageState,
    WorkflowState,
)


_CONTENT = TypeAdapter(tuple[ContentBlock, ...])
_STATUSES = frozenset(
    {
        "idle",
        "running",
        "paused",
        "waiting_for_confirmation",
        "finished",
        "error",
        "stuck",
        "deleting",
        "closed",
    }
)
_MessageRole = Literal["user", "assistant", "system"]
_OperationCategory = Literal["tool", "subagent", "observation"]
_CompactionStatus = Literal["started", "completed", "skipped", "failed"]
_NoticeSeverity = Literal["info", "warning", "error"]


class SnapshotProjectionError(RuntimeError):
    pass


class SnapshotProjector:
    def reduce(
        self,
        snapshot: ConversationSnapshot,
        event: ProductEvent,
    ) -> ConversationSnapshot:
        if event.conversation_id != snapshot.conversation_id:
            raise SnapshotProjectionError("event belongs to another conversation")
        if event.seq != snapshot.through_seq + 1:
            raise SnapshotProjectionError(
                f"expected seq {snapshot.through_seq + 1}, found {event.seq}"
            )

        handler = getattr(self, f"_on_{event.type.replace('.', '_')}", None)
        updated = handler(snapshot, event) if handler is not None else snapshot
        return updated.model_copy(update={"through_seq": event.seq})

    def _on_conversation_created(
        self, snapshot: ConversationSnapshot, _event: ProductEvent
    ) -> ConversationSnapshot:
        return snapshot

    def _on_status_changed(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        status = self._string(event.payload, "status")
        if status not in _STATUSES:
            return self._notice(
                snapshot,
                event,
                code="unsupported_status",
                message=f"Unsupported conversation status: {status}",
            )
        return snapshot.model_copy(update={"status": status})

    def _on_message_started(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        message_id = self._string(event.payload, "message_id")
        existing = self._find(snapshot, "message", message_id)
        if existing is not None:
            return snapshot
        role = self._string(event.payload, "role")
        if role == "agent":
            role = "assistant"
        if role not in {"user", "assistant", "system"}:
            role = "system"
        message = TimelineMessage(
            item_id=message_id,
            started_seq=event.seq,
            role=cast(_MessageRole, role),
            content=self._content(event.payload),
            run_id=event.run_id,
        )
        return self._append(snapshot, message)

    def _on_message_delta(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        message_id = self._string(event.payload, "message_id")
        item = self._find(snapshot, "message", message_id)
        if not isinstance(item, TimelineMessage):
            item = TimelineMessage(
                item_id=message_id,
                started_seq=event.seq,
                role="assistant",
                run_id=event.run_id,
            )
            snapshot = self._append(snapshot, item)
        text = self._string(event.payload, "text", allow_empty=True)
        content = item.content
        if content and isinstance(content[-1], TextContent):
            merged = content[-1].model_copy(update={"text": content[-1].text + text})
            content = (*content[:-1], merged)
        else:
            content = (*content, TextContent(text=text))
        return self._replace(snapshot, item.model_copy(update={"content": content}))

    def _on_message_completed(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        message_id = self._string(event.payload, "message_id")
        item = self._find(snapshot, "message", message_id)
        if not isinstance(item, TimelineMessage):
            role = self._string(event.payload, "role", default="assistant")
            if role == "agent":
                role = "assistant"
            if role not in {"user", "assistant", "system"}:
                role = "system"
            item = TimelineMessage(
                item_id=message_id,
                started_seq=event.seq,
                role=cast(_MessageRole, role),
                run_id=event.run_id,
            )
            snapshot = self._append(snapshot, item)
        update: dict[str, object] = {
            "completed_seq": event.seq,
            "status": "completed",
        }
        if "content" in event.payload:
            update["content"] = self._content(event.payload)
        return self._replace(snapshot, item.model_copy(update=update))

    def _on_operation_started(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        operation_id = self._string(event.payload, "operation_id")
        if self._find(snapshot, "operation", operation_id) is not None:
            return snapshot
        category = self._string(event.payload, "category", default="tool")
        if category not in {"tool", "subagent", "observation"}:
            category = "tool"
        operation = TimelineOperation(
            item_id=operation_id,
            started_seq=event.seq,
            name=self._string(event.payload, "name", default="operation"),
            category=cast(_OperationCategory, category),
            thought=self._content(event.payload, key="thought"),
            arguments=event.payload.get("arguments", {}),
            run_id=event.run_id,
        )
        return self._append(snapshot, operation)

    def _on_operation_progress(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        return self._complete_operation(snapshot, event, terminal=False)

    def _on_operation_completed(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        return self._complete_operation(snapshot, event, terminal=True)

    def _on_operation_failed(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        return self._complete_operation(snapshot, event, terminal=True, failed=True)

    def _complete_operation(
        self,
        snapshot: ConversationSnapshot,
        event: ProductEvent,
        *,
        terminal: bool,
        failed: bool = False,
    ) -> ConversationSnapshot:
        operation_id = self._string(event.payload, "operation_id")
        item = self._find(snapshot, "operation", operation_id)
        if not isinstance(item, TimelineOperation):
            item = TimelineOperation(
                item_id=operation_id,
                started_seq=event.seq,
                name=self._string(event.payload, "name", default="operation"),
                category="observation",
                run_id=event.run_id,
            )
            snapshot = self._append(snapshot, item)
        update: dict[str, object] = {}
        if "output" in event.payload or "content" in event.payload:
            update["output"] = self._content(
                event.payload,
                key="output" if "output" in event.payload else "content",
            )
        if "details" in event.payload:
            update["details"] = event.payload["details"]
        if "error_code" in event.payload:
            error_code = event.payload["error_code"]
            update["error_code"] = error_code if isinstance(error_code, str) else None
        if terminal:
            update["completed_seq"] = event.seq
            update["status"] = "failed" if failed else "completed"
        return self._replace(snapshot, item.model_copy(update=update))

    def _on_permission_requested(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        permission = PendingPermission.model_validate(event.payload)
        pending = tuple(
            item
            for item in snapshot.pending_permissions
            if item.permission_id != permission.permission_id
        )
        return snapshot.model_copy(
            update={
                "status": "waiting_for_confirmation",
                "pending_permissions": (*pending, permission),
            }
        )

    def _on_permission_resolved(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        permission_id = self._string(event.payload, "permission_id")
        pending = tuple(
            item
            for item in snapshot.pending_permissions
            if item.permission_id != permission_id
        )
        return snapshot.model_copy(
            update={
                "pending_permissions": pending,
                "status": "waiting_for_confirmation" if pending else "running",
            }
        )

    def _on_plan_updated(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        payload = dict(event.payload)
        if "steps" not in payload and "plan" in payload:
            payload["steps"] = payload.pop("plan")
        payload["steps"] = self._normalize_plan_steps(payload.get("steps"))
        return snapshot.model_copy(update={"plan": PlanState.model_validate(payload)})

    @staticmethod
    def _normalize_plan_steps(value: JsonValue) -> JsonValue:
        """Read plan steps written by the initial OpenHands adapter release."""
        if not isinstance(value, list):
            return value
        normalized: list[JsonValue] = []
        for item in value:
            if not isinstance(item, str) or not item.startswith("step="):
                normalized.append(item)
                continue
            try:
                step_literal, status_literal = item[5:].rsplit(" status=", 1)
                step = ast.literal_eval(step_literal)
                status = ast.literal_eval(status_literal)
            except (SyntaxError, ValueError):
                normalized.append(item)
                continue
            if isinstance(step, str) and status in {
                "pending",
                "in_progress",
                "completed",
            }:
                normalized.append({"step": step, "status": status})
            else:
                normalized.append(item)
        return normalized

    def _on_context_compaction_updated(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        status = self._string(event.payload, "status")
        if status not in {"started", "completed", "skipped", "failed"}:
            return self._notice(
                snapshot,
                event,
                code="invalid_compaction_status",
                message=f"Invalid compaction status: {status}",
            )
        item = TimelineCompaction(
            item_id=self._string(
                event.payload, "compaction_id", default=event.event_id
            ),
            started_seq=event.seq,
            status=cast(_CompactionStatus, status),
            summary=self._optional_string(event.payload, "summary"),
        )
        return self._append(snapshot, item).model_copy(
            update={"compaction_status": status}
        )

    def _on_workflow_updated(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        workflow = WorkflowState.model_validate(event.payload)
        item = TimelineWorkflow(
            item_id=event.event_id,
            started_seq=event.seq,
            workflow=workflow,
        )
        return self._append(snapshot, item).model_copy(
            update={"current_workflow": workflow}
        )

    def _on_usage_updated(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        return snapshot.model_copy(
            update={"usage": UsageState.model_validate(event.payload)}
        )

    def _on_notice_raised(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        return self._notice(
            snapshot,
            event,
            code=self._string(event.payload, "code", default="notice"),
            message=self._string(event.payload, "message", allow_empty=True),
            severity=self._string(event.payload, "severity", default="error"),
        )

    def _notice(
        self,
        snapshot: ConversationSnapshot,
        event: ProductEvent,
        *,
        code: str,
        message: str,
        severity: str = "warning",
    ) -> ConversationSnapshot:
        if severity not in {"info", "warning", "error"}:
            severity = "warning"
        return self._append(
            snapshot,
            TimelineNotice(
                item_id=event.event_id,
                started_seq=event.seq,
                severity=cast(_NoticeSeverity, severity),
                code=code,
                message=message,
            ),
        )

    @staticmethod
    def _append(snapshot: ConversationSnapshot, item: object) -> ConversationSnapshot:
        return snapshot.model_copy(update={"timeline": (*snapshot.timeline, item)})

    @staticmethod
    def _replace(
        snapshot: ConversationSnapshot, replacement: object
    ) -> ConversationSnapshot:
        item_id = getattr(replacement, "item_id")
        kind = getattr(replacement, "kind")
        return snapshot.model_copy(
            update={
                "timeline": tuple(
                    replacement
                    if item.kind == kind and item.item_id == item_id
                    else item
                    for item in snapshot.timeline
                )
            }
        )

    @staticmethod
    def _find(snapshot: ConversationSnapshot, kind: str, item_id: str) -> object | None:
        for item in snapshot.timeline:
            if item.kind == kind and item.item_id == item_id:
                return item
        return None

    @staticmethod
    def _content(
        payload: JsonObject, *, key: str = "content"
    ) -> tuple[ContentBlock, ...]:
        try:
            return _CONTENT.validate_python(payload.get(key, ()))
        except ValidationError as exc:
            raise SnapshotProjectionError(f"{key} is invalid") from exc

    @staticmethod
    def _string(
        payload: JsonObject,
        key: str,
        *,
        default: str | None = None,
        allow_empty: bool = False,
    ) -> str:
        value = payload.get(key, default)
        if not isinstance(value, str) or (not allow_empty and not value):
            raise SnapshotProjectionError(f"{key} must be a string")
        return value

    @staticmethod
    def _optional_string(payload: JsonObject, key: str) -> str | None:
        value = payload.get(key)
        return value if isinstance(value, str) else None
