from __future__ import annotations

import ast
from datetime import datetime
from typing import Literal, cast

from pydantic import JsonValue, TypeAdapter, ValidationError

from pyromind_runtime.domain.base import as_utc
from pyromind_runtime.domain.content import ContentBlock, JsonObject, TextContent
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    ExternalTaskState,
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
from pyromind_runtime.domain.user_input import PendingUserInput


_CONTENT = TypeAdapter(tuple[ContentBlock, ...])
_STATUSES = frozenset(
    {
        "idle",
        "running",
        "paused",
        "waiting_for_confirmation",
        "waiting_for_input",
        "waiting_for_external_task",
        "finished",
        "error",
        "stuck",
        "deleting",
        "closed",
    }
)
# Background task completion events must not overwrite terminal states with idle or
# running.
_SETTLED_STATUSES = frozenset({"paused", "error", "stuck", "deleting", "closed"})
_MessageRole = Literal["user", "assistant", "system"]
_OperationCategory = Literal["tool", "subagent", "observation"]
_CompactionStatus = Literal["started", "completed", "skipped", "failed"]
_NoticeSeverity = Literal["info", "warning", "error"]
_InputOutcome = Literal["answered", "cancelled", "interrupted"]


class SnapshotProjectionError(RuntimeError):
    pass


def _latest(current: datetime | None, candidate: datetime) -> datetime:
    """Return the later of two instants, tolerating timestamps without an offset.

    Domain models normalize timestamps on validation, but ``model_copy`` and
    direct ``reduce`` calls skip that step, so normalize again before comparing
    aware and legacy naive values.
    """
    if current is None:
        return candidate
    return max(as_utc(current), as_utc(candidate))


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
        return updated.model_copy(
            update={
                "through_seq": event.seq,
                "updated_at": _latest(snapshot.updated_at, event.occurred_at),
            }
        )

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
        if (
            status != "running"
            and event.run_id is not None
            and snapshot.active_run_id is not None
            and event.run_id != snapshot.active_run_id
        ):
            return snapshot
        if snapshot.pending_inputs:
            waiting_run = snapshot.active_run_id or snapshot.pending_inputs[0].run_id
            if event.run_id is not None and event.run_id != waiting_run:
                return snapshot
            if status in {"running", "idle", "finished"}:
                status = "waiting_for_input"
        if snapshot.pending_permissions and status in {"idle", "finished"}:
            status = "waiting_for_confirmation"
        return snapshot.model_copy(
            update={
                "status": status,
                "active_run_id": (
                    event.run_id or snapshot.active_run_id
                    if status in {"running", "waiting_for_input"}
                    else None
                ),
            }
        )

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
        command_id = event.payload.get("command_id")
        if isinstance(command_id, str):
            message = message.model_copy(update={"command_id": command_id})
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
        command_id = event.payload.get("command_id")
        if isinstance(command_id, str):
            update["command_id"] = command_id
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
        if item.name == "ask_user" and isinstance(item.details, dict):
            user_input = item.details.get("user_input")
            if isinstance(user_input, dict) and user_input.get("status") in {
                "cancelled",
                "interrupted",
            }:
                details = update.get("details", item.details)
                update["details"] = {
                    **(details if isinstance(details, dict) else {}),
                    "user_input": user_input,
                }
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
        # Input and permission requests are independent; show input waiting when
        # present.
        status = (
            "waiting_for_input"
            if snapshot.pending_inputs
            else "waiting_for_confirmation"
        )
        return snapshot.model_copy(
            update={
                "status": status,
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
        if snapshot.pending_inputs:
            status = "waiting_for_input"
        elif pending:
            status = "waiting_for_confirmation"
        else:
            status = "idle" if permission_id.startswith("workflow-run:") else "running"
        return snapshot.model_copy(
            update={"pending_permissions": pending, "status": status}
        )

    def _on_input_requested(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        pending = PendingUserInput.model_validate(
            {**event.payload, "run_id": event.run_id}
        )
        if (
            snapshot.active_run_id is not None
            and pending.run_id != snapshot.active_run_id
        ):
            return snapshot
        if any(
            item.request_id == pending.request_id for item in snapshot.pending_inputs
        ):
            return snapshot
        # Keep the owning active_run_id while waiting; the run is active and awaiting
        # user input.
        return snapshot.model_copy(
            update={
                "pending_inputs": (*snapshot.pending_inputs, pending),
                "status": "waiting_for_input",
                "active_run_id": pending.run_id,
            }
        )

    def _on_input_resolved(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        request_id = self._string(event.payload, "request_id")
        outcome = self._string(event.payload, "outcome")
        if outcome not in {"answered", "cancelled", "interrupted"}:
            return self._notice(
                snapshot,
                event,
                code="invalid_input_outcome",
                message=f"Invalid input resolution outcome: {outcome}",
            )
        remaining = tuple(
            item for item in snapshot.pending_inputs if item.request_id != request_id
        )
        resolved = next(
            (item for item in snapshot.pending_inputs if item.request_id == request_id),
            None,
        )
        if resolved is None or (
            event.run_id is not None and event.run_id != resolved.run_id
        ):
            return snapshot
        run_id = event.run_id or (resolved.run_id if resolved is not None else None)
        if remaining:
            status: str = "waiting_for_input"
            update: dict[str, object] = {"pending_inputs": remaining, "status": status}
        elif snapshot.pending_permissions:
            update = {"pending_inputs": remaining, "status": "waiting_for_confirmation"}
        elif outcome == "answered":
            update = {
                "pending_inputs": remaining,
                "status": "running",
                "active_run_id": run_id,
            }
        elif outcome == "cancelled":
            # Persist cancellation and suspension together; the run-end event confirms
            # paused after abort.
            update = {
                "pending_inputs": remaining,
                "status": "paused",
                "active_run_id": None,
            }
        else:
            update = {
                "pending_inputs": remaining,
                "status": "paused",
                "active_run_id": None,
            }
        if outcome != "answered":
            # Persist the gate with the resolution so a crash before the
            # execution.policy.updated event cannot reopen automatic turns.
            update["automatic_turns_suspended"] = True
            operation = (
                self._find(snapshot, "operation", resolved.operation_id)
                if resolved.operation_id is not None
                else None
            )
            if (
                isinstance(operation, TimelineOperation)
                and operation.name == "ask_user"
            ):
                details = (
                    operation.details if isinstance(operation.details, dict) else {}
                )
                snapshot = self._replace(
                    snapshot,
                    operation.model_copy(
                        update={
                            "details": {
                                **details,
                                "user_input": {
                                    "request_id": request_id,
                                    "status": outcome,
                                },
                            }
                        }
                    ),
                )
        return snapshot.model_copy(update=update)

    def _on_execution_policy_updated(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        suspended = event.payload.get("automatic_turns_suspended")
        if not isinstance(suspended, bool):
            return self._notice(
                snapshot,
                event,
                code="invalid_execution_policy",
                message="automatic_turns_suspended must be a boolean",
            )
        return snapshot.model_copy(update={"automatic_turns_suspended": suspended})

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

    def _on_external_task_submitted(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        task = ExternalTaskState.model_validate(event.payload)
        existing = next(
            (item for item in snapshot.external_tasks if item.task_id == task.task_id),
            None,
        )
        if existing is not None and self._terminal_external_task(existing.status):
            return snapshot
        tasks = tuple(
            existing
            for existing in snapshot.external_tasks
            if existing.task_id != task.task_id
        )
        status = snapshot.status
        if not self._background_status_locked(snapshot) and (
            task.kind != "historical_experience"
        ):
            status = "waiting_for_external_task"
        return snapshot.model_copy(
            update={"external_tasks": (*tasks, task), "status": status}
        )

    def _on_external_task_updated(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        return self._update_external_task(snapshot, event)

    def _on_external_task_bound(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        task_id = self._string(event.payload, "task_id")
        graph_run_id = self._string(event.payload, "graph_run_id")
        tasks = list(snapshot.external_tasks)
        for index, existing in enumerate(tasks):
            if existing.task_id != task_id:
                continue
            update = {
                "kind": "graph_await",
                "run_id": graph_run_id,
                "graph_run_id": graph_run_id,
            }
            output_dir = event.payload.get("output_dir")
            if isinstance(output_dir, str) and output_dir:
                update["output_dir"] = output_dir
            tasks[index] = existing.model_copy(update=update)
            return snapshot.model_copy(update={"external_tasks": tuple(tasks)})
        return snapshot

    def _on_external_task_completed(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        updated = self._update_external_task(snapshot, event)
        if event.payload.get("kind") == "historical_experience":
            return updated
        if self._background_status_locked(updated):
            return updated
        running = any(
            task.status in {"pending", "running"} for task in updated.external_tasks
        )
        return updated.model_copy(
            update={"status": "waiting_for_external_task" if running else "idle"}
        )

    @staticmethod
    def _background_status_locked(snapshot: ConversationSnapshot) -> bool:
        """Keep paused, settled, and input-waiting states during background updates."""
        return (
            snapshot.status in _SETTLED_STATUSES
            or snapshot.status == "waiting_for_input"
            or bool(snapshot.pending_inputs)
        )

    def _update_external_task(
        self, snapshot: ConversationSnapshot, event: ProductEvent
    ) -> ConversationSnapshot:
        task_id = self._string(event.payload, "task_id")
        tasks = list(snapshot.external_tasks)
        for index, existing in enumerate(tasks):
            if existing.task_id == task_id:
                incoming_status = event.payload.get("status")
                if (
                    self._terminal_external_task(existing.status)
                    and incoming_status != existing.status
                ):
                    return snapshot
                tasks[index] = existing.model_copy(update=dict(event.payload))
                return snapshot.model_copy(update={"external_tasks": tuple(tasks)})
        return snapshot

    @staticmethod
    def _terminal_external_task(status: str) -> bool:
        return status in {"succeeded", "failed", "terminated", "stopped"}

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
