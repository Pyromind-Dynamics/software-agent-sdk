from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from pyromind_runtime.contracts.content import (
    ContentBlock,
    JsonObject,
    TextContentBlock,
)
from pyromind_runtime.contracts.events import ProductEvent
from pyromind_runtime.product.models import (
    ConversationSnapshot,
    SnapshotMessage,
    SnapshotOperation,
    SnapshotPermission,
    SnapshotUsage,
    WorkflowState,
)


_CONTENT_BLOCKS = TypeAdapter(tuple[ContentBlock, ...])


class SnapshotProjectionError(RuntimeError):
    pass


class ConversationSnapshotProjector:
    def reduce(
        self,
        snapshot: ConversationSnapshot,
        event: ProductEvent,
    ) -> ConversationSnapshot:
        if event.seq != snapshot.through_seq + 1:
            raise SnapshotProjectionError(
                f"expected seq {snapshot.through_seq + 1}, found {event.seq}"
            )

        updated = snapshot
        if event.type == "run.started":
            updated = snapshot.model_copy(update={"status": "running"})
        elif event.type == "run.completed":
            updated = snapshot.model_copy(update={"status": "idle"})
        elif event.type == "run.failed":
            updated = snapshot.model_copy(update={"status": "failed"})
        elif event.type == "message.started":
            updated = self._start_message(
                snapshot,
                event.payload,
                event.run_id,
                event.seq,
            )
        elif event.type == "message.delta":
            updated = self._append_message_delta(snapshot, event.payload)
        elif event.type == "message.completed":
            updated = self._complete_message(snapshot, event.payload)
        elif event.type == "operation.started":
            updated = self._start_operation(
                snapshot,
                event.payload,
                event.run_id,
                event.seq,
            )
        elif event.type == "operation.progress":
            updated = self._update_operation(snapshot, event.payload, "running")
        elif event.type == "operation.completed":
            updated = self._update_operation(snapshot, event.payload, "completed")
        elif event.type == "operation.failed":
            updated = self._update_operation(snapshot, event.payload, "failed")
        elif event.type == "permission.requested":
            updated = self._request_permission(snapshot, event.payload)
        elif event.type == "permission.resolved":
            updated = self._resolve_permission(snapshot, event.payload)
        elif event.type == "usage.updated":
            updated = snapshot.model_copy(
                update={"usage": SnapshotUsage.model_validate(event.payload)}
            )
        elif event.type == "workflow.updated":
            updated = snapshot.model_copy(
                update={"workflow": WorkflowState.model_validate(event.payload)}
            )

        return updated.model_copy(update={"through_seq": event.seq})

    def _start_message(
        self,
        snapshot: ConversationSnapshot,
        payload: JsonObject,
        run_id: str | None,
        started_seq: int,
    ) -> ConversationSnapshot:
        message_id = self._required_string(payload, "message_id")
        if any(message.message_id == message_id for message in snapshot.messages):
            raise SnapshotProjectionError(f"duplicate message_id: {message_id}")
        message = SnapshotMessage.model_validate(
            {
                "message_id": message_id,
                "started_seq": started_seq,
                "role": self._required_string(payload, "role"),
                "content": self._content(payload),
                "status": "streaming",
                "run_id": run_id,
            }
        )
        return snapshot.model_copy(update={"messages": (*snapshot.messages, message)})

    def _append_message_delta(
        self,
        snapshot: ConversationSnapshot,
        payload: JsonObject,
    ) -> ConversationSnapshot:
        message_id = self._required_string(payload, "message_id")
        text = self._required_string(payload, "text", allow_empty=True)
        message = self._find_message(snapshot, message_id)
        content = message.content
        if content and isinstance(content[-1], TextContentBlock):
            merged = content[-1].model_copy(update={"text": content[-1].text + text})
            content = (*content[:-1], merged)
        else:
            content = (*content, TextContentBlock(text=text))
        return self._replace_message(
            snapshot,
            message.model_copy(update={"content": content}),
        )

    def _complete_message(
        self,
        snapshot: ConversationSnapshot,
        payload: JsonObject,
    ) -> ConversationSnapshot:
        message_id = self._required_string(payload, "message_id")
        message = self._find_message(snapshot, message_id)
        update: dict[str, object] = {"status": "completed"}
        if "content" in payload:
            update["content"] = self._content(payload)
        return self._replace_message(snapshot, message.model_copy(update=update))

    def _start_operation(
        self,
        snapshot: ConversationSnapshot,
        payload: JsonObject,
        run_id: str | None,
        started_seq: int,
    ) -> ConversationSnapshot:
        operation_id = self._required_string(payload, "operation_id")
        if any(
            operation.operation_id == operation_id for operation in snapshot.operations
        ):
            raise SnapshotProjectionError(f"duplicate operation_id: {operation_id}")
        operation = SnapshotOperation.model_validate(
            {
                "operation_id": operation_id,
                "started_seq": started_seq,
                "name": self._required_string(payload, "name"),
                "status": "running",
                "arguments": self._object(payload, "arguments"),
                "run_id": run_id,
            }
        )
        return snapshot.model_copy(
            update={"operations": (*snapshot.operations, operation)}
        )

    def _update_operation(
        self,
        snapshot: ConversationSnapshot,
        payload: JsonObject,
        status: str,
    ) -> ConversationSnapshot:
        operation_id = self._required_string(payload, "operation_id")
        operation = self._find_operation(snapshot, operation_id)
        update: dict[str, object] = {"status": status}
        if "content" in payload:
            update["output"] = self._content(payload)
        if "details" in payload:
            update["details"] = self._optional_object(payload, "details")
        if "error_code" in payload:
            error_code = payload["error_code"]
            if error_code is not None and not isinstance(error_code, str):
                raise SnapshotProjectionError("error_code must be a string or null")
            update["error_code"] = error_code
        return self._replace_operation(
            snapshot,
            operation.model_copy(update=update),
        )

    def _request_permission(
        self,
        snapshot: ConversationSnapshot,
        payload: JsonObject,
    ) -> ConversationSnapshot:
        permission = SnapshotPermission.model_validate(payload)
        if any(
            item.permission_id == permission.permission_id
            for item in snapshot.pending_permissions
        ):
            raise SnapshotProjectionError(
                f"duplicate permission_id: {permission.permission_id}"
            )
        return snapshot.model_copy(
            update={
                "status": "waiting_permission",
                "pending_permissions": (*snapshot.pending_permissions, permission),
            }
        )

    def _resolve_permission(
        self,
        snapshot: ConversationSnapshot,
        payload: JsonObject,
    ) -> ConversationSnapshot:
        permission_id = self._required_string(payload, "permission_id")
        pending = tuple(
            item
            for item in snapshot.pending_permissions
            if item.permission_id != permission_id
        )
        if len(pending) == len(snapshot.pending_permissions):
            raise SnapshotProjectionError(
                f"permission_id is not pending: {permission_id}"
            )
        status = "waiting_permission" if pending else "running"
        return snapshot.model_copy(
            update={"status": status, "pending_permissions": pending}
        )

    @staticmethod
    def _find_message(
        snapshot: ConversationSnapshot,
        message_id: str,
    ) -> SnapshotMessage:
        for message in snapshot.messages:
            if message.message_id == message_id:
                return message
        raise SnapshotProjectionError(f"message_id not found: {message_id}")

    @staticmethod
    def _find_operation(
        snapshot: ConversationSnapshot,
        operation_id: str,
    ) -> SnapshotOperation:
        for operation in snapshot.operations:
            if operation.operation_id == operation_id:
                return operation
        raise SnapshotProjectionError(f"operation_id not found: {operation_id}")

    @staticmethod
    def _replace_message(
        snapshot: ConversationSnapshot,
        replacement: SnapshotMessage,
    ) -> ConversationSnapshot:
        messages = tuple(
            replacement if item.message_id == replacement.message_id else item
            for item in snapshot.messages
        )
        return snapshot.model_copy(update={"messages": messages})

    @staticmethod
    def _replace_operation(
        snapshot: ConversationSnapshot,
        replacement: SnapshotOperation,
    ) -> ConversationSnapshot:
        operations = tuple(
            replacement if item.operation_id == replacement.operation_id else item
            for item in snapshot.operations
        )
        return snapshot.model_copy(update={"operations": operations})

    @staticmethod
    def _content(payload: JsonObject) -> tuple[ContentBlock, ...]:
        try:
            return _CONTENT_BLOCKS.validate_python(payload.get("content", ()))
        except ValidationError as exc:
            raise SnapshotProjectionError("content is invalid") from exc

    @staticmethod
    def _required_string(
        payload: JsonObject,
        key: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or (not allow_empty and not value):
            raise SnapshotProjectionError(f"{key} must be a valid string")
        return value

    @staticmethod
    def _object(payload: JsonObject, key: str) -> JsonObject:
        value = payload.get(key, {})
        if not isinstance(value, dict):
            raise SnapshotProjectionError(f"{key} must be an object")
        return value

    @staticmethod
    def _optional_object(payload: JsonObject, key: str) -> JsonObject | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise SnapshotProjectionError(f"{key} must be an object or null")
        return value
