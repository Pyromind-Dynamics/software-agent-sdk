from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from pyromind_runtime.contracts.events import HarnessEvent, ProductEvent
from pyromind_runtime.contracts.sandbox import WorkspaceRef
from pyromind_runtime.product.models import WorkflowState


_WORKFLOW_RELATIVE_PATH = Path("public_data/workflow_canvas/workflow.py")
_STATE_RELATIVE_PATH = Path("public_data/workflow_canvas/state.json")
_MAX_WORKFLOW_BYTES = 2 * 1024 * 1024
_MAX_STATE_BYTES = 8 * 1024 * 1024


class WorkflowProjectionError(RuntimeError):
    pass


class WorkflowStateReader(Protocol):
    def read(
        self,
        workspace: WorkspaceRef,
        resource_id: str,
        version_hint: str,
    ) -> WorkflowState | None: ...


class FileWorkflowStateReader:
    """Read the authoritative workflow without consuming a tool result payload."""

    def read(
        self,
        workspace: WorkspaceRef,
        resource_id: str,
        version_hint: str,
    ) -> WorkflowState | None:
        workspace_root = Path(workspace.root).resolve()
        state_path = self._bounded_path(workspace_root, _STATE_RELATIVE_PATH)
        version = self._read_canvas_version(state_path, version_hint)
        if version is not None:
            return WorkflowState(
                resource_id=resource_id,
                version=self._required_string(version, "versionId"),
                dsl=self._required_string(version, "workflowDslData", allow_empty=True),
                canvas=self._optional_object(version, "workflowXyflowData"),
            )

        workflow_path = self._bounded_path(workspace_root, _WORKFLOW_RELATIVE_PATH)
        if not workflow_path.is_file():
            return None
        self._validate_size(workflow_path, _MAX_WORKFLOW_BYTES)
        return WorkflowState(
            resource_id=resource_id,
            version=version_hint,
            dsl=workflow_path.read_text(encoding="utf-8"),
        )

    def _read_canvas_version(
        self,
        path: Path,
        event_id: str,
    ) -> dict[str, JsonValue] | None:
        if not path.is_file():
            return None
        self._validate_size(path, _MAX_STATE_BYTES)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkflowProjectionError("workflow canvas state is invalid") from exc
        if not isinstance(payload, dict):
            raise WorkflowProjectionError("workflow canvas state must be an object")
        event_snapshots = payload.get("eventSnapshots")
        versions = payload.get("versions")
        if not isinstance(event_snapshots, dict) or not isinstance(versions, dict):
            return None
        record = event_snapshots.get(event_id)
        if not isinstance(record, dict):
            return None
        version_id = record.get("versionId")
        if not isinstance(version_id, str) or not version_id:
            raise WorkflowProjectionError("workflow snapshot versionId is invalid")
        version = versions.get(version_id)
        if not isinstance(version, dict):
            raise WorkflowProjectionError("workflow snapshot version is missing")
        if version.get("isDeleted") is True:
            return None
        return version

    @staticmethod
    def _bounded_path(root: Path, relative_path: Path) -> Path:
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise WorkflowProjectionError(
                f"workflow resource escapes workspace: {relative_path}"
            ) from exc
        return candidate

    @staticmethod
    def _validate_size(path: Path, maximum: int) -> None:
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise WorkflowProjectionError("workflow resource cannot be read") from exc
        if size > maximum:
            raise WorkflowProjectionError(f"workflow resource exceeds {maximum} bytes")

    @staticmethod
    def _required_string(
        payload: dict[str, JsonValue],
        key: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or (not allow_empty and not value):
            raise WorkflowProjectionError(f"workflow {key} is invalid")
        return value

    @staticmethod
    def _optional_object(
        payload: dict[str, JsonValue],
        key: str,
    ) -> dict[str, JsonValue] | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise WorkflowProjectionError(f"workflow {key} is invalid")
        return value


class WorkflowProductProjector:
    def __init__(self, reader: WorkflowStateReader | None = None) -> None:
        self._reader = reader or FileWorkflowStateReader()

    def project(
        self,
        conversation_id: str,
        workspace: WorkspaceRef,
        event: HarnessEvent,
    ) -> ProductEvent | None:
        if event.type != "resource.updated":
            raise WorkflowProjectionError("expected resource.updated")
        resource_type = self._required_payload_string(event, "resource_type")
        if resource_type != "workflow":
            return None
        resource_id = self._required_payload_string(event, "resource_id")
        version_hint = self._required_payload_string(event, "version")
        workflow = self._reader.read(
            workspace,
            resource_id,
            version_hint,
        )
        if workflow is None:
            return None
        return ProductEvent(
            event_id=event.event_id,
            conversation_id=conversation_id,
            occurred_at=event.occurred_at,
            type="workflow.updated",
            run_id=event.run_id,
            payload=workflow.model_dump(mode="json"),
        )

    @staticmethod
    def _required_payload_string(event: HarnessEvent, key: str) -> str:
        value = event.payload.get(key)
        if not isinstance(value, str) or not value:
            raise WorkflowProjectionError(f"resource.updated {key} is invalid")
        return value
