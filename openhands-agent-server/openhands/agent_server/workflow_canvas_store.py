from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from openhands.agent_server.persistence.store import (
    _atomic_write_json,
    _ensure_secure_directory,
    _file_lock,
)
from openhands.agent_server.workflow_canvas_models import (
    SaveWorkflowCanvasEventSnapshotRequest,
    WorkflowCanvasEventSnapshot,
    WorkflowCanvasEventSnapshotRecord,
    WorkflowCanvasState,
    WorkflowCanvasVersion,
    WorkflowCanvasVersionListItem,
)
from openhands.sdk.utils import utc_now


class WorkflowCanvasStoreError(RuntimeError):
    pass


class DuplicateWorkflowCanvasEventSnapshotError(WorkflowCanvasStoreError):
    pass


class WorkflowCanvasEventSnapshotNotFoundError(WorkflowCanvasStoreError):
    pass


class WorkflowCanvasVersionNotFoundError(WorkflowCanvasStoreError):
    pass


class FileWorkflowCanvasStore:
    def __init__(self, conversation_dir: Path | str, session_id: str) -> None:
        self.conversation_dir = Path(conversation_dir)
        self.session_id = session_id
        self._dir = self.conversation_dir / "public_data" / "workflow_canvas"
        self._path = self._dir / "state.json"
        self._lock_path = self._dir / ".workflow_canvas.lock"

    def save_event_snapshot(
        self,
        request: SaveWorkflowCanvasEventSnapshotRequest,
    ) -> WorkflowCanvasEventSnapshot:
        with _file_lock(self._lock_path):
            state = self._load_locked()
            existing = state.event_snapshots.get(request.event_id)
            if existing is not None:
                return self._resolve_existing_snapshot(existing, request, state)

            version = self._create_version(
                state,
                workflow_dsl_data=request.workflow_dsl_data,
                workflow_xyflow_data=request.workflow_xyflow_data,
                summary=request.summary,
                feature=request.feature,
                created_by=request.created_by,
            )
            record = WorkflowCanvasEventSnapshotRecord(
                sessionId=self.session_id,
                eventId=request.event_id,
                snapshotRole=request.snapshot_role,
                versionId=version.version_id,
                parentUserMessageEventId=request.parent_user_message_event_id,
                eventType=request.event_type,
                feature=request.feature,
                createdAt=utc_now(),
            )
            state.event_snapshots[record.event_id] = record
            self._save_locked(state)
            return self._to_event_snapshot(record, version)

    def get_event_snapshot(self, event_id: str) -> WorkflowCanvasEventSnapshot:
        with _file_lock(self._lock_path):
            state = self._load_locked()
            record = state.event_snapshots.get(event_id)
            if record is None:
                raise WorkflowCanvasEventSnapshotNotFoundError(
                    f"Workflow canvas event snapshot not found: {event_id}"
                )
            version = self._get_active_version(state, record.version_id)
            return self._to_event_snapshot(record, version)

    def batch_get_event_snapshots(
        self,
        event_ids: list[str],
    ) -> dict[str, WorkflowCanvasEventSnapshot]:
        with _file_lock(self._lock_path):
            state = self._load_locked()
            snapshots: dict[str, WorkflowCanvasEventSnapshot] = {}
            for event_id in event_ids:
                record = state.event_snapshots.get(event_id)
                if record is None:
                    continue
                version = self._get_active_version(state, record.version_id)
                snapshots[event_id] = self._to_event_snapshot(record, version)
            return snapshots

    def get_version(self, version_id: str) -> WorkflowCanvasVersion:
        with _file_lock(self._lock_path):
            state = self._load_locked()
            return self._get_active_version(state, version_id)

    def list_versions(self) -> list[WorkflowCanvasVersionListItem]:
        with _file_lock(self._lock_path):
            state = self._load_locked()
            versions = [
                version for version in state.versions.values() if not version.is_deleted
            ]
            versions.sort(key=lambda version: version.version_no)
            return [
                WorkflowCanvasVersionListItem.from_version(version)
                for version in versions
            ]

    def _load_locked(self) -> WorkflowCanvasState:
        if not self._path.exists():
            return WorkflowCanvasState()
        try:
            with self._path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return WorkflowCanvasState.from_persisted(data)
        except (PermissionError, OSError):
            raise
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise WorkflowCanvasStoreError(
                f"Failed to load workflow canvas state from {self._path}"
            ) from exc

    def _save_locked(self, state: WorkflowCanvasState) -> None:
        _ensure_secure_directory(self._dir)
        data = state.model_dump(mode="json", by_alias=True)
        _atomic_write_json(self._path, data)

    def _create_version(
        self,
        state: WorkflowCanvasState,
        *,
        workflow_dsl_data: str,
        workflow_xyflow_data: dict[str, Any] | None,
        summary: str | None,
        feature: object | None,
        created_by: str | None,
    ) -> WorkflowCanvasVersion:
        version_no = state.next_version_no
        version_id = _version_id(version_no)
        while version_id in state.versions:
            version_no += 1
            version_id = _version_id(version_no)
        version = WorkflowCanvasVersion(
            sessionId=self.session_id,
            versionId=version_id,
            versionNo=version_no,
            workflowDslData=workflow_dsl_data,
            workflowXyflowData=workflow_xyflow_data,
            summary=summary,
            feature=feature,
            createdBy=created_by,
            createdAt=utc_now(),
        )
        state.versions[version.version_id] = version
        state.next_version_no = version_no + 1
        return version

    def _get_active_version(
        self,
        state: WorkflowCanvasState,
        version_id: str,
    ) -> WorkflowCanvasVersion:
        version = state.versions.get(version_id)
        if version is None or version.is_deleted:
            raise WorkflowCanvasVersionNotFoundError(
                f"Workflow canvas version not found: {version_id}"
            )
        return version

    def _resolve_existing_snapshot(
        self,
        record: WorkflowCanvasEventSnapshotRecord,
        request: SaveWorkflowCanvasEventSnapshotRequest,
        state: WorkflowCanvasState,
    ) -> WorkflowCanvasEventSnapshot:
        version = self._get_active_version(state, record.version_id)
        if (
            record.snapshot_role != request.snapshot_role
            or version.workflow_dsl_data != request.workflow_dsl_data
            or version.workflow_xyflow_data != request.workflow_xyflow_data
        ):
            raise DuplicateWorkflowCanvasEventSnapshotError(
                "Workflow canvas event snapshot already exists for "
                f"eventId={request.event_id}"
            )
        return self._to_event_snapshot(record, version)

    def _to_event_snapshot(
        self,
        record: WorkflowCanvasEventSnapshotRecord,
        version: WorkflowCanvasVersion,
    ) -> WorkflowCanvasEventSnapshot:
        return WorkflowCanvasEventSnapshot(
            sessionId=record.session_id,
            eventId=record.event_id,
            snapshotRole=record.snapshot_role,
            versionId=record.version_id,
            versionNo=version.version_no,
            workflowDslData=version.workflow_dsl_data,
            workflowXyflowData=version.workflow_xyflow_data,
            summary=version.summary,
            parentUserMessageEventId=record.parent_user_message_event_id,
            eventType=record.event_type,
            feature=record.feature,
            createdAt=record.created_at,
        )


def _version_id(version_no: int) -> str:
    return f"v{version_no:06d}"
