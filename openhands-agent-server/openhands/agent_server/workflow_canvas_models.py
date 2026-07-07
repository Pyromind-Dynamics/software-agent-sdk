from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from openhands.sdk.utils import utc_now


WORKFLOW_CANVAS_SCHEMA_VERSION = 1
WorkflowCanvasSnapshotRole = Literal["in", "out"]


class WorkflowCanvasVersion(BaseModel):
    session_id: str = Field(alias="sessionId")
    version_id: str = Field(alias="versionId")
    version_no: int = Field(alias="versionNo", ge=1)
    workflow_dsl_data: str = Field(alias="workflowDslData")
    summary: str | None = None
    feature: Any | None = None
    created_by: str | None = Field(default=None, alias="createdBy")
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")
    is_deleted: bool = Field(default=False, alias="isDeleted")

    model_config = ConfigDict(populate_by_name=True)


class WorkflowCanvasVersionListItem(BaseModel):
    session_id: str = Field(alias="sessionId")
    version_id: str = Field(alias="versionId")
    version_no: int = Field(alias="versionNo")
    summary: str | None = None
    feature: Any | None = None
    created_by: str | None = Field(default=None, alias="createdBy")
    created_at: datetime = Field(alias="createdAt")

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_version(
        cls, version: WorkflowCanvasVersion
    ) -> WorkflowCanvasVersionListItem:
        return cls.model_validate(version.model_dump(mode="json", by_alias=True))


class WorkflowCanvasEventSnapshotRecord(BaseModel):
    session_id: str = Field(alias="sessionId")
    event_id: str = Field(alias="eventId")
    snapshot_role: WorkflowCanvasSnapshotRole = Field(alias="snapshotRole")
    version_id: str = Field(alias="versionId")
    parent_user_message_event_id: str | None = Field(
        default=None,
        alias="parentUserMessageEventId",
    )
    event_type: str | None = Field(default=None, alias="eventType")
    feature: Any | None = None
    created_at: datetime = Field(default_factory=utc_now, alias="createdAt")

    model_config = ConfigDict(populate_by_name=True)


class WorkflowCanvasEventSnapshot(BaseModel):
    session_id: str = Field(alias="sessionId")
    event_id: str = Field(alias="eventId")
    snapshot_role: WorkflowCanvasSnapshotRole = Field(alias="snapshotRole")
    version_id: str = Field(alias="versionId")
    version_no: int = Field(alias="versionNo")
    workflow_dsl_data: str = Field(alias="workflowDslData")
    summary: str | None = None
    parent_user_message_event_id: str | None = Field(
        default=None,
        alias="parentUserMessageEventId",
    )
    event_type: str | None = Field(default=None, alias="eventType")
    feature: Any | None = None
    created_at: datetime = Field(alias="createdAt")

    model_config = ConfigDict(populate_by_name=True)


class SaveWorkflowCanvasEventSnapshotRequest(BaseModel):
    session_id: str | None = Field(default=None, alias="sessionId")
    event_id: str = Field(alias="eventId", min_length=1)
    snapshot_role: WorkflowCanvasSnapshotRole = Field(alias="snapshotRole")
    workflow_dsl_data: str = Field(alias="workflowDslData")
    parent_user_message_event_id: str | None = Field(
        default=None,
        alias="parentUserMessageEventId",
    )
    event_type: str | None = Field(default=None, alias="eventType")
    summary: str | None = None
    feature: Any | None = None
    created_by: str | None = Field(default=None, alias="createdBy")

    model_config = ConfigDict(populate_by_name=True)


class WorkflowCanvasEventSnapshotBatchRequest(BaseModel):
    event_ids: list[str] = Field(alias="eventIds", min_length=1)

    model_config = ConfigDict(populate_by_name=True)


class WorkflowCanvasEventSnapshotBatchResponse(BaseModel):
    snapshots: dict[str, WorkflowCanvasEventSnapshot]

    model_config = ConfigDict(populate_by_name=True)


class WorkflowCanvasState(BaseModel):
    schema_version: int = Field(
        default=WORKFLOW_CANVAS_SCHEMA_VERSION,
        alias="schemaVersion",
    )
    next_version_no: int = Field(default=1, alias="nextVersionNo", ge=1)
    versions: dict[str, WorkflowCanvasVersion] = Field(default_factory=dict)
    event_snapshots: dict[str, WorkflowCanvasEventSnapshotRecord] = Field(
        default_factory=dict,
        alias="eventSnapshots",
    )

    model_config = ConfigDict(populate_by_name=True)

    @classmethod
    def from_persisted(cls, data: Any) -> WorkflowCanvasState:
        if not isinstance(data, dict):
            return cls.model_validate(data)
        payload = dict(data)
        version = payload.get("schemaVersion", payload.get("schema_version", 1))
        if not isinstance(version, int):
            raise ValueError("WorkflowCanvasState schemaVersion must be an integer")
        if version > WORKFLOW_CANVAS_SCHEMA_VERSION:
            raise ValueError(
                f"WorkflowCanvasState schemaVersion {version} is newer than "
                f"supported version {WORKFLOW_CANVAS_SCHEMA_VERSION}"
            )
        payload["schemaVersion"] = WORKFLOW_CANVAS_SCHEMA_VERSION
        return cls.model_validate(payload)
