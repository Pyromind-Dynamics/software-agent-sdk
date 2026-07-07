from __future__ import annotations

from pathlib import Path

from openhands.agent_server.pyromind_constants import PYROMIND_WORKFLOW_EVENT_KEY
from openhands.agent_server.workflow_canvas_models import (
    SaveWorkflowCanvasEventSnapshotRequest,
)
from openhands.agent_server.workflow_canvas_store import (
    FileWorkflowCanvasStore,
    WorkflowCanvasStoreError,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class WorkflowCanvasSnapshotHook:
    def __init__(self, conversation_dir: Path | str, session_id: str) -> None:
        self.store = FileWorkflowCanvasStore(conversation_dir, session_id)

    def save_in_snapshot(
        self,
        *,
        event_id: str | None,
        workflow_dsl_data: str | None,
    ) -> None:
        if event_id is None or workflow_dsl_data is None:
            return
        self._save_snapshot(
            SaveWorkflowCanvasEventSnapshotRequest(
                eventId=event_id,
                snapshotRole="in",
                workflowDslData=workflow_dsl_data,
                summary="用户输入时的 workflow DSL 快照",
                createdBy="workflow_canvas_snapshot_hook",
            )
        )

    def save_out_snapshot(
        self,
        *,
        event_id: str,
        workflow_dsl_data: str,
        parent_user_message_event_id: str | None,
        summary: str | None,
    ) -> None:
        self._save_snapshot(
            SaveWorkflowCanvasEventSnapshotRequest(
                eventId=event_id,
                snapshotRole="out",
                workflowDslData=workflow_dsl_data,
                parentUserMessageEventId=parent_user_message_event_id,
                eventType=PYROMIND_WORKFLOW_EVENT_KEY,
                summary=summary or "Agent workflow 输出快照",
                createdBy="workflow_canvas_snapshot_hook",
            )
        )

    def _save_snapshot(self, request: SaveWorkflowCanvasEventSnapshotRequest) -> None:
        try:
            self.store.save_event_snapshot(request)
        except WorkflowCanvasStoreError:
            logger.warning(
                "Failed to save workflow canvas snapshot for event_id=%s",
                request.event_id,
                exc_info=True,
            )
