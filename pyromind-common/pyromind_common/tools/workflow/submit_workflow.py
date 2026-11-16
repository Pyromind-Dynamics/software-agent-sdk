"""Agent-facing entry point for the Product Runtime's workflow submission."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, cast
from uuid import uuid4

import httpx
from pydantic import Field, field_validator

from pyromind_common.sdk.tool.registry import register_tool
from pyromind_common.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from pyromind_common.tools.workflow.validate_workflow_dsl import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)
from pyromind_runtime.domain.storage_paths import (
    StoragePathError,
    normalize_workspace_path,
)


if TYPE_CHECKING:
    from pyromind_common.sdk.conversation.base import ToolConversation
    from pyromind_common.sdk.conversation.state import ToolState


PYROMIND_PRODUCT_USER_ID_SECRET = "pyromind_product_user_id"


class SubmitWorkflowAction(Action):
    workflow_path: str = Field(
        min_length=1,
        description=(
            "Existing user Storage workflow DSL path, "
            "e.g. /workspace/project/workflow.py."
        ),
    )

    @field_validator("workflow_path")
    @classmethod
    def validate_workflow_path(cls, value: str) -> str:
        try:
            return normalize_workspace_path(value, field="workflow_path")
        except StoragePathError as exc:
            raise ValueError(str(exc)) from exc


class SubmitWorkflowObservation(Observation):
    request_id: str | None = None
    status: str | None = None
    task_id: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class SubmitWorkflowExecutor(
    ToolExecutor[SubmitWorkflowAction, SubmitWorkflowObservation]
):
    def __call__(
        self,
        action: SubmitWorkflowAction,
        conversation: ToolConversation | None = None,
    ) -> SubmitWorkflowObservation:
        if conversation is None or conversation.id is None:
            return SubmitWorkflowObservation.from_text(
                text="submit_workflow requires a conversation", is_error=True
            )
        state = cast("ToolState", conversation.state)
        cookie = state.secret_registry.get_secret_value(
            PYROMIND_VALIDATE_AUTH_COOKIE_SECRET
        )
        if not cookie:
            return SubmitWorkflowObservation.from_text(
                text="workflow submission authentication is unavailable", is_error=True
            )
        headers: dict[str, str] = {"cookie": cookie}
        user_id = state.secret_registry.get_secret_value(
            PYROMIND_PRODUCT_USER_ID_SECRET
        )
        if user_id:
            headers["x-pyromind-debug-user-id"] = user_id
        forwarded = (getattr(state, "agent_state", None) or {}).get(
            PYROMIND_VALIDATE_HEADERS_STATE_KEY
        )
        if isinstance(forwarded, dict) and isinstance(forwarded.get("x-cluster"), str):
            headers["x-cluster"] = forwarded["x-cluster"]
        base = os.getenv("OH_INTERNAL_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
        conversation_id = str(conversation.id).replace("-", "")
        try:
            response = httpx.post(
                f"{base}/api/v2/pyromind/conversations/{conversation_id}/workflow-executions",
                headers=headers,
                json={"workflow_path": action.workflow_path, "request_id": uuid4().hex},
                timeout=90,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return SubmitWorkflowObservation.from_text(
                text=f"workflow submission failed: {exc}", is_error=True
            )
        if not isinstance(result, dict):
            return SubmitWorkflowObservation.from_text(
                text="workflow submission returned invalid data", is_error=True
            )
        return SubmitWorkflowObservation.from_text(
            text=(
                f"Workflow submission status: {result.get('status')}. "
                f"Task ID: {result.get('task_id') or 'unavailable'}."
                + (f" {result['error']}" if result.get("error") else "")
            ),
            request_id=result.get("request_id"),
            status=result.get("status"),
            task_id=result.get("task_id"),
            summary=result.get("summary")
            if isinstance(result.get("summary"), dict)
            else {},
        )


class SubmitWorkflowTool(
    ToolDefinition[SubmitWorkflowAction, SubmitWorkflowObservation]
):
    @classmethod
    def create(
        cls,
        conv_state: ToolState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        if params:
            raise ValueError("submit_workflow takes no tool configuration")
        return [
            cls(
                description=(
                    "Submit an existing Storage workflow for formal execution. "
                    "The server validates it and creates the platform task directly, "
                    "without a user confirmation dialog. "
                    "Call only when the user requests execution."
                ),
                action_type=SubmitWorkflowAction,
                observation_type=SubmitWorkflowObservation,
                executor=SubmitWorkflowExecutor(),
                annotations=ToolAnnotations(
                    title="submit_workflow",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(SubmitWorkflowTool.name, SubmitWorkflowTool)
