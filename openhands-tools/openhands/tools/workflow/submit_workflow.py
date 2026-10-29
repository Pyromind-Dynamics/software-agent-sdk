"""Agent-facing entry point for the Product Runtime's confirmed workflow run."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, cast
from uuid import uuid4

import httpx
from pydantic import Field

from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)
from openhands.tools.workflow.validate_workflow_dsl import (
    PYROMIND_VALIDATE_AUTH_COOKIE_SECRET,
    PYROMIND_VALIDATE_HEADERS_STATE_KEY,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


PYROMIND_PRODUCT_USER_ID_SECRET = "pyromind_product_user_id"


class SubmitWorkflowAction(Action):
    workflow_path: str = Field(
        min_length=1,
        description=("Existing Storage workflow DSL path, e.g. /project/workflow.py."),
    )


class SubmitWorkflowObservation(Observation):
    request_id: str | None = None
    status: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class SubmitWorkflowExecutor(
    ToolExecutor[SubmitWorkflowAction, SubmitWorkflowObservation]
):
    def __call__(
        self,
        action: SubmitWorkflowAction,
        conversation: BaseConversation | None = None,
    ) -> SubmitWorkflowObservation:
        if conversation is None or conversation.id is None:
            return SubmitWorkflowObservation.from_text(
                text="submit_workflow requires a conversation", is_error=True
            )
        state = cast("ConversationState", conversation.state)
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
        try:
            response = httpx.post(
                f"{base}/api/v2/pyromind/conversations/{conversation.id.hex}/workflow-executions",
                headers=headers,
                json={"workflow_path": action.workflow_path, "request_id": uuid4().hex},
                timeout=90,
            )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return SubmitWorkflowObservation.from_text(
                text=f"workflow preparation failed: {exc}", is_error=True
            )
        if not isinstance(result, dict):
            return SubmitWorkflowObservation.from_text(
                text="workflow preparation returned invalid data", is_error=True
            )
        return SubmitWorkflowObservation.from_text(
            text="The workflow is awaiting user confirmation in the Copilot dialog.",
            request_id=result.get("request_id"),
            status=result.get("status"),
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
        conv_state: ConversationState | None = None,  # noqa: ARG003
        **params: Any,
    ) -> Sequence[Self]:
        if params:
            raise ValueError("submit_workflow takes no tool configuration")
        return [
            cls(
                description=(
                    "Request formal execution of an existing Storage workflow. "
                    "The server validates it and opens a user confirmation dialog; "
                    "the platform task is created only after confirmation. "
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
