from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field
from pyromind_runtime.domain.content import JsonObject

from openhands.agent_server.pyromind_router import PyromindLLMConfig


class CreateConversationRequest(BaseModel):
    """The proven Pyromind create inputs, exposed through the Product API."""

    llm: PyromindLLMConfig
    message: str | None = None
    workflow_xyflow: JsonObject | None = Field(
        default=None,
        validation_alias=AliasChoices("workflow_xyflow", "workflowXyflow"),
    )
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True, "extra": "forbid"}


class ForkConversationRequest(BaseModel):
    event_id: str = Field(
        alias="eventId",
        min_length=1,
        description="Existing Pyromind workflow snapshot event id.",
    )
    title: str | None = Field(default=None, max_length=200)

    model_config = {"populate_by_name": True, "extra": "forbid"}
