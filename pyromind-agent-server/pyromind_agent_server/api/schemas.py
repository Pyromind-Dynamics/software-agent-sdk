from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

from pyromind_runtime.domain.content import JsonObject


_OPENAI_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


class AgentLLMConfig(BaseModel):
    """前端提交的模型配置，供 Product API 与 Pi 链路使用。"""

    model: str = Field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o"),
    )
    api_key: str | None = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY"),
    )
    base_url: str | None = Field(
        default_factory=lambda: os.environ.get("LLM_BASE_URL"),
    )
    api: Literal["openai-completions", "openai-responses"] | None = None
    context_window: int | None = Field(default=None, gt=0)
    timeout: int | None = Field(
        default=None,
        description=("LLM request timeout in seconds. Server-default 60s when unset."),
    )

    @field_validator("base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: Any) -> str | None:
        if not isinstance(value, str):
            return value

        base_url = value.strip().rstrip("/")
        if not base_url:
            return None

        if base_url.endswith(_OPENAI_CHAT_COMPLETIONS_SUFFIX):
            return base_url[: -len(_OPENAI_CHAT_COMPLETIONS_SUFFIX)]

        return base_url


class CreateConversationRequest(BaseModel):
    """The proven Pyromind create inputs, exposed through the Product API."""

    llm: AgentLLMConfig
    message: str | None = None
    workflow_xyflow: JsonObject | None = Field(
        default=None,
        validation_alias=AliasChoices("workflow_xyflow", "workflowXyflow"),
    )
    user_input_enabled: bool = False
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
