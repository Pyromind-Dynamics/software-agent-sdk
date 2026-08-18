from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from pyromind_runtime.contracts.base import ContractModel
from pyromind_runtime.contracts.content import ContentBlock, JsonObject


type ToolRiskLevel = Literal["low", "medium", "high"]


class ToolSpec(ContractModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    description: str = Field(min_length=1)
    input_schema: JsonObject
    timeout_seconds: float = Field(gt=0)
    risk_level: ToolRiskLevel


class ToolResult(ContractModel):
    content: tuple[ContentBlock, ...] = ()
    details: JsonObject | None = None
    is_error: bool = False
    error_code: str | None = None

    @model_validator(mode="after")
    def validate_error_code(self) -> Self:
        if self.is_error and self.error_code is None:
            raise ValueError("error_code is required when is_error is true")
        if not self.is_error and self.error_code is not None:
            raise ValueError("error_code is only valid when is_error is true")
        return self
