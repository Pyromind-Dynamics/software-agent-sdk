"""Harness-neutral return address carried unchanged by Studio."""

from __future__ import annotations

from typing import Literal, Self
from uuid import UUID

from pydantic import Field, ValidationError, field_validator, model_validator

from pyromind_runtime.domain.base import ContractModel


class TaskOrigin(ContractModel):
    type: Literal["main", "subagent"] = "main"
    subagent_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_recipient(self) -> Self:
        if (self.type == "subagent") != (self.subagent_id is not None):
            raise ValueError("only subagent origins require subagent_id")
        return self


class TaskCorrelation(TaskOrigin):
    version: Literal[1] = 1
    source: Literal["agent"] = "agent"
    conversation_id: str = Field(min_length=1)
    task_kind: str = Field(min_length=1)

    @field_validator("conversation_id")
    @classmethod
    def canonicalize_conversation(cls, value: str) -> str:
        try:
            return UUID(value).hex
        except ValueError:
            return value

    @property
    def origin(self) -> TaskOrigin:
        return TaskOrigin(type=self.type, subagent_id=self.subagent_id)

    def to_out_id(self) -> str:
        return self.model_dump_json(exclude_none=True)


def parse_task_correlation(out_id: object) -> TaskCorrelation | None:
    if not isinstance(out_id, str):
        return None
    try:
        correlation = TaskCorrelation.model_validate_json(out_id)
        required = {"version", "source", "conversation_id", "type", "task_kind"}
        return correlation if required <= correlation.model_fields_set else None
    except ValidationError:
        return None
