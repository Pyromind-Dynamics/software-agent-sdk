from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from pyromind_runtime.contracts.base import ContractModel
from pyromind_runtime.contracts.harness import (
    PermissionDecision,
    UserMessageCommand,
)


class CancelCommand(ContractModel):
    command_id: str = Field(min_length=1)
    type: Literal["cancel"] = "cancel"


class PermissionResponseCommand(ContractModel):
    command_id: str = Field(min_length=1)
    type: Literal["permission_response"] = "permission_response"
    permission_id: str = Field(min_length=1)
    decision: PermissionDecision
    reason: str | None = None


type ProductCommand = Annotated[
    UserMessageCommand | CancelCommand | PermissionResponseCommand,
    Field(discriminator="type"),
]
