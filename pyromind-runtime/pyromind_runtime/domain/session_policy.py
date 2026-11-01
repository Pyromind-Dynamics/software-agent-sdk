from __future__ import annotations

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject


class SessionPolicy(ContractModel):
    tool_allowlist: tuple[str, ...] | None = None
    skill_allowlist: tuple[str, ...] | None = None
    configuration_version: int = 0
    product_instructions: str | None = None
    delegated_roles: tuple[JsonObject, ...] | None = None
    managed_delegation: bool | None = None
    command_allowlist: tuple[str, ...] | None = None
