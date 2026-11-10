from __future__ import annotations

from pyromind_runtime.domain.base import ContractModel
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.preparation import PreparationPolicy


class SessionPolicy(ContractModel):
    preparation_policy: PreparationPolicy | None = None
    tool_allowlist: tuple[str, ...] | None = None
    skill_allowlist: tuple[str, ...] | None = None
    development_tool_allowlist: tuple[str, ...] | None = None
    development_skill_allowlist: tuple[str, ...] | None = None
    configuration_version: int = 0
    product_instructions: str | None = None
    delegated_roles: tuple[JsonObject, ...] | None = None
    command_allowlist: tuple[str, ...] | None = None
