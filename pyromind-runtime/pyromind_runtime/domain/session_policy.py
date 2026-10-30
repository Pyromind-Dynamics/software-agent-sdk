from __future__ import annotations

from pyromind_runtime.domain.base import ContractModel


class SessionPolicy(ContractModel):
    tool_allowlist: tuple[str, ...] | None = None
    skill_allowlist: tuple[str, ...] | None = None
