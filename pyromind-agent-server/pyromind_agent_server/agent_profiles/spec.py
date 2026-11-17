from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentProfileSpec:
    """Server-owned capability contract for one Agent."""

    name: str
    tag_value: str
    usage_id: str
    system_prompt: str | None
    skill_allowlist: tuple[str, ...]
    tool_allowlist: tuple[str, ...]
