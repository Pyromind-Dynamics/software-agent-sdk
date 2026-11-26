from __future__ import annotations

from dataclasses import dataclass
from re import Pattern


@dataclass(frozen=True, slots=True)
class ModelPrompts:
    model_ids: Pattern[str]
    preparation: str
    execution: str


@dataclass(frozen=True, slots=True)
class AgentProfileSpec:
    """Server-owned capability contract for one Agent."""

    name: str
    tag_value: str
    usage_id: str
    system_prompt: str | None
    skill_allowlist: tuple[str, ...]
    tool_allowlist: tuple[str, ...]
