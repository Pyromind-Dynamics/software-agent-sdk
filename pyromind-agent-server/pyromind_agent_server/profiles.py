from __future__ import annotations

from dataclasses import dataclass

from pyromind_runtime.application.product_instructions import DATA_FLYWHEEL_INSTRUCTIONS

from openhands.agent_server.agent_profiles.echomind import ECHOMIND_PROFILE
from openhands.agent_server.agent_profiles.echomind_manager import (
    ECHOMIND_MANAGER_PROFILE,
)


@dataclass(frozen=True, slots=True)
class ConversationProfile:
    route_prefix: str
    profile_id: str | None = None
    harness_id: str | None = None
    instructions: str = ""
    tool_allowlist: tuple[str, ...] | None = None
    skill_allowlist: tuple[str, ...] | None = None
    command_allowlist: tuple[str, ...] | None = None
    conversation_marker: str | None = None


_PRODUCT_ROUTING_INSTRUCTIONS = """For Pyromind workflow requests, read the matching
skill before editing exactly
public_data/workflow_canvas/workflow.py, then call validate_workflow_dsl without
dsl_path; pass it only when validating another workspace file. Do not inspect
credentials or work around failed validation authentication.

For model evaluation, baseline benchmarks, post-training evaluation,
ground-truth comparisons or evaluation reports, read the inference-evaluation
skill and follow its instructions.

Route other dataset work before acting. Use the data-processing skill for any
dataset request; its SKILL.md routing table selects the paradigm:
format-conversion for deterministic field, format, structure, regex, keyword,
and length transformations; llm-pipeline for content assessment, DataFlow
operators, LLM processing, images, and multimodal work; environment-processing
for tasks that require a specific sandbox runtime, including embodied robot
data cleaning. If the intent is ambiguous, ask the user first.
Never start both full-run paths for one request."""

DEFAULT_PRODUCT_PROFILE = ConversationProfile(
    route_prefix="/api/v2/pyromind/conversations",
    instructions=DATA_FLYWHEEL_INSTRUCTIONS + "\n" + _PRODUCT_ROUTING_INSTRUCTIONS,
)

ECHOMIND_PRODUCT_PROFILE = ConversationProfile(
    route_prefix="/api/v2/echomind/conversations",
    profile_id=ECHOMIND_MANAGER_PROFILE.name,
    harness_id="pi",
    instructions=ECHOMIND_MANAGER_PROFILE.system_prompt or "",
    tool_allowlist=ECHOMIND_MANAGER_PROFILE.tool_allowlist,
    skill_allowlist=ECHOMIND_MANAGER_PROFILE.skill_allowlist,
    command_allowlist=(
        "user_message",
        "cancel",
        "confirm_experiment_plan",
        "reject_experiment_plan",
    ),
    conversation_marker=ECHOMIND_MANAGER_PROFILE.usage_id,
)

ECHOMIND_CHAT_PROFILE = ConversationProfile(
    route_prefix="/api/echomind/conversations",
    profile_id=ECHOMIND_PROFILE.name,
    harness_id="pi",
    instructions=ECHOMIND_PROFILE.system_prompt or "",
    tool_allowlist=ECHOMIND_PROFILE.tool_allowlist,
    skill_allowlist=ECHOMIND_PROFILE.skill_allowlist,
    conversation_marker=ECHOMIND_PROFILE.tag_value,
)
