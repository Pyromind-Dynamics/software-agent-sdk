from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pyromind_agent_server.agent_profiles.echomind_manager import (
    ECHOMIND_MANAGER_PROFILE,
)
from pyromind_runtime.application.product_instructions import DATA_FLYWHEEL_INSTRUCTIONS
from pyromind_runtime.domain.preparation import PreparationPolicy
from pyromind_runtime.domain.reward_bundle import (
    START_REWARD_REVIEW,
)
from pyromind_runtime.ports.harness import DelegatedRole


@dataclass(frozen=True, slots=True)
class ConversationProfile:
    route_prefix: str
    configuration_version: int = 0
    profile_id: str | None = None
    harness_id: str | None = None
    instructions: str = ""
    tool_allowlist: tuple[str, ...] | None = None
    skill_allowlist: tuple[str, ...] | None = None
    development_tool_allowlist: tuple[str, ...] | None = None
    development_skill_allowlist: tuple[str, ...] | None = None
    command_allowlist: tuple[str, ...] | None = None
    delegated_roles: tuple[DelegatedRole, ...] | None = None
    conversation_marker: str | None = None
    preparation_policy: PreparationPolicy | None = None
    execution_only: bool = False


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

_PREPARATION_SKILLS = ("model-evaluation-preparation", "rubric-elicitation")
_PREPARATION_TOOLS = (
    "read",
    "write",
    "edit",
    "terminal",
    "ask_user",
    "set_preparation_plan",
    "publish_echomind_update",
    START_REWARD_REVIEW,
)


def _load_preparation_policy() -> PreparationPolicy:
    configured = os.getenv("PYROMIND_SKILLS_PATH")
    if configured:
        roots = (Path(configured),)
    else:
        package = Path(__file__).resolve()
        roots = (
            package.parents[1] / ".agents/skills",
            package.parents[2] / ".agents/skills",
        )
    for root in roots:
        manifest = root / "model-evaluation-preparation/workflow-nodes.json"
        if manifest.is_file():
            return PreparationPolicy.model_validate_json(manifest.read_text())
    raise FileNotFoundError(
        "Preparation workflow catalog is missing from the configured skills directory"
    )


_PREPARATION_POLICY = _load_preparation_policy()


ECHOMIND_PRODUCT_PROFILE = ConversationProfile(
    configuration_version=34,
    route_prefix="/api/v2/echomind/conversations",
    execution_only=True,
    profile_id=ECHOMIND_MANAGER_PROFILE.name,
    harness_id="pi",
    instructions="""Before Project creation only, read model-evaluation-preparation.
Use the Skill to determine the information needed for the user's task. Read
data and knowledge with the available workspace tools. For training without a
programmatically verifiable answer, consult rubric-elicitation to choose and
confirm the scoring method. Use ask_user only for
essential information or decisions that cannot be established. Once ready,
submit a compact review plan with set_preparation_plan(plan, entry_task_id).
After each new preparation request or answer, submit the reviewed plan again
before ending the turn, even when the saved draft does not need changes.
Include confirmed inputs and constraints in the plan, not a separate Checklist.
The user may revise it in chat and creates/starts the Project with the UI button.
Use the Sandbox to inspect data and prepare files. Start business tasks only after
the user creates the Project.
After creation, preparation is finished. Use the accepted plan supplied in the
Project notification to create public_data/workflowPLAN.json with write, then
execute the accepted entry task directly.
Before finishing an execution round, give a concise public result in the user's
language. State what was completed, or what is still waiting or blocked.
Deliver requested summaries in that result. Keep internal reasoning,
credentials and tool transcripts out of the result.
"""
    + (ECHOMIND_MANAGER_PROFILE.system_prompt or ""),
    tool_allowlist=_PREPARATION_TOOLS,
    skill_allowlist=_PREPARATION_SKILLS,
    delegated_roles=(),
    development_tool_allowlist=ECHOMIND_MANAGER_PROFILE.tool_allowlist,
    development_skill_allowlist=ECHOMIND_MANAGER_PROFILE.skill_allowlist,
    command_allowlist=(
        "user_message",
        "input_response",
        "cancel",
        "create_project",
        "review_preparation",
        "enqueue_task",
        "permission_response",
        "workflow_execution",
    ),
    conversation_marker=ECHOMIND_MANAGER_PROFILE.tag_value,
    preparation_policy=_PREPARATION_POLICY,
)
