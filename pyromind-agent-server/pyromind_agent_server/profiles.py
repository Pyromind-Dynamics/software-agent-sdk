from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pyromind_runtime.application.product_instructions import DATA_FLYWHEEL_INSTRUCTIONS
from pyromind_runtime.domain.preparation import PreparationPolicy
from pyromind_runtime.ports.harness import DelegatedRole

from openhands.agent_server.agent_profiles.echomind_manager import (
    ECHOMIND_MANAGER_PROFILE,
)


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

_PREPARATION_SKILLS = ("model-evaluation-preparation",)
_PREPARATION_TOOLS = (
    "read",
    "write",
    "edit",
    "terminal",
    "ask_user",
    "get_preparation",
    "update_preparation",
    "set_preparation_plan",
    "inspect_resource",
)
_CHILD_WORKSPACE_TOOLS = ("read", "write", "edit", "terminal")
_EVALUATION_TOOLS = _CHILD_WORKSPACE_TOOLS + (
    "df_submit_evaluation",
    "df_check_progress",
    "get_storage_url",
    "df_stop_task",
)
_EVALUATION_SKILLS = ("inference-evaluation", "avi-baseline-evaluation")


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
        manifest = root / "model-evaluation-preparation/checklist.json"
        if manifest.is_file():
            return PreparationPolicy.model_validate_json(manifest.read_text())
    raise FileNotFoundError(
        "Model evaluation preparation checklist is missing "
        "from the configured skills directory"
    )


_PREPARATION_POLICY = _load_preparation_policy()


ECHOMIND_PRODUCT_PROFILE = ConversationProfile(
    configuration_version=22,
    route_prefix="/api/v2/echomind/conversations",
    profile_id=ECHOMIND_MANAGER_PROFILE.name,
    harness_id="pi",
    instructions="""Before Project creation only, read model-evaluation-preparation.
Collect the scenario checklist using user messages, data inspection, workspace
scripts and knowledge retrieval. Use get_preparation for gaps and evidence.
Record facts incrementally with update_preparation. After user
input use preparation_revision from ask_user as expected_revision.
On revision conflict, check the returned current facts and revision before retrying.
Once the checklist is complete, save a compact plan using set_preparation_plan.
The user may revise it in chat and creates/starts the Project with the UI button.
Use the Sandbox to inspect data and prepare files. Start business tasks only after
the user creates the Project.
After creation, preparation is finished. Use the accepted plan supplied in the
Project notification to create public_data/workflowPLAN.json with write, then
delegate the entry task through native subagent. Do not register stages, revisions
or business run IDs for delegation.
"""
    + (ECHOMIND_MANAGER_PROFILE.system_prompt or ""),
    tool_allowlist=_PREPARATION_TOOLS,
    skill_allowlist=_PREPARATION_SKILLS,
    delegated_roles=(
        DelegatedRole(name="init", tools=_EVALUATION_TOOLS, skills=_EVALUATION_SKILLS),
        DelegatedRole(
            name="dataset",
            tools=_CHILD_WORKSPACE_TOOLS
            + (
                "get_storage_url",
                "df_submit_pipeline",
                "df_check_progress",
                "df_stop_task",
            ),
            skills=("data-processing",),
        ),
        DelegatedRole(
            name="train",
            tools=_CHILD_WORKSPACE_TOOLS
            + (
                "get_storage_url",
                "validate_workflow_dsl",
                "submit_workflow",
                "training_analysis",
                "analyze_task_failure",
                "df_stop_task",
            ),
            skills=("generate-workflow-dsl", "workflow-execution", "training-analysis"),
        ),
        DelegatedRole(name="eval", tools=_EVALUATION_TOOLS, skills=_EVALUATION_SKILLS),
    ),
    development_tool_allowlist=ECHOMIND_MANAGER_PROFILE.tool_allowlist,
    development_skill_allowlist=ECHOMIND_MANAGER_PROFILE.skill_allowlist,
    command_allowlist=(
        "user_message",
        "input_response",
        "cancel",
        "create_project",
        "permission_response",
        "workflow_execution",
    ),
    conversation_marker=ECHOMIND_MANAGER_PROFILE.tag_value,
    preparation_policy=_PREPARATION_POLICY,
)
