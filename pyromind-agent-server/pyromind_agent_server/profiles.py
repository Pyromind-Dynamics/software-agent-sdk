from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pyromind_runtime.application.product_instructions import DATA_FLYWHEEL_INSTRUCTIONS
from pyromind_runtime.domain.preparation import PreparationPolicy
from pyromind_runtime.ports.harness import DelegatedRole

from openhands.agent_server.agent_profiles.echomind_dataset import ECHOMIND_DATASET_ROLE
from openhands.agent_server.agent_profiles.echomind_eval import ECHOMIND_EVAL_ROLE
from openhands.agent_server.agent_profiles.echomind_init import ECHOMIND_INIT_ROLE
from openhands.agent_server.agent_profiles.echomind_manager import (
    ECHOMIND_MANAGER_PROFILE,
)
from openhands.agent_server.agent_profiles.echomind_train import ECHOMIND_TRAIN_ROLE


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
    managed_delegation: bool = False
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

_PREPARATION_SKILLS = ("model-evaluation-preparation", "inference-evaluation")
_PREPARATION_TOOLS = (
    "read",
    "write",
    "edit",
    "terminal",
    "grep",
    "find",
    "ls",
    "ask_user",
    "get_preparation",
    "update_preparation",
    "set_preparation_plan",
    "inspect_resource",
)
_EXECUTION_TOOLS = _PREPARATION_TOOLS + (
    "get_experiment_context",
    "update_experiment_context",
    "request_experiment_task",
    "get_experiment_execution",
    "control_experiment_task",
    "read_experiment_artifact",
    "get_storage_url",
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
        manifest = root / "model-evaluation-preparation/checklist.json"
        if manifest.is_file():
            return PreparationPolicy.model_validate_json(manifest.read_text())
    raise FileNotFoundError(
        "Model evaluation preparation checklist is missing "
        "from the configured skills directory"
    )


_PREPARATION_POLICY = _load_preparation_policy()


ECHOMIND_PRODUCT_PROFILE = ConversationProfile(
    configuration_version=13,
    route_prefix="/api/v2/echomind/conversations",
    profile_id=ECHOMIND_MANAGER_PROFILE.name,
    harness_id="pi",
    instructions="""You are the EchoMind Manager. Read model-evaluation-preparation.
Before Project creation, collect the scenario checklist using user messages,
data inspection, workspace scripts and knowledge retrieval. Use get_preparation
for gaps and valid evidence references. Ask for essential facts that cannot be
established.
Persist the complete facts using update_preparation and its current revision.
Only after the checklist is complete, use set_preparation_plan with the plan,
entry_task_id and role_id. Runtime reuses the saved facts and resolves inputs.
Select stage.kind and task.kind only from preparation.policy.workflow_nodes.
Present a compact business-node plan; leave implementation details to execution.
The user may revise it in chat and creates/starts the Project with the UI button.
Use the Sandbox to inspect data and prepare files. Start business tasks only after
the user creates the Project.
After creation, get_experiment_context is the current adaptable workflow plan.
Use request_experiment_task with a new stable request_id for independent work
requiring fresh context. Use control_experiment_task(action=continue,run_id,message)
to guide or adjust an existing child's work in its original context, including
while it is running, completed, blocked or awaiting a platform response.
Choose based on context needs, not TaskRun state. A reused request_id returns the
existing task; it does not send new guidance. A request to retry or investigate
is guidance for that child, not permission to substitute a different pipeline.
Delegate complete tasks with a stable request_id and valid stage_id, choosing
init for baseline work, dataset for cleaning, train for training and eval for
candidate evaluation. Follow the declared node roles. Init is one business stage.
Freeze exact source file paths in task.inputs for tasks that review a sample.
If the approved source is a directory, mark it with input_kind=directory or
preserve its trailing slash in task.inputs. A bare path is treated as a file.
Delegated roles can ask the user directly with ask_user. Their answers return
to the asking role; do not repeat those questions. Resolve technical blockers
within the accepted scope. Ask only when a decision or missing fact needs the user.
Use get_experiment_execution and read_experiment_artifact for verified evidence.
When asking for a decision, use ask_user.presentation for a concise title and
Chat content blocks. Use absolute Storage image paths without signed URLs or
credentials. Questions do not grant platform permissions.
For historical tasks parked at a frozen review checkpoint, use the existing
sample_review choices (confirm_full or revise), then continue the same TaskRun.
Do not repeat a submitted task. After successful verified data preparation,
delegate Train only within the accepted scope. Bind verified outputs using
task.artifact_refs entries with input_name, task_run_id and artifact_name;
Runtime resolves and freezes their actual URIs. Bind Dataset's training and
optional validation files separately to Train inputs, preserving the delivered
files rather than producing a second frozen dataset. Train uses the existing workflow
confirmation. After training, bind the verified model artifact to Eval through
task.artifact_refs, honoring the agreed evaluation scope.
Missing inputs or failures need user input; do not infer authorization for training
from a model evaluation. Never enable an unimplemented node.
Do not use legacy task_id/expected_revision. A model evaluation authorizes no
training or deployment. Wait for Runtime notifications, verified metrics and
reports; do not busy-poll. Use read_experiment_artifact for registered reports,
and control_experiment_task to continue/cancel/retry reports without rerunning GPU.
Use the user's language and keep credentials out of persisted facts and results.
""",
    tool_allowlist=_PREPARATION_TOOLS,
    skill_allowlist=_PREPARATION_SKILLS,
    delegated_roles=(
        ECHOMIND_INIT_ROLE,
        ECHOMIND_DATASET_ROLE,
        ECHOMIND_TRAIN_ROLE,
        ECHOMIND_EVAL_ROLE,
    ),
    managed_delegation=True,
    development_tool_allowlist=_EXECUTION_TOOLS,
    development_skill_allowlist=_PREPARATION_SKILLS,
    command_allowlist=(
        "user_message",
        "input_response",
        "cancel",
        "create_project",
        "permission_response",
        "workflow_execution",
        "control_experiment_task",
        "resolve_experiment_submission",
    ),
    conversation_marker=ECHOMIND_MANAGER_PROFILE.tag_value,
    preparation_policy=_PREPARATION_POLICY,
)
