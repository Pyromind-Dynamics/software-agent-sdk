from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pyromind_runtime.application.product_instructions import DATA_FLYWHEEL_INSTRUCTIONS
from pyromind_runtime.domain.preparation import PreparationPolicy
from pyromind_runtime.domain.session_policy import SessionPolicy
from pyromind_runtime.ports.harness import DelegatedRole

from openhands.agent_server.agent_profiles.echomind_dataset import (
    ECHOMIND_DATASET_ROLE,
)
from openhands.agent_server.agent_profiles.echomind_eval import ECHOMIND_EVAL_ROLE
from openhands.agent_server.agent_profiles.echomind_inference import (
    ECHOMIND_INFERENCE_ROLE,
)
from openhands.agent_server.agent_profiles.echomind_manager import (
    ECHOMIND_MANAGER_DEVELOPMENT_SKILLS,
    ECHOMIND_MANAGER_DEVELOPMENT_TOOLS,
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
    agentgenome_enabled: bool = True


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

_PREPARATION_SKILLS = ("experiment-planning",)
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
_DEVELOPMENT_TOOLS = ECHOMIND_MANAGER_DEVELOPMENT_TOOLS + tuple(
    tool
    for tool in _PREPARATION_TOOLS
    if tool not in ECHOMIND_MANAGER_DEVELOPMENT_TOOLS
)
_DEVELOPMENT_SKILLS = ECHOMIND_MANAGER_DEVELOPMENT_SKILLS + tuple(
    skill
    for skill in _PREPARATION_SKILLS
    if skill not in ECHOMIND_MANAGER_DEVELOPMENT_SKILLS
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
        manifest = root / "experiment-planning/checklist.json"
        if manifest.is_file():
            return PreparationPolicy.model_validate_json(manifest.read_text())
    raise FileNotFoundError(
        "Experiment planning checklist is missing from the configured skills directory"
    )


_PREPARATION_POLICY = _load_preparation_policy()

_ECHOMIND_PRODUCT_INSTRUCTIONS = (
    (ECHOMIND_MANAGER_PROFILE.system_prompt or "")
    + """
For the product preparation flow, read experiment-planning.
Before Project creation, collect the scenario checklist using user messages,
data inspection, workspace scripts and knowledge retrieval. Use get_preparation
for gaps and valid evidence references. Ask for essential facts that cannot be
established.
Persist the complete facts using update_preparation and its current revision.
Only after the checklist is complete, use set_preparation_plan with the plan,
and optional entry_task_id/role_id when a delegated entry task is explicitly
selected. The current four-node workflow starts nodes from the Project workspace.
Select stage.kind and task.kind only from preparation.policy.workflow_nodes.
Present a compact business-node plan; leave implementation details to execution.
The user may revise it in chat and creates the Project with the UI button.
Creating a Project materializes the confirmed plan as a Working Graph. An
explicit preparation entry task wins; otherwise Runtime uses the Graph's unique
root node. It then sends a hidden continuation so the Manager opens that
StageRun automatically. A user can also ask in chat to start any Graph node;
call open_project_stage with its project_id and node_id instead of asking for a
/start request.
Use the Sandbox to inspect data and prepare files. Start business tasks only after
the user creates the Project. Never execute Dataset processing, write manifests,
run validators, or call df_submit_pipeline from the Manager turn. If StageRun
creation, approval, or delegation fails, stop and report the error.
Dataset, Train, Inference and Evaluator are first-reference business nodes, not
fixed stages. Use the reference to understand the scenario, but let the user's
confirmed data, outputs and graph determine the final workflow. A node's
execution_role is authoritative; evaluator nodes use role_id="eval".
After creation, get_project_graph and the stage run tools are authoritative for
development. Do not use confirm_experiment_plan for new EchoMind sessions.
Delegate complete node tasks with the node's execution_role and a stable
request_id. Dataset uses data-processing. Train uses generate-workflow-dsl,
workflow-execution and training-analysis, and may run the existing platform
training flow after user confirmation. Inference uses inference-evaluation to
produce predictions. Evaluator/eval uses inference-evaluation to score existing
predictions or run an explicitly requested baseline. Pass confirmed scope,
inputs, outputs and acceptance criteria.

For a user request to redo a Node, do not steer or resume the existing TaskRun.
Call request_project_stage_task with a new request_id. Runtime cancels unfinished
TaskRuns in that StageRun before creating the replacement. A reused request_id
returns the existing task and does not dispatch new work.
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
Do not repeat an active submitted task. After successful verified data preparation,
delegate Train only within the accepted scope. Bind verified outputs using
task.artifact_refs entries with input_name, task_run_id and artifact_name;
Runtime resolves and freezes their actual URIs. Bind Dataset's training and
optional validation files separately to Train inputs, preserving the delivered
files rather than producing a second frozen dataset. Train uses the existing workflow
confirmation. After training, bind the verified model artifact to Eval through
task.artifact_refs, honoring the agreed evaluation scope.
A Dataset sample is never Dataset completion, regardless of row count. Dataset
success requires the submit TaskRun and its verified platform receipt.
Missing inputs or failures need user input; do not infer authorization for training
from a model evaluation. Never enable an unimplemented node.
Do not use legacy task_id/expected_revision. A model evaluation authorizes no
training or deployment. Wait for Runtime notifications, verified metrics and
reports; do not busy-poll. Use read_experiment_artifact for registered reports,
and control_experiment_task for explicit cancellation or report retry without
rerunning GPU.
Use the user's language and keep credentials out of persisted facts and results.
"""
)


ECHOMIND_PRODUCT_PROFILE = ConversationProfile(
    configuration_version=14,
    route_prefix="/api/v2/echomind/conversations",
    profile_id=ECHOMIND_MANAGER_PROFILE.name,
    harness_id="pi",
    instructions=_ECHOMIND_PRODUCT_INSTRUCTIONS,
    tool_allowlist=_PREPARATION_TOOLS,
    skill_allowlist=_PREPARATION_SKILLS,
    delegated_roles=(
        ECHOMIND_EVAL_ROLE,
        ECHOMIND_DATASET_ROLE,
        ECHOMIND_TRAIN_ROLE,
        ECHOMIND_INFERENCE_ROLE,
    ),
    managed_delegation=True,
    development_tool_allowlist=_DEVELOPMENT_TOOLS,
    development_skill_allowlist=_DEVELOPMENT_SKILLS,
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
    agentgenome_enabled=False,
)


def session_policy_for_profile(profile: ConversationProfile) -> SessionPolicy:
    return SessionPolicy(
        configuration_version=profile.configuration_version,
        tool_allowlist=profile.tool_allowlist,
        skill_allowlist=profile.skill_allowlist,
        development_tool_allowlist=profile.development_tool_allowlist,
        development_skill_allowlist=profile.development_skill_allowlist,
        product_instructions=profile.instructions,
        delegated_roles=tuple(
            role.model_dump(mode="json") for role in profile.delegated_roles or ()
        ),
        managed_delegation=profile.managed_delegation,
        command_allowlist=profile.command_allowlist,
        preparation_policy=profile.preparation_policy,
        agentgenome_enabled=profile.agentgenome_enabled,
    )
