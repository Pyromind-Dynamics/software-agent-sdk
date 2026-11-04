"""Frozen v1 fixture: keep legacy API/lease tests separate from the new contract."""

from dataclasses import replace

from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.ports.harness import DelegatedRole

from openhands.agent_server.agent_profiles.echomind_manager import (
    ECHOMIND_MANAGER_DEVELOPMENT_SKILLS,
    ECHOMIND_MANAGER_DEVELOPMENT_TOOLS,
    ECHOMIND_MANAGER_PROFILE,
)


LEGACY_EXPERIMENT_PROFILE = replace(
    ECHOMIND_PRODUCT_PROFILE,
    configuration_version=0,
    preparation_policy=None,
    command_allowlist=(
        "user_message",
        "cancel",
        "confirm_experiment_plan",
        "reject_experiment_plan",
        "control_experiment_task",
        "resolve_experiment_submission",
    ),
    skill_allowlist=("experiment-planning",),
    tool_allowlist=(
        "read",
        "terminal",
        "create_experiment_plan",
        "revise_experiment_plan",
        "get_experiment_plan",
        "request_experiment_task",
        "get_experiment_execution",
    ),
    development_tool_allowlist=(
        "read",
        "write",
        "edit",
        "terminal",
        "subagent",
        "update_plan",
        "create_experiment_plan",
        "revise_experiment_plan",
        "get_experiment_plan",
        "request_experiment_task",
        "get_experiment_execution",
        "validate_workflow_dsl",
        "df_submit_pipeline",
        "df_submit_evaluation",
        "sandbox_create",
        "sandbox_terminal",
    ),
    development_skill_allowlist=(
        "experiment-planning",
        "inference-evaluation",
        "data-processing",
        "sandbox",
    ),
    delegated_roles=(
        DelegatedRole(
            name="eval",
            description="Legacy proposal preparation",
            prompt=(
                "Prepare a proposal with read_experiment_resource and "
                "submit_experiment_evaluation. Do not start platform tasks."
            ),
            tools=(
                "read",
                "read_experiment_resource",
                "submit_experiment_evaluation",
                "get_experiment_execution",
            ),
            skills=("experiment-evaluation", "avi-baseline-evaluation"),
        ),
    ),
)


EXPERIMENT_PROFILE_V4 = replace(
    ECHOMIND_PRODUCT_PROFILE,
    configuration_version=4,
    preparation_policy=None,
    instructions=ECHOMIND_MANAGER_PROFILE.system_prompt or "",
    tool_allowlist=ECHOMIND_MANAGER_DEVELOPMENT_TOOLS,
    skill_allowlist=ECHOMIND_MANAGER_DEVELOPMENT_SKILLS,
    development_tool_allowlist=ECHOMIND_MANAGER_DEVELOPMENT_TOOLS,
    development_skill_allowlist=ECHOMIND_MANAGER_DEVELOPMENT_SKILLS,
    command_allowlist=LEGACY_EXPERIMENT_PROFILE.command_allowlist,
)
