"""Frozen v1 fixture: keep legacy API/lease tests separate from the new contract."""

from dataclasses import replace

from pyromind_agent_server.profiles import ECHOMIND_PRODUCT_PROFILE
from pyromind_runtime.ports.harness import DelegatedRole


LEGACY_EXPERIMENT_PROFILE = replace(
    ECHOMIND_PRODUCT_PROFILE,
    configuration_version=0,
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
