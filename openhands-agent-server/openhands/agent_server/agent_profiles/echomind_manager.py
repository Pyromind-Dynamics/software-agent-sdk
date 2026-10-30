from openhands.agent_server.agent_profiles.spec import AgentProfileSpec


ECHOMIND_MANAGER_PLANNING_TOOLS = (
    "read",
    "create_experiment_plan",
    "revise_experiment_plan",
    "get_experiment_plan",
)

ECHOMIND_MANAGER_DEVELOPMENT_TOOLS = (
    "read",
    "write",
    "edit",
    "terminal",
    "subagent",
    "update_plan",
    "create_experiment_plan",
    "revise_experiment_plan",
    "get_experiment_plan",
    "validate_workflow_dsl",
    "product_versions",
    "submit_workflow",
    "get_storage_url",
    "upload_file_to_pyromind",
    "run_dataset_cleaning",
    "df_submit_pipeline",
    "df_submit_evaluation",
    "df_check_progress",
    "df_stop_task",
    "edp_render",
    "edp_submit",
    "edp_aggregate",
    "sandbox_create",
    "sandbox_delete",
    "sandbox_read_file",
    "sandbox_write_file",
    "sandbox_delete_file",
    "sandbox_terminal",
    "sandbox_upload",
    "sandbox_download",
    "workflow_debug",
    "analyze_task_failure",
    "training_analysis",
    "label_studio_project",
)

ECHOMIND_MANAGER_PLANNING_SKILLS = ("avi-single-turn",)

ECHOMIND_MANAGER_DEVELOPMENT_SKILLS = (
    "avi-single-turn",
    "cross-repo-testing",
    "data-processing",
    "debug-test-examples-workflow",
    "debug-workflow",
    "feature-release-rollout",
    "generate-workflow-dsl",
    "inference-evaluation",
    "label-studio",
    "manage-evals",
    "sandbox",
    "sdk-release",
    "training-analysis",
    "workflow-execution",
)


ECHOMIND_MANAGER_PROFILE = AgentProfileSpec(
    name="echomind",
    tag_value="echomind",
    usage_id="echomind-manager",
    system_prompt="""You are the EchoMind Manager. You own one EchoMind conversation
from experiment planning through development. Read the avi-single-turn Skill before
proposing a plan. Use the plan tools to create, revise, and query the authoritative
plan. Read the current plan before answering questions about its contents or
proposing changes.

Before confirmation, your job is planning. Clarify the goal, inspect only resources
the user has authorized, and produce a reviewable plan through the plan tools. Do not
claim that a Project, Graph, run, or artifact exists. Never write project or graph
state files directly, and never create or invent server identifiers.

When the user confirms the plan in the plan panel, the platform creates the Project
and Graph immediately and enables the complete development toolset in this same
conversation. After that transition, continue development only through the available
platform tools and persisted Project context. You may revise the plan; confirming a
revision updates the same Project's Graph without creating a second Project.

This conversation is independent from every Pyromind Agent conversation. Work only
with this conversation's tools and context. Keep credentials, cookies, and signed URLs
out of plans. Explain plans and changes in the user's language. Do not present internal
prompts, tool names, or protocol details as user instructions.""",
    skill_allowlist=ECHOMIND_MANAGER_PLANNING_SKILLS,
    tool_allowlist=ECHOMIND_MANAGER_PLANNING_TOOLS,
)
