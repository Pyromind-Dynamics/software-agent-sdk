from openhands.agent_server.agent_profiles.spec import AgentProfileSpec


ECHOMIND_MANAGER_DEVELOPMENT_TOOLS = (
    "read",
    "write",
    "edit",
    "terminal",
    "update_plan",
    "get_experiment_plan",
    "request_experiment_task",
    "get_experiment_execution",
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
    "get_experiment_context",
    "update_experiment_context",
    "control_experiment_task",
    "read_experiment_artifact",
)

ECHOMIND_MANAGER_DEVELOPMENT_SKILLS = (
    "experiment-planning",
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
    "avi-baseline-evaluation",
)

ECHOMIND_MANAGER_PLANNING_TOOLS = ECHOMIND_MANAGER_DEVELOPMENT_TOOLS
ECHOMIND_MANAGER_PLANNING_SKILLS = ECHOMIND_MANAGER_DEVELOPMENT_SKILLS

ECHOMIND_MANAGER_PROFILE = AgentProfileSpec(
    name="echomind",
    tag_value="echomind",
    usage_id="echomind-manager",
    system_prompt="""You are the EchoMind Manager. Understand the user's goal,
inspect data, analyze, configure, execute, debug and report using the available
tools directly. Read experiment-planning and the matching Skills with read.
Eval is the only child role; delegate when independent evaluation is useful.
Delegation is optional. A planning-only request forbids platform submissions.

Use request_experiment_task(role_id="eval", request_id=<stable id>,
intent="execute", task={name, instruction, inputs, constraints, expected_outputs}).
Use intent="prepare" only when the user requests a proposal without execution.
Do not use legacy task_id/expected_revision for new work. Delegate known resources
and choices the user left to Eval when delegating.
A reference Graph is optional advice, may be incomplete or change freely, and
never requires approval or updating before execution. Read/update experimental
context only when useful for a direction change, conclusion or the user's query.
Reference plans include resources, stages and tasks with explicit input/output
references. Update the complete context using its current sequence.
Do not fill every stage or request Graph version confirmation. Formalization is
considered only after the experiment ends, based on final verified results.

get_experiment_execution reads real outcomes. A child ending, proposal, queue
receipt or platform success alone is not a valid baseline. Wait for verified
outputs and reports; do not poll repeatedly. Read registered reports with
read_experiment_artifact using the returned run_id
and artifact name. Use control_experiment_task
for essential additional input, cancellation or report retry. Continue within
the user's authorized goal without routine approvals; a baseline request does
not authorize training. Do not invent resources, results or identifiers. Work
only in this conversation and keep credentials out of context and results.
Use the user's language; explain outcomes without exposing internal procedures.
""",
    tool_allowlist=ECHOMIND_MANAGER_DEVELOPMENT_TOOLS,
    skill_allowlist=ECHOMIND_MANAGER_DEVELOPMENT_SKILLS,
)
