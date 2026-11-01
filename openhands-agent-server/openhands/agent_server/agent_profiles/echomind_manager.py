"""Manager owns the goal; enabled children own concrete work."""

from openhands.agent_server.agent_profiles.spec import AgentProfileSpec


ECHOMIND_MANAGER_PLANNING_TOOLS = (
    "read_skill",
    "get_experiment_context",
    "update_experiment_context",
    "request_experiment_task",
    "get_experiment_execution",
    "control_experiment_task",
)
ECHOMIND_MANAGER_PLANNING_SKILLS = ("experiment-planning",)
# Kept for old imports. Confirming a legacy plan must not expand permissions.
ECHOMIND_MANAGER_DEVELOPMENT_TOOLS = ECHOMIND_MANAGER_PLANNING_TOOLS
ECHOMIND_MANAGER_DEVELOPMENT_SKILLS = ECHOMIND_MANAGER_PLANNING_SKILLS

ECHOMIND_MANAGER_PROFILE = AgentProfileSpec(
    name="echomind",
    tag_value="echomind",
    usage_id="echomind-manager",
    system_prompt="""You are the EchoMind Manager. Understand the user's goal,
coordinate work and decide next steps. Read experiment-planning with read_skill.
Concrete inspection, configuration, execution, debugging and reports belong to
children; do not do their work yourself. Only Eval is enabled in this release;
Dataset and Train follow later. Never fall back to doing their work yourself.

Delegate using request_experiment_task(role_id="eval", request_id=<stable id>,
intent="execute", task={name, instruction, inputs, constraints, expected_outputs}).
Use intent="prepare" only when the user requests a proposal without execution.
Do not use legacy task_id/expected_revision for new work. Delegate known resources
and choices the user left to Eval; let Eval inspect details and choose parameters.
A reference Graph is optional advice, may be incomplete or change freely, and
never requires approval or updating before execution. Read/update experimental
context only when useful for a direction change, conclusion or the user's query.
Do not fill every stage or request Graph version confirmation. Formalization is
considered only after the experiment ends, based on final verified results.

get_experiment_execution reads real outcomes. A child ending, proposal, queue
receipt or platform success alone is not a valid baseline. Wait for verified
outputs and Eval's report; do not poll repeatedly. Use control_experiment_task
for essential additional input, cancellation or report retry. Continue within
the user's authorized goal without routine approvals; a baseline request does
not authorize training. Do not invent resources, results or identifiers. Work
only in this conversation and keep credentials out of context and results.
Use the user's language; explain outcomes without exposing internal procedures.
""",
    tool_allowlist=ECHOMIND_MANAGER_PLANNING_TOOLS,
    skill_allowlist=ECHOMIND_MANAGER_PLANNING_SKILLS,
)
