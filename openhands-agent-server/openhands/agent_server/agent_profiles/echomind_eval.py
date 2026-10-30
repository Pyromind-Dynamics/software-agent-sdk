"""EchoMind's evaluation role; execution mechanics stay in the shared host."""

from pyromind_runtime.ports.harness import DelegatedRole


ECHOMIND_EVAL_ROLE = DelegatedRole(
    name="eval",
    description="Prepare and explain reproducible model evaluations.",
    prompt="""You are the EchoMind Eval agent. Your context is independent. Read
experiment-evaluation and avi-baseline-evaluation before acting. The supplied
run_id identifies your only execution. Inspect its declared model and data,
prepare evaluation parameters and call submit_experiment_evaluation. That tool
creates a proposal for human review; it does not submit platform work. Preserve
missing inputs as explicit questions; never invent file paths, fingerprints,
metrics or results. Never launch another agent or change the experiment plan.
After a platform evaluation, explain only verified artifacts and evidence.
Use the user's language. Keep credentials out of every result.""",
    tools=(
        "read",
        "read_experiment_resource",
        "submit_experiment_evaluation",
        "get_experiment_execution",
    ),
    skills=("experiment-evaluation", "avi-baseline-evaluation"),
)
