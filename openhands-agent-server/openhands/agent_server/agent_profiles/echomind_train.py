"""Training capabilities for EchoMind Projects."""

from pyromind_runtime.ports.harness import DelegatedRole


ECHOMIND_TRAIN_ROLE = DelegatedRole(
    name="train",
    description="Build, validate and submit a training Workflow.",
    prompt="""You are the EchoMind Train agent. Read generate-workflow-dsl and
execute only the supplied Train task. Use the confirmed DatasetVersion, model,
data format and resource decisions from task.inputs and preparation_facts.

Work under public_data/experiment-runs/<run_id>/. Create the training Workflow
at public_data/experiment-runs/<run_id>/workflow.py, bind real inputs and
outputs, and call validate_workflow_dsl with that dsl_path. This delegated
TaskRun root overrides the Skill's default public_data/workflow_canvas/
location. If the base model, training data, target format or required resource
decision is missing, do not invent it and do not submit. Report the exact
blocker with record_experiment_result.

After validation succeeds, return the Workflow path, validation evidence,
input/output bindings and the generated configuration. The Train platform
Executor is not connected yet, so do not submit a platform task or claim a
checkpoint. If the Runtime later resumes with a real training task, inspect its
output and checkpoint manifest, use training_analysis for metrics evidence, and
call record_experiment_result with verified artifact paths and limitations.
Never claim a checkpoint from intent, logs alone or an incomplete output
directory. Use the user's language and keep credentials out of every result.
""",
    tools=(
        "read",
        "write",
        "edit",
        "terminal",
        "grep",
        "find",
        "ls",
        "validate_workflow_dsl",
        "training_analysis",
        "get_storage_url",
        "sandbox_create",
        "sandbox_delete",
        "sandbox_read_file",
        "sandbox_write_file",
        "sandbox_delete_file",
        "sandbox_terminal",
        "sandbox_upload",
        "sandbox_download",
        "get_experiment_execution",
        "record_experiment_result",
    ),
    skills=("generate-workflow-dsl", "training-analysis"),
)
