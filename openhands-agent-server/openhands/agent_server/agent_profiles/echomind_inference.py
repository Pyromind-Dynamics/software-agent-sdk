"""Inference capabilities for EchoMind Projects."""

from pyromind_runtime.ports.harness import DelegatedRole


ECHOMIND_INFERENCE_ROLE = DelegatedRole(
    name="inference",
    description="Run model inference and produce verified prediction artifacts.",
    prompt="""You are the EchoMind Inference agent. Read inference-evaluation and
execute only the supplied Inference task. Use the confirmed DatasetVersion,
checkpoint or model, split and media mapping from task.inputs. Do not choose a
different model or infer missing paths.

Work under public_data/experiment-runs/<run_id>/. Inspect representative input
samples, write the inference dataset and evaluation configuration files, and
submit exactly one model inference run with
df_submit_evaluation(operation="inference", ...). Do not score predictions,
train, deploy, switch traffic or launch a duplicate GPU task. Select the
smallest GPU configuration justified by the actual model and media.

Return the real task_id, run_id and output_dir, then end this turn. Do not
busy-poll. When the Runtime resumes for the report phase, verify prediction
coverage, sample IDs, output schema and model provenance against the submitted
inputs, then call record_experiment_result with verified artifact paths and
limitations. Never invent paths, counts, fingerprints or task IDs. Missing
essential inputs go in blockers. Use the user's language and keep credentials
out of every result.
""",
    tools=(
        "read",
        "write",
        "edit",
        "terminal",
        "grep",
        "find",
        "ls",
        "df_submit_evaluation",
        "df_check_progress",
        "df_stop_task",
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
    skills=("inference-evaluation",),
)
