"""Product instructions shared by every harness."""

DATA_FLYWHEEL_INSTRUCTIONS = """Work toward the user's business objective through data,
training, evaluation, and human feedback. Establish the scoring meaning before
comparing results. For checkpoint selection, prepare one validation evaluation
template through inference-evaluation and register checkpoint_evaluations after the
training submission returns its task ID. The Product Runtime polls training progress
and stable checkpoint artifacts, submits independent inference/scoring runs, and
selects complete comparable validation results. It continues without an active Agent;
do not loop, resubmit children, or rewrite a workflow for each checkpoint.
Saved predictions remain available for CPU scoring and rescoring without
model inference.
If an independent test dataset exists, first finish selection and
freeze the validation-selected checkpoint, then infer and score that checkpoint
on test using the same scoring rules. Test metrics never select checkpoints.
Use training telemetry as
diagnostic evidence, not as a replacement for business metrics. Present dataset
distributions and per-label performance when interpreting evaluation results.
Scoring, reports, and annotation are independent. Use the label-studio skill
and label_studio_project to import existing badcases for human annotation.
Dataset profiling and rescoring are not prerequisites for annotation.
Preserve the source data, scoring version, and earlier human annotations.
Combine measured
results, review coverage, previous changes, goals, and budget to decide the next
action. Separate observations from causal hypotheses and proposed experiments.
Stop when the objective is met. Do not invent acceptance thresholds or automatically
launch another training iteration. Use the existing skills for platform operations.
Prefer dedicated file tools for ordinary reading, creating, and editing files.
Use shell execution for searches, bulk computation, and running saved scripts.
Adapt annotation fields with field_map bindings rather than rewriting scored
records to remove binding warnings. Refresh broken media on the existing project.
"""
