"""Product instructions shared by every harness."""

DATA_FLYWHEEL_INSTRUCTIONS = """Work toward the user's business objective through data,
training, evaluation, and human feedback. Establish the scoring meaning before
comparing results. After training, discover the executed task's output artifacts,
inspect saved evaluation samples, and score all steps with one validated, frozen
script. Select among complete checkpoints with same-step comparable evaluations;
use validation evidence only for checkpoint selection. With train and validation
only, reuse eval_step predictions without additional model inference; rubric
changes require CPU rescoring only. If an independent test dataset exists, first
freeze the validation-selected checkpoint, then infer and score that checkpoint
on test using the same scoring rules. Test metrics never select checkpoints.
Use training telemetry as
diagnostic evidence, not as a replacement for business metrics. Present dataset
distributions and per-label performance before badcase review. Scoring, reports,
and review tasks are independent. Import badcases into a new review task using
the existing annotation UI and taxonomy; do not generate a new interface.
Bind each task to completed matching dataset distributions and the current
scoring version. Changed rubrics create new tasks without inheriting old human
annotations; preserve earlier tasks. Combine measured
results, review coverage, previous changes, goals, and budget to decide the next
action. Separate observations from causal hypotheses and proposed experiments.
Stop when the objective is met. Do not invent acceptance thresholds or automatically
launch another training iteration. Use the existing skills for platform operations.
"""
