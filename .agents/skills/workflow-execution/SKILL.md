---
name: workflow-execution
description: Formally run an existing Pyromind Storage workflow and follow its task result when requested.
---

# Workflow execution

When the workflow is in the conversation workspace, upload the actual generated `public_data/workflow_canvas/workflow.py` with `upload_file_to_pyromind` first. If upload fails, stop. Use the returned file at its `/workspace/.../workflow.py` Storage path, preserving the returned directory and filename. For an existing Storage workflow, use its actual `/workspace/.../workflow.py` path directly; do not invent a filename. Call `submit_workflow` with that path. It validates and stages a frozen definition; it does not start the platform task. Tell the user to review the Copilot confirmation dialog, then end the turn. Do not call `workflow_debug` or submit the same workflow through another tool.

Set `continue_after_completion=true` on `submit_workflow` only when the user explicitly requests result analysis or further actions after completion. Otherwise leave it false: the host reports completion without starting another agent turn.

After confirmation, the product runtime submits the task and records its `taskId`. The host checks status in the background. End the current turn; never use terminal sleep, loops, or repeated status tools to wait for completion, even when the user asks for the final result. A user asking for the current status can be answered with a single status query.

When the host resumes the conversation for an explicitly requested follow-up, use that taskId:

- Failed: call `inspect_workflow_task(task_id)` and report the error and a concrete fix. Do not rerun automatically.
- Succeeded with a training node: use [training-analysis](../training-analysis/SKILL.md) for requested run diagnostics and [inference-evaluation](../inference-evaluation/SKILL.md) for explicitly requested model evaluation. Do not start evaluation solely because training completed. Use actual model artifacts and report missing validation data or artifacts instead of substituting training loss for quality evidence.
- Succeeded without training, stopped, or terminated: report the task status.

For workflows with several training stages, the current taskId-based analysis may select only the first available W&B run. State which run was analyzed; do not claim that all stages were covered.
