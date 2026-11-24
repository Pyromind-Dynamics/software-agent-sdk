---
name: workflow-execution
description: Formally run an existing Pyromind Storage workflow and follow its task result when requested.
---

# Workflow execution

Call `submit_workflow` with the actual workflow path. For the generated file, use the conversation path `public_data/workflow_canvas/workflow.py`; for an existing Storage workflow, use its actual `/workspace/.../workflow.py` path. Do not upload or copy a conversation file first, and do not invent a filename. The tool resolves the execution location, validates a frozen definition and submits it directly, returning the platform `taskId` without a confirmation dialog. While the task is pending, end the turn. Do not call `workflow_debug` or submit the same workflow through another tool. Reuse an existing receipt; if the submission outcome is uncertain, check the platform before retrying.

Set `continue_after_completion=true` on `submit_workflow` only when the user explicitly requests result analysis or further actions after completion. Otherwise leave it false: the host reports completion without starting another agent turn.

For flat Vision training with the default collator, verify the image references in the final bound train/validation manifests using the training mount. DSL validation checks the workflow definition. A successful check against a separate source root does not establish that the serialized paths can be opened.

On the terminal callback, use the submitted taskId:

- Failed: call `inspect_workflow_task(task_id)` to find failed nodes, then inspect each failed node with `node_id` and `include_log=true`. Compare its logs with the submitted DSL and bound data. For a missing training image, verify the corrected manifests before another submission. Fix any independently evidenced DSL problems in the existing task, such as cloning into a path that already contains a verified compatible model. Treat launcher argument errors separately from DSL node inputs. Validate the revised file and submit that same workspace path with `submit_workflow`. A valid DSL only establishes that the definition passes validation; use the new task's node results to verify execution.
- Succeeded with a training node: continue the follow-up requested by the user or included in the accepted plan. Use [training-analysis](../training-analysis/SKILL.md) for run diagnostics and [inference-evaluation](../inference-evaluation/SKILL.md) for evaluation. For a model evaluation, use the actual output of this training task and the specified dataset. For checkpoint selection, score the matching saved validation predictions. Return the task status when training is the complete request.
- Succeeded without training, stopped, or terminated: report the task status.

For workflows with several training stages, the current taskId-based analysis may select only the first available W&B run. State which run was analyzed; do not claim that all stages were covered.
