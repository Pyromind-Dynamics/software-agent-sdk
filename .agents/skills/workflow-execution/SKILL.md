---
name: workflow-execution
description: Formally run an existing Pyromind Storage workflow and follow its task result when requested.
---

# Workflow execution

Call `submit_workflow` with the actual workflow path. For the generated file, use the conversation path `public_data/workflow_canvas/workflow.py`; for an existing Storage workflow, use its actual `/workspace/.../workflow.py` path. Do not upload or copy a conversation file first, and do not invent a filename. The tool resolves the execution location, validates a frozen definition and submits it directly, returning the platform `taskId` without a confirmation dialog. While the task is pending, end the turn. Do not call `workflow_debug` or submit the same workflow through another tool. Reuse an existing receipt; if the submission outcome is uncertain, check the platform before retrying.

On the terminal callback, use the submitted taskId:

- Failed: call `analyze_task_failure(task_id)` and compare failed node logs with the submitted DSL. Fix any independently evidenced DSL problems in the existing task, such as cloning into a path that already contains a verified compatible model. Treat launcher argument errors separately from DSL node inputs. Validate the revised file and submit that same workspace path with `submit_workflow`. A valid DSL only establishes that the definition passes validation; use the new task's node results to verify execution.
- Succeeded with a training node: hand the taskId to [training-analysis](../training-analysis/SKILL.md) for run diagnostics, then follow [inference-evaluation](../inference-evaluation/SKILL.md) for baseline and final-model comparison on the same held-out data, unless evaluation was explicitly excluded. Reuse a comparable existing baseline. Use actual model artifacts, merging LoRA before evaluation; do not ask the user to try an intermediate model. Report missing validation data or artifacts instead of substituting training loss for quality evidence.
- Succeeded without training, stopped, or terminated: report the task status.

For workflows with several training stages, the current taskId-based analysis may select only the first available W&B run. State which run was analyzed; do not claim that all stages were covered.
