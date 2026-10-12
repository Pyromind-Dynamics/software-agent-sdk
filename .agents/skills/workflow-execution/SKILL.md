---
name: workflow-execution
description: Formally run an existing Pyromind Storage workflow and follow its task result when requested.
---

# Workflow execution

When the workflow is in the conversation workspace, upload `public_data/workflow_canvas/workflow.py` with `upload_file_to_pyromind` first. If upload fails, stop. Convert its returned absolute Storage path `/.../workflow.py` to `storage/.../workflow.py`. For an existing Storage workflow, use its `storage/.../workflow.py` path directly. Call `submit_workflow` with that path. It validates and stages a frozen definition; it does not start the platform task. Tell the user to review the Copilot confirmation dialog, then end the turn. Do not call `workflow_debug` or submit the same workflow through another tool.

After confirmation, the product runtime submits the task and records its `taskId`. On the terminal callback, use that taskId:

- Failed: call `analyze_task_failure(task_id)` and report the error and a concrete fix. Do not rerun automatically.
- Succeeded with a training node: hand the taskId to [training-analysis](../training-analysis/SKILL.md) and follow that Skill's analysis and reporting flow. Report the W&B run actually analyzed.
- Succeeded without training, stopped, or terminated: report the task status.

For workflows with several training stages, the current taskId-based analysis may select only the first available W&B run. State which run was analyzed; do not claim that all stages were covered.
