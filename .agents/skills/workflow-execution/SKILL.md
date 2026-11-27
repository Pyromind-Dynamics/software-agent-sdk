---
name: workflow-execution
description: Formally run an existing Pyromind Storage workflow and follow its task result when requested.
---

# Workflow execution

Read [SDK execution](references/sdk-execution.md) when the system enables SDK
access. Use the retained `validate_workflow_dsl` and read the actual graph directly
from `validated_workflow_path` in its visible content, then call the native SDK once
from the current terminal. Stop when the path is absent or `artifact_unavailable`
is returned, even if `valid=true`; do not reconstruct a graph from internal details.
Lightweight preparation helpers are installed in the SDK interpreter. An explicit
HTTP validator is only for callers that already hold its separate authentication.
When SDK prerequisites are missing and `submit_workflow` is listed, use that entry.

Use `public_data/workflow_canvas/workflow.py` for the generated canvas, or the
actual `/workspace/.../workflow.py` Storage path. Do not copy or upload the
conversation file first. Use the current session or Project task/attempt out_id,
disable create retries, save the returned task ID, and end the turn while pending.
An unconfirmed create result requires callback/platform reconciliation before
another submission; restarting the session does not authorize another create.

Set `continue_after_completion=true` on `submit_workflow` only when the user explicitly requests result analysis or further actions after completion. Otherwise leave it false: the host reports completion without starting another agent turn.

For flat Vision training with the default collator, verify the image references in the final bound train/validation manifests using the training mount. DSL validation checks the workflow definition. A successful check against a separate source root does not establish that the serialized paths can be opened.

On the terminal callback, use the submitted taskId:

- Failed: with SDK access, read [SDK inspection](references/sdk-inspection.md)
  and use the ordinary inspection function to find failed nodes and read their
  logs. If node states are unavailable, report that limit and do not infer them
  from timestamps or task state. Without SDK access, use `inspect_workflow_task`
  only when listed. Compare the logs with the submitted DSL and bound data. For
  a missing training image, verify the corrected manifests before another
  submission. Fix independently evidenced DSL problems in the existing task,
  such as cloning into a path that already contains a verified compatible model.
  Treat launcher argument errors separately from DSL node inputs. Validate the
  revised file and submit that same workspace path through the entry selected
  above. A valid DSL only establishes that the definition passes validation;
  use the new task's node results to verify execution.
- Succeeded with a training node: continue the follow-up requested by the user or included in the accepted plan. Use [training-analysis](../training-analysis/SKILL.md) for run diagnostics and [inference-evaluation](../inference-evaluation/SKILL.md) for evaluation. For a model evaluation, use the actual output of this training task and the specified dataset. For checkpoint selection, score the matching saved validation predictions. Return the task status when training is the complete request.

- After a Project training task succeeds, record its task ID and verified model or checkpoint path in the accepted plan. Discover artifacts with `training_analysis(operation="artifacts")` when needed, then use that exact model for the authorized Eval task. Include W&B metrics when requested and available.
- Succeeded without training, stopped, or terminated: report the task status.

For workflows with several training stages, the current taskId-based analysis may select only the first available W&B run. State which run was analyzed; do not claim that all stages were covered.

## Project report delivery

Use [export_plan_report.py](scripts/export_plan_report.py) to write the versioned
primary report. Custom analysis scripts write intermediate JSON or JSONL results
for the exporter. Create the mapping from the actual
fields of the evaluated artifact. For example, a mapping can contain
`{"metrics":[{"label":"result_accuracy","unit":"ratio","path":["metrics","accuracy"]}]}`.
A path is a list of object keys and zero-based array indexes. For JSONL, the
first index selects a row. Optional `summary`, `findings`, `badcases`, `files`,
`checks` and `source` mappings point to matching source fields. For a platform
report, pass one `--platform-task-id` for each verified task in the approved
attempt. For Train with separate SFT and LoRA merge tasks, pass both IDs. The
exporter writes `{"source":{"platform_task_ids":["1398","1400"]}}` for multiple
tasks and `{"source":{"platform_task_id":"1398"}}` for one. It validates ID
values and rejects source IDs outside the supplied set. An Init attribution
report needs a summary or findings. Baseline, Train and Eval result reports
need a metrics array with finite numeric values. Dataset reports need a
nonempty files array and use boolean values for optional checks. Retain
existing Markdown, HTML and per-sample outputs for detailed inspection.

For a two-task report, run this from the skill directory with the actual
result, mapping, target paths and verified IDs:

```sh
python3 scripts/export_plan_report.py result.json report_mapping.json \
  --platform-task-id 1398 --platform-task-id 1400 > report.json
```

Check the verified Storage paths with `product_versions`, action
`deliver_plan_outputs`, and `validate_only=true` before recording completion in
`workflowPLAN.json` or submitting the same outputs for registration. Wait for
`report_state=ready` and `report_version_id`
before reporting a formal delivery.
