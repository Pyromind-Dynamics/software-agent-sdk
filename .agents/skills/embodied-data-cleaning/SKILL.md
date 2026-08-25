---
name: embodied-data-cleaning
description: Inspect, clean, align, validate, and convert self-collected S2 robot recordings with joints.jsonl, camera CSV/MP4, manifest metadata, and optional depth into LeRobotDataset v2.1. Use for static-frame removal, subtask validation, multimodal timestamp alignment, v2.1 materialization, and embodied-data quality reports.
---

# Embodied Data Cleaning

Build a reversible representative plan, then process the confirmed dataset with
one resumable deterministic sandbox batch. Do not narrate skill selection to the
user. Let models interpret physical state changes; let deterministic sandbox
tools calculate frames, timestamps, intervals, output files, and validation
results. Never materialize robot payloads in the conversation workspace.

## Workflow

1. Read [self-collected source format](references/source-formats.md), then call
   `run_embodied_cleaning_sandbox` with `mode="plan"`, the Storage source path,
   and no `run_id`. The sandbox mounts Storage directly, inspects at most three
   summaries, and writes one representative reversible plan. Do not call
   `materialize_storage_files`, Terminal, or local embodied-data tools.
2. When the platform callback resumes the conversation, call `preview_dataset`
   on `<output_dir>/report.json` and the representative plan or summary paths in
   that report. Resolve every invalid required stream before conversion. Treat
   empty files as invalid; a bounded preview that skipped a large payload is not
   evidence that the mounted sandbox payload is invalid.
3. Ask once for dataset-level confirmation of task text, source-derived subtask
   ranges, and the reference dataset's next-state action convention. Use one task
   for the complete dataset, for example `Pick and place the item on the table`.
   Treat color or shape prompts as target metadata, not separate training tasks.
4. Read [LeRobot v2.1 output](references/lerobot-v21.md), then call the same tool
   with `mode="full"`, the plan `run_id`, explicit target Storage path, confirmed
   task, `confirm_subtasks=true`, and `confirm_derived_action=true`. The fixed
   cleaning thresholds and `robot_type` must match the plan phase. The sandbox
   runtime rebuilds every plan, isolates quality rejections, materializes accepted
   episodes, checkpoints each episode, merges, validates, and publishes.
5. If the full report contains runtime failures, resolve the cause and call the
   same tool with `mode="resume"`, the same arguments, and the same `run_id`.
   Never submit another `full` phase for that run. Do not restart accepted or
   quality-rejected episodes.
6. After every callback, inspect `<output_dir>/report.json` only with
   `preview_dataset`. Do not read sandbox artifacts through Terminal or workspace
   file APIs. Quality rejection is a completed episode decision; runtime failure
   is resumable.
7. Completion requires `report.complete=true`, `report.published=true`, a valid
   final validation block, and `preview_dataset(mode="inspect")` on the target
   Storage path with matching files. Report the `/workspace/...` target plus
   deterministic discovered/accepted/rejected/frame/video counts.

## Invariants

- Use MP4 frame count and FPS as the output media clock. Use camera CSV timestamps
  only to align state and labels; never infer video duration from wall time.
- Reconcile different sensor rates by timestamp; raw stream counts do not need to
  match.
- Align every retained RGB frame independently to each secondary RGB or depth
  stream with their shared sensor timestamps. Never reuse primary RGB frame indexes
  for another stream merely because their raw frame counts match.
- Use half-open frame intervals: `[start_frame, end_frame)`.
- Do not scale camera wall-time and MP4 PTS by a constant ratio.
- Preserve context around motion, suction changes, and subtask boundaries.
- Record every dropped frame through a source-to-clean timeline mapping.
- Reset `clean_frame_index` and `clean_time_s` to zero after any additional drop.
- Keep stream and provenance paths relative to the episode directory.
- Record feature names, units, and sources before producing numeric vectors.
- Match the supplied v2.1 reference convention: after cleaning and alignment,
  `action[t] = observation.state[t+1]`; repeat the final state for the final action.
  Record this as `derived/next_state`. For the fixed 24-dimensional S2 reference
  profile, use suction state as transition evidence rather than a vector feature.
- Do not treat a suction command as proof that a grasp succeeded.
- Do not assign confidence `1.0` merely because manifest ranges were made
  non-overlapping; report whether visual or execution evidence was checked.
- Do not use an LLM as the final format or interval validator.
- Do not use terminal heredocs or ad hoc scripts to iterate episodes. Use the batch
  sandbox tool and its report for timestamp failures and rejected IDs.
- Treat quality rejection as a completed episode decision. Treat unexpected tool
  exceptions as failures that keep the batch incomplete and resumable.
- Do not publish a representative sample. Publish only the final merged dataset.
- Do not present a conversation-local or sandbox run path as a delivered dataset.
  Completion requires `report.complete=true` and a successful target Storage listing.
- Publish only validated LeRobot v2.1 output files. Never upload episode plans,
  raw recordings, temporary files, or credentials with the final dataset.
- Treat `output_dir` as audit/checkpoint Storage and `target_path` as the only
  deliverable dataset. Never publish files from the audit directory as training data.

Read [canonical schema](references/canonical-schema.md) when consuming or extending
`EpisodePlan`. Read [quality gates](references/quality-gates.md) before accepting an
episode or adding a validator. Read
[LeRobot v2.1 output](references/lerobot-v21.md) before materialization.

## Bundled Script

This local inspector is for developer diagnostics only. The platform Agent must
use `run_embodied_cleaning_sandbox` and must not invoke this script.

Run the deterministic inspector without starting an Agent:

```bash
uv run python .agents/skills/embodied-data-cleaning/scripts/inspect_dataset.py \
  /absolute/path/to/dataset --sample-limit 3
```

The script prints JSON and does not modify the source dataset.
