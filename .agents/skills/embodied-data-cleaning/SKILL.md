---
name: embodied-data-cleaning
description: Inspect, clean, align, validate, and convert self-collected S2 robot recordings with joints.jsonl, camera CSV/MP4, manifest metadata, and optional depth into LeRobotDataset v2.1. Use for static-frame removal, subtask validation, multimodal timestamp alignment, v2.1 materialization, and embodied-data quality reports.
---

# Embodied Data Cleaning

Build a reversible representative plan, then process the confirmed dataset with
one resumable deterministic batch call. Do not narrate skill selection to the
user. Let models interpret physical state changes; let deterministic tools
calculate frames, timestamps, intervals, output files, and validation results.

## Workflow

1. Materialize the required Storage files once, then run
   `inspect_embodied_dataset` with `sample_limit=3`. Resolve every invalid or
   `not_materialized` required stream before conversion. Treat empty files as
   invalid; do not call a large Storage object invalid merely because preview
   skipped its payload.
2. Read [self-collected source format](references/source-formats.md) and run
   `build_embodied_episode_plan` for one representative episode. Prefer contiguous
   `joints.jsonl` subtask runs, then manifest steps as weak fallback labels.
3. Ask once for dataset-level confirmation of task text, source-derived subtask
   ranges, and the reference dataset's next-state action convention. Use one task
   for the complete dataset, for example `Pick and place the item on the table`.
   Treat color or shape prompts as target metadata, not separate training tasks.
4. Read [LeRobot v2.1 output](references/lerobot-v21.md), then call
   `batch_clean_le_robot_v21` once. It rebuilds and finalizes every plan, isolates
   quality rejections, materializes accepted episodes, validates each result,
   checkpoints after each episode, and deterministically merges the batch. Do not
   repeat per-episode build/finalize/materialize/validate calls in the normal path.
5. If the batch reports runtime failures, resolve the cause and call it again with
   the same arguments and `resume=true`. Do not restart accepted or quality-rejected
   episodes. Read detailed errors from `report_path`; keep tool observations compact.
6. Call `validate_le_robot_v21` once on the merged local output. Treat the local
   directory as an intermediate artifact, not the final delivery location.
7. Call `publish_le_robot_v21` once with an explicit Pyromind Storage dataset path,
   then run `preview_dataset(mode="inspect")` on that path. Completion requires
   `complete=true` and a matching remote file listing. Report the `/workspace/...`
   platform path plus deterministic discovered/accepted/rejected/frame/video counts.

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
  tool and its report for timestamp failures and rejected IDs.
- Treat quality rejection as a completed episode decision. Treat unexpected tool
  exceptions as failures that keep the batch incomplete and resumable.
- Do not publish a representative sample. Publish only the final merged dataset.
- Do not present a conversation-local path as a delivered dataset. Completion
  requires `publish_le_robot_v21.complete=true` and a successful Storage listing.
- Publish only validated LeRobot v2.1 output files. Never upload episode plans,
  raw recordings, temporary files, or credentials with the final dataset.

Read [canonical schema](references/canonical-schema.md) when consuming or extending
`EpisodePlan`. Read [quality gates](references/quality-gates.md) before accepting an
episode or adding a validator. Read
[LeRobot v2.1 output](references/lerobot-v21.md) before materialization.

## Bundled Script

Run the deterministic inspector without starting an Agent:

```bash
uv run python .agents/skills/embodied-data-cleaning/scripts/inspect_dataset.py \
  /absolute/path/to/dataset --sample-limit 3
```

The script prints JSON and does not modify the source dataset.
