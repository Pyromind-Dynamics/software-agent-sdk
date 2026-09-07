---
name: embodied-data-cleaning
description: Inspect, clean, align, validate, and convert self-collected S2 recordings or mounted Hugging Face LeRobot v2.1 datasets with optional online episode/action labels into validated LeRobotDataset v2.1. Use for static-frame removal, label joins, subtask validation, multimodal alignment, batch conversion, and embodied-data quality reports in a Python 3.10 sandbox.
---

# Embodied Data Cleaning

This is an environment-processing case under `data-processing`. Use the generic
Sandbox lifecycle and the deterministic `openhands-embodied-runtime` package.
Do not use Studio workflows, `run_embodied_cleaning_sandbox`, local embodied
Agent tools, or per-episode ad hoc scripts.

## Runtime Contract

- Sandbox Python: 3.10.
- Sandbox image: `pyrominddynamics/jupyter-lab-with-ssh:v0.9`. This image is
  the manifest `image` field for every episode record; do not substitute an
  example, local, or inferred registry image.
- Runtime: `openhands-embodied-runtime==1.29.5`, shipped inside the skill
  bundle (`edp/wheels/`) and staged into the run directory by `edp_submit`;
  the profile exec installs it from the run dir on the Storage mount
  (third-party deps resolve from public PyPI). No agent-side wheel
  provisioning; a missing wheel bundle fails at submit time. Image preinstall
  remains the productized end state.
- Storage mount: host `/workspace` is mounted read-write into the sandbox at
  `/target-workspace` (declared by the profile's `volume_mounts`); per-episode
  outputs and result JSONs land there and outlive the sandbox.
- Source and target paths are under `/target-workspace`; normalize an input such
  as `workspace/robot/x` or `/workspace/robot/x` to `/target-workspace/robot/x`.
- The source is always read-only at the application level. Output and audit
  paths must not overlap the source.

If the pinned runtime cannot be imported or installed, stop and report a
deployment configuration error. Do not ask an end user for a wheel path.

## Batch Workflow (EDP orchestration)

Execution mirrors the tmax case: the platform runs every episode through
`edp_render` / `edp_submit`; the agent submits and reads verdicts, it never
drives long-running commands. Profiles:
`data-processing/scripts/edp/profiles/embodied-cleaning.json` (per-episode)
and `embodied-cleaning-merge.json` (aggregate publish).

1. Read [supported source formats](references/source-formats.md). Confirm the
   source shape with `preview_dataset` (Storage paths only; no local reads).
2. Render the episode shards: write a render template pointing
   `data_source` at `meta/episodes/chunk-*.parquet`; every record carries
   `episode_id`, the pinned `image`, and a `config_json` blob (source,
   work_root, task text, thresholds). Run `edp_render`.
3. Plan stage: run `--mode plan` for one representative episode (smoke-sized
   sandbox or a small `edp_submit` batch) and show the user the representative
   plan plus inspection summary.
4. Confirm once with the user: dataset-wide task text, subtask ranges,
   next-state action convention, thresholds, and target path. No render or
   submit before this gate.
5. `edp_submit(manifest=..., limit=3, profile_name="embodied-cleaning")`
   smoke (always pass profile_name explicitly; the tool defaults to
   tmax-validation), then triage verdicts
   (`reward` 1.0 accepted / 0.5 needs-review / 0.0 rejected; errors split out
   by `error_category`, image-missing failures get their own bucket).
6. After user confirmation submit the remaining shards in batches; observe
   with `df_check_progress`. Resume is shard-based: failed episodes are the
   only records worth resubmitting.
7. When every episode reached a terminal state, submit the single merge
   record with `embodied-cleaning-merge`. It merges accepted fragments only,
   validates LeRobot v2.1, and publishes to the target. Verify the published
   dataset and merge report with `preview_dataset`.

There is no repair phase. Do not retry a quality-rejected episode
automatically. If every episode is rejected, `processing_complete=true` and
`published=false`: report the terminal all-rejected conclusion and do not
resubmit.

## Alignment and Rejection Policy

- Use MP4 frame count and FPS as the primary media clock.
- Timestamp-align every retained RGB frame to state and secondary streams.
- A camera/state lead or lag, internal state gap, or primary/secondary camera
  gap above 100 ms is a warning.
- Reject only when that gap is greater than 500 ms. Exactly 500 ms is accepted.
- Other terminal rejection gates include invalid or overlapping subtask ranges,
  missing/corrupt required Parquet or MP4 payloads, unusable timeline mappings,
  and incompatible state/action schemas.
- Apply the same 500 ms gate to secondary RGB/depth alignment; report the
  affected stream and whether the primary stream leads or lags.
- Every rejected episode must appear in `rejected_episode_reports` with
  `episode_id`, `stage`, `error_code`, `message`, measured details when
  available, and a suggested source-data check.
- A rejection is not a batch failure. Report accepted and rejected counts and
  list each rejection reason; never describe the whole dataset as unusable when
  an accepted subset was published.

## Data Invariants

- Reconcile different sensor rates by timestamp; raw stream counts need not
  match.
- Use half-open intervals `[start_frame, end_frame)` and preserve context around
  motion, suction transitions, and subtask boundaries.
- Record every dropped frame through source-to-clean timeline mappings and reset
  clean indexes/timestamps after the drop.
- Join optional `labels.json`, `annotations.json`, or `labels.jsonl` by stable
  source episode ID before falling back to LeRobot episode index.
- Match LeRobot v2.1 next-state actions:
  `action[t] = observation.state[t+1]`, repeating the final state at the end.
- Never use an LLM as the final format, interval, or alignment validator.
- Publish only validated `meta/`, `data/`, and `videos/` output, never audit
  plans, raw data, logs, temporary files, or credentials.

Read [canonical schema](references/canonical-schema.md) before extending an
episode plan, [quality gates](references/quality-gates.md) before changing a
validator, and [LeRobot v2.1 output](references/lerobot-v21.md) before
materialization.

## Local Developer CLI

For an explicitly local developer dataset only, use
`scripts/embodied_cli.py` through Terminal. It supports `inspect`, `plan`,
`clean --mode full`, `clean --mode resume`, and `validate`. It never modifies
the source and does not publish to Storage.
