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
- Sandbox image: `pyrominddynamics/jupyter-lab-with-ssh:v0.9`. Pass this exact
  image to `sandbox_create`; do not substitute an example, local, or inferred
  registry image.
- Runtime: `openhands-embodied-runtime==1.29.5`, centrally provisioned by the
  deployment package index or a deployment-mounted wheel. A platform user must
  not upload a wheel.
- Storage mount: host `/workspace` to sandbox `/target-workspace`.
- Source and target paths are under `/target-workspace`; normalize an input such
  as `workspace/robot/x` or `/workspace/robot/x` to `/target-workspace/robot/x`.
- The source is always read-only at the application level. Output and audit
  paths must not overlap the source.

If the pinned runtime cannot be imported or installed, stop and report a
deployment configuration error. Do not ask an end user for a wheel path.

## Batch Workflow

1. Read [supported source formats](references/source-formats.md). Call
   `sandbox_create` with
   `image="pyrominddynamics/jupyter-lab-with-ssh:v0.9"`, create one CUSTOM
   sandbox with the Storage mount, and wait for `running`. Record its
   `sandbox_id`.
2. Use `sandbox_terminal` to verify `python3.10` and the exact runtime version.
   If the image does not include it, install the pinned package from the
   deployment package index:

   ```bash
   python3.10 -m pip install 'openhands-embodied-runtime==1.29.5'
   ```

3. Choose one UUID `run_id` and create an audit directory
   `/target-workspace/.pyromind-agent/<conversation-id>/embodied-cleaning/<run-id>`.
   Submit the representative plan with `sandbox_terminal`:

   ```bash
   python3.10 -m openhands_embodied_runtime.sandbox_runner \
     --mode plan --source <source> --run-dir <audit-dir> \
     --robot-type s2 --motion-speed-threshold 0.02 \
     --idle-min-duration-s 1.5 --context-s 0.5 \
     --runtime-revision 'openhands-embodied-runtime==1.29.5'
   ```

4. Read `<audit-dir>/report.json` with `sandbox_read_file`. Delete the plan
   sandbox in a finally-style cleanup step, even when planning fails. Ask once
   for the dataset-wide task text, source-derived subtask confirmation, target
   path, and next-state action convention.
5. Create a fresh sandbox with the same mount, verify the pinned runtime, and
   run `--mode full` with the same audit directory, thresholds, `run_id`, target,
   task text, `--confirm-subtasks`, and `--confirm-derived-action`.
6. For a long full run, start the command with `nohup`, redirect stdout/stderr to
   `<audit-dir>/full.log`, save `$!` to `<audit-dir>/full.pid`, and poll with
   `sandbox_terminal` (`kill -0 $(cat <pid>)`) while reading the tail of the log.
   Do not infer completion from silence. Completion is the process exit plus a
   parseable `<audit-dir>/report.json`.
7. If and only if `failed_episode_count > 0`, fix the runtime cause and run the
   same command once with `--mode resume`. Resume retries only `failed` episodes.
   It never reprocesses accepted, needs-review, or rejected episodes.
8. Always delete the sandbox after reading the report. Verify the target with a
   new read-only inspection or Storage preview.

There is no repair phase. Do not retry a quality-rejected episode automatically.
The batch publishes the validated accepted subset when at least one episode is
accepted and there are no runtime failures.
If every episode is rejected, `processing_complete=true` and `published=false`:
report the terminal all-rejected conclusion and do not call resume.

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
