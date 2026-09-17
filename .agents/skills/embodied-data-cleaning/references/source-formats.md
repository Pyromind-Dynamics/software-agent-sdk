# Source Formats

## Self-collected S2 recordings

Detect a directory containing `joints.jsonl` plus at least one camera timestamp CSV.
Use `manifest.json` as weak task and step metadata. MP4 frame count and FPS define
the media clock. Align joints and camera rows with their shared sensor clock:
prefer `stamp_ns`, then `stamp_sec` plus `stamp_nsec`. Use `wall_time` only when
both streams lack sensor stamps. Manifest `started_at` and `completed_at` values
remain in the wall-time domain.

Expected optional streams include RGB MP4, raw depth `.u16`, depth metadata, and
per-camera CSV timestamps. An empty depth file is invalid and must be omitted; it
does not block a valid RGB-only output.

A payload listed by Storage but skipped by bounded preview is `not_materialized`,
not missing, empty, or invalid. Materialize and validate the full payload before
conversion.

State and camera streams commonly use different sampling rates. Reconcile state
samples onto the RGB clock with source timestamps and store the bracketing state
indexes plus interpolation weight. Do not reject an episode solely because raw
stream counts differ.

Camera streams can also contain equal frame counts with different duplicate or
dropped timestamps. Treat the primary RGB CSV as the retained-frame reference and
map each secondary RGB/depth stream independently by nearest timestamp within the
alignment tolerance. Allow repeated target indexes for slower streams.

For the S2 reference profile, build the 24-dimensional `observation.state` from
the fixed 22-joint order plus `left_ee_states[4]` and `right_ee_states[4]` as
`left_grip_pos` and `right_grip_pos`. Use `suction_state` to protect cleaning
boundaries, but do not append it to this profile. If recorded gripper values are
unavailable, do not invent zeros. After interpolation, derive action as the next
cleaned state.

## Mounted Hugging Face LeRobot v2.1 datasets

Detect a complete LeRobot v2.1 root before detecting a standalone online label
file. The mounted source must contain:

```text
dataset_root/
├── meta/info.json
├── meta/episodes.jsonl
├── meta/tasks.jsonl
├── data/chunk-*/episode_*.parquet
└── videos/chunk-*/observation.images.*/episode_*.mp4
```

Use the MP4 frame count and FPS as the source timeline. Require each episode
Parquet to contain finite numeric `observation.state` and `action` vectors, and
require its row count to match every declared episode MP4. Preserve declared
feature order. Read feature units from `features.<key>.units`; for the fixed S2
24-dimensional profile only, infer joint units as radians and gripper units as
meters when older v2.1 metadata omitted units. Do not guess units for other robot
profiles.

An optional `labels.json`, `annotations.json`, or `labels.jsonl` beside `meta/`
may contain `episode_id`, `task_name`, `init_scene_text`, and half-open
`label_info.action_config` frame ranges. Join labels in this order:

1. `source_episode_id` in `meta/episodes.jsonl`;
2. `episode_id` in `meta/episodes.jsonl`;
3. one unique `episode_id` value in the episode Parquet;
4. the numeric LeRobot `episode_index` as a compatibility fallback.

Duplicate source IDs, overlapping ranges, and ranges beyond the authoritative
MP4/Parquet frame count are invalid. An unmatched sidecar episode requires review.
Frames before the first label remain unlabeled; never silently assign them to the
first action or delete them solely because they lack a label.

A standalone label `.doc`, JSON, or JSONL file is not a trainable dataset. Convert
legacy `.doc` exports to UTF-8 JSON for inspection, but block full conversion until
the matching state, action, and media payload is mounted in the same dataset root.
