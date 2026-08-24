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
