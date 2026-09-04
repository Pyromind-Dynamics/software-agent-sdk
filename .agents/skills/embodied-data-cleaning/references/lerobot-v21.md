# LeRobotDataset v2.1 Output

Materialize only version `v2.1`. Do not emit v3 files, indexes, or compatibility
metadata. The output must use this structure:

```text
dataset_root/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/
    └── observation.images.head/episode_000000.mp4
```

Add another video directory only for a valid video stream. Omit empty or invalid
depth rather than creating a placeholder feature.

Valid S2 raw depth uses `head_depth.u16`, `head_depth.csv`, and
`head_depth_meta.json`. Validate byte size as `frames * width * height * 2`. For
each retained primary RGB frame, use its camera timestamp to select the nearest
depth timestamp within the alignment tolerance; repeated depth indexes are valid
when the depth stream is slower. Write JET-colored RGB MP4 under
`observation.images.head_depth`. Match the reference conversion by applying
`depth_scale`, clipping at 3 meters, and mapping the resulting 8-bit values with
JET. Declare `s2_depth_video_colormap: "jet"`.

## Parquet contract

Each episode has one Parquet file with these columns in order:

1. `observation.state`: `list<float32>`
2. `action`: `list<float32>`
3. `timestamp`: `float32`, equal to `frame_index / fps`
4. `frame_index`: zero-based contiguous `int64`
5. `episode_index`: `int64`
6. `index`: global zero-based contiguous `int64`
7. `task_index`: `int64`

The supplied reference dataset defines action as a next-state target:
`action[t] = observation.state[t+1]`. The last action repeats the final state.
Its S2 state/action profile has 24 ordered values: 22 named joints followed by
`left_grip_pos` and `right_grip_pos` from end-effector state index 4.

## Metadata contract

`meta/info.json` declares `codebase_version: "v2.1"`, totals, `fps`, train split,
chunk size, path templates, and every feature. Use these exact templates:

```text
data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
```

`episodes.jsonl` records episode index, task list, and length. `tasks.jsonl` maps
task indexes to confirmed task text. `episodes_stats.jsonl` stores min, max, mean,
standard deviation, and count for every numeric Parquet feature.

MP4 FPS and frame count are authoritative for output timing. Camera CSV timestamps
are alignment evidence, not a replacement video clock. Keep the reviewed
`episode_plan.json` outside the final dataset root so the training directory stays
aligned with the reference case.
