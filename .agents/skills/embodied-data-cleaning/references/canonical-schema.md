# Canonical Episode Schema

`EpisodePlan` is the boundary between source-specific adapters and downstream
cleaning or conversion.

## Required identity

- `schema_version`: currently `1`.
- `episode_id`: stable source identifier represented as a string.
- `source_type`: `self_collected` for the supported production path.
- `task`: text plus its source and whether user confirmation was required. Do not
  replace a weak identifier such as `composition_id=demo` with generated prose
  without recording `origin=generated` and obtaining confirmation.

## Timeline

- `basis`: `frame_index` or `stamp_ns`.
- `frame_count`: number of RGB source frames when known.
- `fps`: measured or declared source rate when reliable.
- `start_stamp_ns` and `end_stamp_ns`: preserve source clock values when present.

All segments and cleaning intervals are half-open. A segment with start `8` and
end `218` contains source frames 8 through 217.

## Segments

Each segment contains source frame boundaries, instruction, normalized event type,
optional source timestamps, and origin. Keep model confidence separate from
deterministic validation status. `user_confirmed` records whether the reviewed
source-derived range was explicitly approved; source parsing must leave it false.

## Cleaning plan

- `drop_intervals`: source frame ranges proposed for removal.
- `timeline_mapping`: one record per retained frame with source and clean indexes.
  For self-collected multi-rate data, each record also stores the bracketing state
  row indexes and interpolation weight used to reconcile state onto the RGB clock.
- `feature_schema`: ordered observation/action feature names, units, and sources.
- `action_provenance`: whether action was recorded, derived, or missing. The v2.1
  self-collected path uses `derived/next_state` with a relative source path,
  description, and user-confirmation bit.
- `quality`: `pending`, `accepted`, `needs_review`, or `rejected`, plus errors and
  warnings. This serialized value is informational. Finalization rebuilds streams,
  feature schema, segments, alignment, and quality from the source, then applies
  only reviewed task text and drop intervals.

Stream status distinguishes `not_materialized` (known Storage object unavailable
in the bounded local preview) from `missing` and `invalid`.

After filtering a mapping, renumber `clean_frame_index` from zero and recompute
`clean_time_s = clean_frame_index / fps`. Stream and provenance paths must be
relative to the episode directory so downloaded plans remain portable.

Do not write runtime audit fields into the final training examples.
