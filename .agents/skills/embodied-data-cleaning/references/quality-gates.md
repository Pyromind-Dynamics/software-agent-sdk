# Quality Gates

## Accepted

- Required files parse successfully.
- Episode IDs are unique.
- Segment ranges are ordered, non-overlapping, and within known frame bounds.
- State and action samples have deterministic mappings onto every retained RGB
  frame. Raw stream counts may differ when sensors use different sampling rates.
- Every secondary RGB or depth frame is selected by timestamp from its own stream;
  primary RGB frame indexes are not reused as cross-stream alignment.
- Timeline mapping is complete, monotonic, and starts at clean frame zero.
- Clean time starts at zero and equals `clean_frame_index / fps`.
- Ordered state/action feature names, units, and sources are present.
- Action provenance is `derived/next_state`, matching the supplied v2.1 reference:
  action is the next cleaned state and the last action repeats the last state.
- Self-collected stream and provenance paths are relative and portable.
- Task text has source provenance; generated text and weak identifiers were
  explicitly confirmed.
- Source-derived subtask ranges and derived next-state action semantics were
  explicitly confirmed and recorded in the plan.
- No drop interval removes a protected transition boundary.

## Needs review

- Matching RGB video or state samples are unavailable.
- The first label starts after frame zero or the final frame bound is unknown.
- Pick/place visual evidence or end-effector evidence is inconclusive.
- A recoverable required stream is missing or invalid.
- An optional depth stream is invalid and will be omitted from RGB-only output.
- A Storage payload is known to exist but is not materialized locally.
- Derived action semantics are documented but not explicitly confirmed.
- Source-derived subtask ranges are not explicitly confirmed.
- Feature schema or action provenance is incomplete.
- Task text is generated, missing, or only a weak identifier and is unconfirmed.

## Rejected

- Required metadata cannot be parsed.
- Segment ranges are invalid or irreconcilably overlap.
- The source-to-clean mapping is non-monotonic or loses a protected frame.
- A required stream cannot be mapped onto the RGB clock within the alignment gap
  limit, or reconciled samples are missing after alignment.
- Clean indexes and clean timestamps disagree after frame filtering.

Warnings alone do not block inspection. Conversion requires `accepted` and must not
silently downgrade errors to warnings.

Never trust a serialized `quality.status` during conversion. Rebuild source-derived
fields and rerun these gates so editing `accepted` or clearing `errors` cannot
bypass validation.
