"""Deterministic idle planning, timeline mapping, and quality validation."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openhands_embodied_runtime.adapters import (
    SelfCollectedAdapter,
    build_episode_plan,
    load_self_collected_signals,
)
from openhands_embodied_runtime.models import (
    ActionProvenanceMode,
    EpisodePlan,
    FrameInterval,
    QualityReport,
    QualityStatus,
    SourceType,
    StreamStatus,
    TimelineMapping,
)
from openhands_embodied_runtime.timestamps import (
    common_alignment_times,
    sensor_time_s,
    wall_time_s,
)


def plan_episode_cleaning(
    path: Path,
    *,
    episode_id: str | None = None,
    motion_speed_threshold: float = 0.02,
    idle_min_duration_s: float = 1.5,
    context_s: float = 0.5,
) -> EpisodePlan:
    """Build a reversible plan without modifying source data."""
    if motion_speed_threshold <= 0:
        raise ValueError("motion_speed_threshold must be positive")
    if idle_min_duration_s <= 0:
        raise ValueError("idle_min_duration_s must be positive")
    if context_s < 0:
        raise ValueError("context_s must be non-negative")

    plan = build_episode_plan(path, episode_id=episode_id)
    camera_rows: list[dict[str, str]] = []
    state_rows: list[dict[str, Any]] = []
    if plan.source_type == SourceType.SELF_COLLECTED:
        adapter = SelfCollectedAdapter(path)
        episode_dir = adapter.select(episode_id)
        state_rows, camera_rows = load_self_collected_signals(episode_dir)
        drop_intervals = detect_idle_intervals(
            state_rows,
            camera_rows,
            motion_speed_threshold=motion_speed_threshold,
            idle_min_duration_s=idle_min_duration_s,
            context_s=context_s,
        )
        drop_intervals = preserve_segment_boundaries(
            drop_intervals,
            plan,
            context_s=context_s,
        )
        plan = plan.model_copy(update={"drop_intervals": drop_intervals})

    try:
        mapping = build_timeline_mapping(
            plan,
            camera_rows=camera_rows,
            state_rows=state_rows
            if plan.source_type == SourceType.SELF_COLLECTED
            else None,
        )
    except ValueError as exc:
        quality = plan.quality.model_copy(
            update={
                "status": QualityStatus.REJECTED,
                "errors": [
                    *plan.quality.errors,
                    f"state timeline reconciliation failed: {exc}",
                ],
            }
        )
        plan = plan.model_copy(update={"quality": quality})
        mapping = build_timeline_mapping(plan, camera_rows=camera_rows)
    plan = plan.model_copy(update={"timeline_mapping": mapping})
    return validate_episode_plan(plan)


def finalize_episode_plan(
    source: Path,
    candidate: EpisodePlan,
    *,
    task_text: str,
    confirm_subtasks: bool,
    confirm_derived_action: bool = False,
) -> EpisodePlan:
    """Rebuild source-derived fields and apply only reviewed plan decisions."""
    if not task_text.strip():
        raise ValueError("task_text must contain a natural-language task")
    rebuilt = build_episode_plan(source, episode_id=candidate.episode_id)
    if rebuilt.source_type != candidate.source_type:
        raise ValueError("episode plan source type does not match the source dataset")
    if rebuilt.segments and not confirm_subtasks:
        raise ValueError("subtask ranges require explicit confirmation")

    adapter = SelfCollectedAdapter(source)
    episode_dir = adapter.select(candidate.episode_id)
    state_rows, camera_rows = load_self_collected_signals(episode_dir)
    action_provenance = rebuilt.action_provenance
    if action_provenance is not None:
        action_provenance = action_provenance.model_copy(
            update={"user_confirmed": confirm_derived_action}
        )
    reviewed = rebuilt.model_copy(
        update={
            "task": rebuilt.task.model_copy(
                update={
                    "text": task_text.strip(),
                    "origin": "user",
                    "user_confirmed": True,
                }
            ),
            "segments": [
                segment.model_copy(update={"user_confirmed": confirm_subtasks})
                for segment in rebuilt.segments
            ],
            "drop_intervals": candidate.drop_intervals,
            "action_provenance": action_provenance,
            "quality": rebuilt.quality.model_copy(
                update={
                    "warnings": _unresolved_source_warnings(
                        rebuilt.quality.warnings,
                        confirm_subtasks=confirm_subtasks,
                        confirm_derived_action=confirm_derived_action,
                    )
                }
            ),
        }
    )
    if rebuilt.quality.status == QualityStatus.REJECTED:
        return validate_episode_plan(reviewed)
    mapping = build_timeline_mapping(
        reviewed,
        camera_rows=camera_rows,
        state_rows=state_rows,
    )
    signal_errors = _protected_signal_errors(
        reviewed.drop_intervals,
        state_rows,
        camera_rows,
    )
    quality = reviewed.quality.model_copy(
        update={"errors": [*reviewed.quality.errors, *signal_errors]}
    )
    reviewed = reviewed.model_copy(
        update={"timeline_mapping": mapping, "quality": quality}
    )
    return validate_episode_plan(reviewed)


def detect_idle_intervals(
    joints: list[dict[str, Any]],
    camera_rows: list[dict[str, str]],
    *,
    motion_speed_threshold: float,
    idle_min_duration_s: float,
    context_s: float,
) -> list[FrameInterval]:
    """Return droppable middle ranges of sustained static joint intervals."""
    if len(joints) < 2 or len(camera_rows) < 2:
        return []
    try:
        joint_times, resolved_camera_times, _ = common_alignment_times(
            joints, camera_rows
        )
    except ValueError:
        return []
    names = sorted(
        {
            name
            for row in joints
            for name in row.get("joints", {})
            if isinstance(name, str)
        }
    )
    if not names:
        return []

    static_runs: list[tuple[float, float]] = []
    run_start: float | None = None
    run_end: float | None = None
    for index, (before, current) in enumerate(zip(joints, joints[1:])):
        before_time = joint_times[index]
        current_time = joint_times[index + 1]
        if current_time <= before_time:
            if run_start is not None and run_end is not None:
                static_runs.append((run_start, run_end))
            run_start = run_end = None
            continue
        dt_s = current_time - before_time
        before_joints = before.get("joints", {})
        current_joints = current.get("joints", {})
        if not isinstance(before_joints, dict) or not isinstance(current_joints, dict):
            continue
        max_speed = max(
            abs(_joint_value(current_joints, name) - _joint_value(before_joints, name))
            / dt_s
            for name in names
        )
        transition = before.get("suction_state") != current.get(
            "suction_state"
        ) or before.get("subtask", "") != current.get("subtask", "")
        if max_speed <= motion_speed_threshold and not transition:
            if run_start is None:
                run_start = before_time
            run_end = current_time
        else:
            if run_start is not None and run_end is not None:
                static_runs.append((run_start, run_end))
            run_start = run_end = None
    if run_start is not None and run_end is not None:
        static_runs.append((run_start, run_end))

    intervals: list[FrameInterval] = []
    for start_time, end_time in static_runs:
        if end_time - start_time < idle_min_duration_s:
            continue
        drop_start_time = start_time + context_s
        drop_end_time = end_time - context_s
        if drop_end_time <= drop_start_time:
            continue
        start_frame = bisect.bisect_left(resolved_camera_times, drop_start_time)
        end_frame = bisect.bisect_left(resolved_camera_times, drop_end_time)
        if end_frame > start_frame:
            intervals.append(
                FrameInterval(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    reason="sustained_joint_idle",
                )
            )
    return _merge_intervals(intervals)


def preserve_segment_boundaries(
    intervals: list[FrameInterval],
    plan: EpisodePlan,
    *,
    context_s: float,
) -> list[FrameInterval]:
    """Remove protected neighborhoods around segment boundaries from drop ranges."""
    fps = plan.source_timeline.fps
    if not intervals or not plan.segments or fps is None:
        return intervals
    radius = max(1, round(context_s * fps))
    frame_count = plan.source_timeline.frame_count
    protected: list[tuple[int, int]] = []
    for segment in plan.segments:
        for boundary in (segment.source_start_frame, segment.source_end_frame):
            start = max(0, boundary - radius)
            end = boundary + radius + 1
            if frame_count is not None:
                end = min(frame_count, end)
            protected.append((start, end))

    remaining = intervals
    for protected_start, protected_end in protected:
        split: list[FrameInterval] = []
        for interval in remaining:
            if (
                interval.end_frame <= protected_start
                or interval.start_frame >= protected_end
            ):
                split.append(interval)
                continue
            if interval.start_frame < protected_start:
                split.append(
                    FrameInterval(
                        start_frame=interval.start_frame,
                        end_frame=protected_start,
                        reason=interval.reason,
                    )
                )
            if interval.end_frame > protected_end:
                split.append(
                    FrameInterval(
                        start_frame=protected_end,
                        end_frame=interval.end_frame,
                        reason=interval.reason,
                    )
                )
        remaining = split
    return _merge_intervals(remaining)


def build_timeline_mapping(
    plan: EpisodePlan,
    *,
    camera_rows: list[dict[str, str]] | None = None,
    state_rows: list[dict[str, Any]] | None = None,
    alignment_max_gap_s: float = 0.1,
) -> list[TimelineMapping]:
    frame_count = plan.source_timeline.frame_count
    if frame_count is None:
        return []
    dropped = [False] * frame_count
    for interval in plan.drop_intervals:
        for frame_index in range(
            max(0, interval.start_frame), min(frame_count, interval.end_frame)
        ):
            dropped[frame_index] = True

    fps = plan.source_timeline.fps
    source_origin = _camera_origin(camera_rows or [])
    state_alignment: dict[int, tuple[int, int, float]] = {}
    if state_rows is not None:
        state_alignment = _build_state_alignment(
            state_rows,
            camera_rows or [],
            max_gap_s=alignment_max_gap_s,
        )
    else:
        state_stream = plan.streams.get("state")
        if (
            state_stream is not None
            and state_stream.status == StreamStatus.AVAILABLE
            and state_stream.frame_count == frame_count
        ):
            state_alignment = {
                index: (index, index, 0.0) for index in range(frame_count)
            }
    mapping: list[TimelineMapping] = []
    for source_index, is_dropped in enumerate(dropped):
        if is_dropped:
            continue
        clean_index = len(mapping)
        source_time_s = _source_time(
            source_index, camera_rows or [], source_origin, fps
        )
        clean_time_s = clean_index / fps if fps is not None else None
        state_before_index: int | None = None
        state_after_index: int | None = None
        state_interpolation_weight: float | None = None
        if source_index in state_alignment:
            (
                state_before_index,
                state_after_index,
                state_interpolation_weight,
            ) = state_alignment[source_index]
        mapping.append(
            TimelineMapping(
                source_frame_index=source_index,
                clean_frame_index=clean_index,
                source_time_s=source_time_s,
                clean_time_s=clean_time_s,
                state_before_index=state_before_index,
                state_after_index=state_after_index,
                state_interpolation_weight=state_interpolation_weight,
            )
        )
    return mapping


def validate_episode_plan(plan: EpisodePlan) -> EpisodePlan:
    errors = list(plan.quality.errors)
    warnings = list(plan.quality.warnings)
    frame_count = plan.source_timeline.frame_count
    previous_end = 0
    previous_origin: str | None = None
    for index, segment in enumerate(plan.segments):
        if index and segment.source_start_frame < previous_end:
            message = f"segments[{index}] overlaps the previous segment"
            if "weak_label" in segment.origin or (
                previous_origin is not None and "weak_label" in previous_origin
            ):
                warnings.append(f"{message}; weak labels require validation")
            else:
                errors.append(message)
        previous_end = max(previous_end, segment.source_end_frame)
        previous_origin = segment.origin
        if frame_count is not None and segment.source_end_frame > frame_count:
            errors.append(f"segments[{index}] exceeds the source frame count")

    previous_end = 0
    for index, interval in enumerate(plan.drop_intervals):
        if index and interval.start_frame < previous_end:
            errors.append(f"drop_intervals[{index}] overlaps the previous interval")
        previous_end = max(previous_end, interval.end_frame)
        if frame_count is not None and interval.end_frame > frame_count:
            errors.append(f"drop_intervals[{index}] exceeds the source frame count")
        if any(
            interval.start_frame <= boundary < interval.end_frame
            for segment in plan.segments
            for boundary in (segment.source_start_frame, segment.source_end_frame)
        ):
            errors.append(
                f"drop_intervals[{index}] removes a protected subtask boundary"
            )

    if frame_count is not None:
        dropped_count = sum(
            max(0, min(frame_count, item.end_frame) - max(0, item.start_frame))
            for item in plan.drop_intervals
        )
        expected_mapping_count = frame_count - dropped_count
        if len(plan.timeline_mapping) != expected_mapping_count:
            errors.append("timeline mapping does not cover every retained source frame")
        if any(
            item.clean_frame_index != index
            for index, item in enumerate(plan.timeline_mapping)
        ):
            errors.append("timeline mapping clean indexes are not contiguous")
        source_indexes = [item.source_frame_index for item in plan.timeline_mapping]
        if any(
            current <= previous
            for previous, current in zip(source_indexes, source_indexes[1:])
        ):
            errors.append("timeline mapping source indexes are not strictly increasing")
        if any(index >= frame_count for index in source_indexes):
            errors.append("timeline mapping contains an out-of-range source index")

    if plan.timeline_mapping:
        clean_times = [item.clean_time_s for item in plan.timeline_mapping]
        if plan.source_timeline.fps is not None and any(
            item.clean_time_s is None
            or not math.isclose(
                item.clean_time_s,
                index / plan.source_timeline.fps,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for index, item in enumerate(plan.timeline_mapping)
        ):
            errors.append("timeline mapping clean times are not zero-based")
        resolved_clean_times = [value for value in clean_times if value is not None]
        if len(resolved_clean_times) > 1 and any(
            current <= previous
            for previous, current in zip(resolved_clean_times, resolved_clean_times[1:])
        ):
            errors.append("timeline mapping clean times are not strictly increasing")

    state_stream = plan.streams.get("state")
    if (
        frame_count is not None
        and state_stream is not None
        and state_stream.status == StreamStatus.AVAILABLE
        and state_stream.frame_count != frame_count
        and plan.timeline_mapping
    ):
        if any(
            item.state_before_index is None
            or item.state_after_index is None
            or item.state_interpolation_weight is None
            for item in plan.timeline_mapping
        ):
            errors.append("state stream was not reconciled to the source timeline")
        elif state_stream.frame_count is not None and any(
            item.state_before_index >= state_stream.frame_count
            or item.state_after_index >= state_stream.frame_count
            for item in plan.timeline_mapping
            if item.state_before_index is not None
            and item.state_after_index is not None
        ):
            errors.append("state alignment contains an out-of-range source index")

    invalid_streams = [
        name
        for name, stream in plan.streams.items()
        if stream.status == StreamStatus.INVALID
    ]
    blocking_invalid_streams = [
        name for name in invalid_streams if plan.streams[name].kind != "depth"
    ]
    if blocking_invalid_streams:
        warnings.append(
            "invalid streams require review: " + ", ".join(blocking_invalid_streams)
        )
    optional_invalid_streams = [
        name for name in invalid_streams if plan.streams[name].kind == "depth"
    ]
    if optional_invalid_streams:
        warnings.append(
            "invalid optional depth streams will be omitted from RGB-only output: "
            + ", ".join(optional_invalid_streams)
        )
    missing_required_streams = [
        name
        for name in ("state", "action")
        if name not in plan.streams or plan.streams[name].status == StreamStatus.MISSING
    ]
    if missing_required_streams:
        warnings.append(
            "required streams are missing: " + ", ".join(missing_required_streams)
        )

    not_materialized_streams = [
        name
        for name, stream in plan.streams.items()
        if stream.status == StreamStatus.NOT_MATERIALIZED
    ]
    if not_materialized_streams:
        warnings.append(
            "Storage payloads must be materialized before conversion: "
            + ", ".join(not_materialized_streams)
        )

    review_required = bool(
        blocking_invalid_streams or missing_required_streams or not_materialized_streams
    )
    if (
        plan.task.origin
        in {
            "unspecified",
            "manifest.composition_id",
            "episode_directory",
            "generated",
        }
        and not plan.task.user_confirmed
    ):
        warnings.append(
            "natural-language task text and provenance require user confirmation"
        )
        review_required = True
    action_stream = plan.streams.get("action")
    if action_stream is not None and action_stream.status == StreamStatus.AVAILABLE:
        provenance = plan.action_provenance
        if provenance is None:
            warnings.append("available action stream has no provenance")
            review_required = True
        elif provenance.mode == ActionProvenanceMode.MISSING:
            errors.append("available action stream has missing provenance")
        elif (
            provenance.mode == ActionProvenanceMode.DERIVED
            and not provenance.user_confirmed
        ):
            warnings.append(
                "derived action semantics require explicit user confirmation"
            )
            review_required = True
        if not plan.feature_schema.action:
            warnings.append("action feature names and units are missing")
            review_required = True

    if (
        plan.streams.get("state") is not None
        and plan.streams["state"].status == StreamStatus.AVAILABLE
        and not plan.feature_schema.observation_state
    ):
        warnings.append("observation state feature names and units are missing")
        review_required = True

    for feature_type, features in (
        ("observation state", plan.feature_schema.observation_state),
        ("action", plan.feature_schema.action),
    ):
        feature_names = [feature.name for feature in features]
        if len(feature_names) != len(set(feature_names)):
            errors.append(f"{feature_type} feature names must be unique")
        if any(feature.unit is None for feature in features):
            warnings.append(f"{feature_type} feature units are incomplete")
            review_required = True

    has_suction_segment = any(
        segment.event_type == "suction" for segment in plan.segments
    )
    if (
        has_suction_segment
        and plan.action_provenance is not None
        and plan.action_provenance.mode == ActionProvenanceMode.DERIVED
    ):
        action_names = [feature.name for feature in plan.feature_schema.action]
        state_names = [
            feature.name for feature in plan.feature_schema.observation_state
        ]
        if not any("suction" in name or "gripper" in name for name in action_names):
            warnings.append("derived action schema omits the suction actuator command")
            review_required = True
        if "suction_state" not in state_names:
            warnings.append("observation state schema omits suction_state")
            review_required = True

    if plan.source_type == SourceType.SELF_COLLECTED:
        absolute_streams = [
            name
            for name, stream in plan.streams.items()
            if stream.path and Path(stream.path).is_absolute()
        ]
        if absolute_streams:
            warnings.append(
                "self-collected stream paths must be portable and relative: "
                + ", ".join(absolute_streams)
            )
            review_required = True
        if (
            plan.action_provenance is not None
            and plan.action_provenance.source_path is not None
            and Path(plan.action_provenance.source_path).is_absolute()
        ):
            warnings.append("action provenance source_path must be relative")
            review_required = True

    errors = _unique(errors)
    warnings = _unique(warnings)
    if not plan.segments and plan.source_type == SourceType.SELF_COLLECTED:
        warnings.append("no trusted subtask labels are available")
        review_required = True
    elif any(not segment.user_confirmed for segment in plan.segments):
        warnings.append("source-derived subtask ranges require explicit confirmation")
        review_required = True

    if errors:
        status = QualityStatus.REJECTED
    elif review_required:
        status = QualityStatus.NEEDS_REVIEW
    else:
        status = QualityStatus.ACCEPTED
    return plan.model_copy(
        update={
            "quality": QualityReport(
                status=status,
                errors=errors,
                warnings=warnings,
            )
        }
    )


def reindex_timeline_mapping(
    mapping: Sequence[TimelineMapping],
    *,
    fps: float | None,
) -> list[TimelineMapping]:
    """Renumber a filtered mapping and reset its clean clock to zero."""
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive")
    return [
        item.model_copy(
            update={
                "clean_frame_index": index,
                "clean_time_s": index / fps if fps is not None else None,
            }
        )
        for index, item in enumerate(mapping)
    ]


def _merge_intervals(intervals: list[FrameInterval]) -> list[FrameInterval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (item.start_frame, item.end_frame))
    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start_frame <= previous.end_frame:
            merged[-1] = FrameInterval(
                start_frame=previous.start_frame,
                end_frame=max(previous.end_frame, current.end_frame),
                reason=previous.reason,
            )
        else:
            merged.append(current)
    return merged


def _camera_origin(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    sensor_origin = sensor_time_s(rows[0])
    return sensor_origin if sensor_origin is not None else wall_time_s(rows[0])


def _source_time(
    index: int,
    rows: list[dict[str, str]],
    origin: float | None,
    fps: float | None,
) -> float | None:
    if index < len(rows) and origin is not None:
        value = sensor_time_s(rows[index])
        if value is None:
            value = wall_time_s(rows[index])
        if value is not None:
            return max(0.0, value - origin)
    return index / fps if fps is not None else None


def _build_state_alignment(
    state_rows: list[dict[str, Any]],
    camera_rows: list[dict[str, str]],
    *,
    max_gap_s: float,
) -> dict[int, tuple[int, int, float]]:
    if max_gap_s <= 0:
        raise ValueError("alignment_max_gap_s must be positive")
    if not state_rows or not camera_rows:
        raise ValueError("state and camera timestamp rows are required")
    resolved_state_times, resolved_camera_times, _ = common_alignment_times(
        state_rows, camera_rows
    )
    if any(
        current <= previous
        for previous, current in zip(resolved_state_times, resolved_state_times[1:])
    ):
        raise ValueError("state timestamps are not strictly increasing")
    if any(
        current < previous
        for previous, current in zip(resolved_camera_times, resolved_camera_times[1:])
    ):
        raise ValueError("camera timestamps are decreasing")

    alignment: dict[int, tuple[int, int, float]] = {}
    for camera_index, camera_time in enumerate(resolved_camera_times):
        after_index = bisect.bisect_left(resolved_state_times, camera_time)
        if after_index == 0:
            if resolved_state_times[0] - camera_time > max_gap_s:
                raise ValueError(f"camera frame {camera_index} precedes state coverage")
            alignment[camera_index] = (0, 0, 0.0)
            continue
        if after_index == len(resolved_state_times):
            if camera_time - resolved_state_times[-1] > max_gap_s:
                raise ValueError(f"camera frame {camera_index} exceeds state coverage")
            last_index = len(resolved_state_times) - 1
            alignment[camera_index] = (last_index, last_index, 0.0)
            continue
        if resolved_state_times[after_index] == camera_time:
            alignment[camera_index] = (after_index, after_index, 0.0)
            continue
        before_index = after_index - 1
        interval_s = (
            resolved_state_times[after_index] - resolved_state_times[before_index]
        )
        if interval_s > max_gap_s:
            raise ValueError(
                f"camera frame {camera_index} falls inside a {interval_s:.6f}s "
                "state gap"
            )
        weight = (camera_time - resolved_state_times[before_index]) / interval_s
        alignment[camera_index] = (before_index, after_index, weight)
    return alignment


def _joint_value(values: dict[str, Any], name: str) -> float:
    value = _as_float(values.get(name))
    return value if value is not None else 0.0


def _protected_signal_errors(
    intervals: list[FrameInterval],
    state_rows: list[dict[str, Any]],
    camera_rows: list[dict[str, str]],
) -> list[str]:
    if not intervals or len(state_rows) < 2 or not camera_rows:
        return []
    try:
        state_times, camera_times, _ = common_alignment_times(state_rows, camera_rows)
    except ValueError:
        return []
    protected_frames = {
        min(
            bisect.bisect_left(camera_times, state_times[index]),
            len(camera_rows) - 1,
        )
        for index in range(1, len(state_rows))
        if state_rows[index - 1].get("suction_state")
        != state_rows[index].get("suction_state")
    }
    return [
        f"drop_intervals[{interval_index}] removes a suction transition frame"
        for interval_index, interval in enumerate(intervals)
        if any(
            interval.start_frame <= frame_index < interval.end_frame
            for frame_index in protected_frames
        )
    ]


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _unresolved_source_warnings(
    warnings: list[str],
    *,
    confirm_subtasks: bool,
    confirm_derived_action: bool,
) -> list[str]:
    resolved_prefixes = ["task metadata is missing;"]
    if confirm_subtasks:
        resolved_prefixes.append("subtask ranges were imported from")
    if confirm_derived_action:
        resolved_prefixes.append("action will use the reference-dataset convention:")
    return [
        warning
        for warning in warnings
        if not any(warning.startswith(prefix) for prefix in resolved_prefixes)
    ]
