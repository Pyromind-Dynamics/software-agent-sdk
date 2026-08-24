"""Human-readable summaries for embodied-data cleaning plans."""

from __future__ import annotations

from openhands.tools.embodied_data.models import EpisodePlan, QualityStatus


def render_episode_plan_summary(plan: EpisodePlan) -> str:
    """Render the decision and important counts without exposing raw vectors."""
    source_frames = plan.source_timeline.frame_count
    retained_frames = len(plan.timeline_mapping)
    dropped_frames = (
        source_frames - retained_frames if source_frames is not None else None
    )
    can_convert = plan.quality.status == QualityStatus.ACCEPTED
    subtasks_confirmed = bool(plan.segments) and all(
        segment.user_confirmed for segment in plan.segments
    )
    action_confirmed = bool(
        plan.action_provenance and plan.action_provenance.user_confirmed
    )
    lines = [
        f"# Episode {plan.episode_id} cleaning summary",
        "",
        "## Decision",
        "",
        f"- Quality: `{plan.quality.status.value}`",
        f"- Ready for LeRobot conversion: `{'yes' if can_convert else 'no'}`",
        f"- Task: {plan.task.text}",
        f"- Task source: {plan.task.origin}",
        f"- Task confirmed by user: `{'yes' if plan.task.user_confirmed else 'no'}`",
        "",
        "## Frames",
        "",
        f"- Source RGB frames: {_display(source_frames)}",
        f"- Retained frames: {_display(retained_frames)}",
        f"- Dropped frames: {_display(dropped_frames)}",
        f"- Subtask segments: {len(plan.segments)}",
        f"- Subtask ranges confirmed: `{'yes' if subtasks_confirmed else 'no'}`",
        "",
        "## Training features",
        "",
        f"- Observation state dimensions: {len(plan.feature_schema.observation_state)}",
        f"- Action dimensions: {len(plan.feature_schema.action)}",
        "- Action provenance: "
        f"{plan.action_provenance.mode.value if plan.action_provenance else 'unknown'}",
        f"- Derived action confirmed by user: `{'yes' if action_confirmed else 'no'}`",
    ]
    if plan.quality.errors:
        lines.extend(["", "## Blocking errors", ""])
        lines.extend(f"- {error}" for error in plan.quality.errors)
    if plan.quality.warnings:
        lines.extend(["", "## Review items", ""])
        lines.extend(f"- {warning}" for warning in plan.quality.warnings)
    if plan.drop_intervals:
        lines.extend(["", "## Proposed removals", ""])
        fps = plan.source_timeline.fps
        for interval in plan.drop_intervals:
            frame_count = interval.end_frame - interval.start_frame
            time_range = "time unavailable"
            if fps is not None:
                time_range = (
                    f"{interval.start_frame / fps:.3f}s–{interval.end_frame / fps:.3f}s"
                )
            lines.append(
                f"- Frames `[{interval.start_frame}, {interval.end_frame})`, "
                f"{time_range}, {frame_count} frames: {interval.reason}"
            )
    if plan.segments:
        lines.extend(["", "## Subtask ranges", ""])
        for segment in plan.segments:
            lines.append(
                f"- Frames `[{segment.source_start_frame}, "
                f"{segment.source_end_frame})`: {segment.instruction} "
                f"(`{segment.origin}`, confirmed="
                f"{'yes' if segment.user_confirmed else 'no'})"
            )
    lines.extend(
        [
            "",
            "## What the JSON file is",
            "",
            "The episode plan is a machine-readable, reversible audit record. It is ",
            "not a cleaned video and not a packaged LeRobot dataset.",
            "",
        ]
    )
    return "\n".join(lines)


def _display(value: int | None) -> str:
    return str(value) if value is not None else "unknown"
