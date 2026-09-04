"""Resumable deterministic batch cleaning for supported embodied datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openhands_embodied_runtime.adapters import (
    LeRobotV21Adapter,
    SelfCollectedAdapter,
    detect_source_type,
)
from openhands_embodied_runtime.lerobot_v21 import (
    materialize_finalized_lerobot_v21_episode,
    merge_lerobot_v21_datasets,
    validate_lerobot_v21_dataset,
)
from openhands_embodied_runtime.models import EpisodePlan, QualityStatus, SourceType
from openhands_embodied_runtime.planning import (
    finalize_episode_plan,
    plan_episode_cleaning,
)
from openhands_embodied_runtime.reporting import render_episode_plan_summary


BatchEpisodeStatus = Literal["accepted", "needs_review", "rejected", "failed"]


class RejectedEpisodeDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    stage: str
    error_code: str
    message: str
    details: dict[str, str | int | float] = Field(default_factory=dict)
    suggestion: str


class BatchEpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    status: BatchEpisodeStatus
    plan_path: str
    dataset_path: str | None = None
    frame_count: int = Field(default=0, ge=0)
    video_count: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rejection_diagnostics: list[RejectedEpisodeDiagnostic] = Field(default_factory=list)


class BatchLeRobotV21Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str
    work_path: str
    checkpoint_path: str
    report_path: str
    complete: bool = False
    processing_complete: bool = False
    discovered_episode_count: int = Field(ge=0)
    accepted_episode_count: int = Field(ge=0)
    needs_review_episode_count: int = Field(ge=0)
    rejected_episode_count: int = Field(ge=0)
    failed_episode_count: int = Field(ge=0)
    frame_count: int = Field(default=0, ge=0)
    video_count: int = Field(default=0, ge=0)
    accepted_episode_ids: list[str] = Field(default_factory=list)
    needs_review_episode_ids: list[str] = Field(default_factory=list)
    rejected_episode_ids: list[str] = Field(default_factory=list)
    failed_episode_ids: list[str] = Field(default_factory=list)
    rejected_episode_reports: list[RejectedEpisodeDiagnostic] = Field(
        default_factory=list
    )


class _BatchCheckpoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int = 2
    source_path: str
    output_path: str
    task_text: str
    robot_type: str
    motion_speed_threshold: float
    idle_min_duration_s: float
    context_s: float
    episode_ids: list[str]
    results: dict[str, BatchEpisodeResult] = Field(default_factory=dict)
    complete: bool = False
    frame_count: int = Field(default=0, ge=0)
    video_count: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_checkpoint_schema(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        migrated["schema_version"] = 2
        return migrated


def batch_clean_lerobot_v21_dataset(
    source: Path,
    output_root: Path,
    *,
    task_text: str,
    confirm_subtasks: bool,
    confirm_derived_action: bool,
    robot_type: str = "s2",
    motion_speed_threshold: float = 0.02,
    idle_min_duration_s: float = 1.5,
    context_s: float = 0.5,
    resume: bool = False,
) -> BatchLeRobotV21Result:
    """Clean, isolate failures, checkpoint, and merge a supported batch."""
    source = source.resolve()
    output_root = output_root.resolve()
    task_text = task_text.strip()
    if not task_text:
        raise ValueError("task_text must contain a natural-language task")
    if not confirm_subtasks:
        raise ValueError("batch cleaning requires dataset-level subtask confirmation")
    if not confirm_derived_action:
        raise ValueError(
            "batch cleaning requires dataset-level derived action confirmation"
        )

    source_type = detect_source_type(source)
    if source_type == SourceType.SELF_COLLECTED:
        adapter = SelfCollectedAdapter(source)
        episode_ids = [path.name for path in adapter.episode_dirs]
    elif source_type == SourceType.LEROBOT_V21:
        episode_ids = LeRobotV21Adapter(source).episode_ids
    else:
        raise ValueError(
            "online labels require a LeRobot v2.1 data payload before batch cleaning"
        )
    work_root = output_root.parent / f".{output_root.name}.batch"
    checkpoint_path = work_root / "batch_checkpoint.json"
    report_path = work_root / "batch_report.json"
    expected = _BatchCheckpoint(
        source_path=str(source),
        output_path=str(output_root),
        task_text=task_text,
        robot_type=robot_type,
        motion_speed_threshold=motion_speed_threshold,
        idle_min_duration_s=idle_min_duration_s,
        context_s=context_s,
        episode_ids=episode_ids,
    )
    checkpoint = _load_or_create_checkpoint(
        expected,
        work_root=work_root,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )

    if checkpoint.complete:
        validation = validate_lerobot_v21_dataset(output_root)
        if validation.valid:
            result = _batch_result(checkpoint, work_root, checkpoint_path, report_path)
            _write_batch_report(report_path, result, checkpoint)
            return result
        raise ValueError(
            "checkpoint is complete but output dataset is invalid: "
            + "; ".join(validation.errors)
        )
    plans_root = work_root / "plans"
    episodes_root = work_root / "episodes"
    plans_root.mkdir(parents=True, exist_ok=True)
    episodes_root.mkdir(parents=True, exist_ok=True)

    for episode_id in episode_ids:
        previous = checkpoint.results.get(episode_id)
        should_process = previous is None or previous.status == "failed"
        if not should_process:
            continue
        episode_result = _clean_one_episode(
            source,
            episode_id=episode_id,
            plans_root=plans_root,
            episodes_root=episodes_root,
            task_text=task_text,
            robot_type=robot_type,
            motion_speed_threshold=motion_speed_threshold,
            idle_min_duration_s=idle_min_duration_s,
            context_s=context_s,
        )
        checkpoint.results[episode_id] = episode_result
        _write_checkpoint(checkpoint_path, checkpoint)

    failed_ids = _episode_ids_with_status(checkpoint, "failed")
    accepted_paths = [
        Path(checkpoint.results[episode_id].dataset_path or "")
        for episode_id in episode_ids
        if checkpoint.results[episode_id].status == "accepted"
    ]
    if not accepted_paths:
        result = _batch_result(checkpoint, work_root, checkpoint_path, report_path)
        _write_batch_report(report_path, result, checkpoint)
        return result

    if not failed_ids:
        if not _output_root_has_files(output_root):
            merge_lerobot_v21_datasets(
                accepted_paths,
                output_root,
                task_text=task_text,
            )
        validation = validate_lerobot_v21_dataset(output_root)
        if not validation.valid:
            raise RuntimeError(
                "merged LeRobot v2.1 dataset is invalid: "
                + "; ".join(validation.errors)
            )
        checkpoint.complete = True
        checkpoint.frame_count = validation.frame_count
        checkpoint.video_count = validation.video_count
        _write_checkpoint(checkpoint_path, checkpoint)

    result = _batch_result(checkpoint, work_root, checkpoint_path, report_path)
    _write_batch_report(report_path, result, checkpoint)
    return result


def _clean_one_episode(
    source: Path,
    *,
    episode_id: str,
    plans_root: Path,
    episodes_root: Path,
    task_text: str,
    robot_type: str,
    motion_speed_threshold: float,
    idle_min_duration_s: float,
    context_s: float,
) -> BatchEpisodeResult:
    plan_dir = plans_root / episode_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_stem = "episode_plan"
    candidate_path = plan_dir / f"{plan_stem}.json"
    finalized_path = plan_dir / f"{plan_stem}.accepted.json"
    dataset_path = episodes_root / episode_id
    try:
        if _output_root_has_files(dataset_path):
            validation = validate_lerobot_v21_dataset(dataset_path)
            if not validation.valid:
                raise ValueError(
                    "checkpoint episode output is invalid: "
                    + "; ".join(validation.errors)
                )
            if not finalized_path.is_file():
                raise ValueError(
                    "validated episode output has no finalized audit plan: "
                    f"{finalized_path}"
                )
            return BatchEpisodeResult(
                episode_id=episode_id,
                status="accepted",
                plan_path=str(finalized_path),
                dataset_path=str(dataset_path),
                frame_count=validation.frame_count,
                video_count=validation.video_count,
            )

        candidate = plan_episode_cleaning(
            source,
            episode_id=episode_id,
            motion_speed_threshold=motion_speed_threshold,
            idle_min_duration_s=idle_min_duration_s,
            context_s=context_s,
        )
        _write_plan(candidate_path, candidate)
        if candidate.quality.status == QualityStatus.REJECTED:
            return _quality_result(candidate, candidate_path)

        finalized = finalize_episode_plan(
            source,
            candidate,
            task_text=task_text,
            confirm_subtasks=True,
            confirm_derived_action=True,
        )
        if finalized.quality.status != QualityStatus.ACCEPTED:
            finalized_path = plan_dir / f"{plan_stem}.reviewed.json"
        _write_plan(finalized_path, finalized)
        if finalized.quality.status != QualityStatus.ACCEPTED:
            return _quality_result(finalized, finalized_path)

        materialized = materialize_finalized_lerobot_v21_episode(
            source,
            finalized,
            dataset_path,
            robot_type=robot_type,
        )
        return BatchEpisodeResult(
            episode_id=episode_id,
            status="accepted",
            plan_path=str(finalized_path),
            dataset_path=materialized.root,
            frame_count=materialized.frame_count,
            video_count=materialized.video_count,
            warnings=finalized.quality.warnings,
        )
    except Exception as exc:
        return BatchEpisodeResult(
            episode_id=episode_id,
            status="failed",
            plan_path=str(candidate_path),
            errors=[str(exc)],
        )


def _quality_result(plan: EpisodePlan, plan_path: Path) -> BatchEpisodeResult:
    status: BatchEpisodeStatus
    if plan.quality.status == QualityStatus.REJECTED:
        status = "rejected"
    else:
        status = "needs_review"
    return BatchEpisodeResult(
        episode_id=plan.episode_id,
        status=status,
        plan_path=str(plan_path),
        errors=plan.quality.errors,
        warnings=plan.quality.warnings,
        rejection_diagnostics=(
            [
                _diagnose_rejection(plan.episode_id, error)
                for error in plan.quality.errors
            ]
            if status == "rejected"
            else []
        ),
    )


def _load_or_create_checkpoint(
    expected: _BatchCheckpoint,
    *,
    work_root: Path,
    checkpoint_path: Path,
    resume: bool,
) -> _BatchCheckpoint:
    if checkpoint_path.exists():
        if not resume:
            raise ValueError(
                f"batch checkpoint already exists; retry with resume=true: "
                f"{checkpoint_path}"
            )
        checkpoint = _BatchCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        _validate_checkpoint_config(checkpoint, expected)
        return checkpoint
    if resume:
        raise ValueError(f"batch checkpoint does not exist: {checkpoint_path}")
    if _output_root_has_files(Path(expected.output_path)):
        raise ValueError("output_root must not contain existing files")
    if work_root.exists() and any(work_root.iterdir()):
        raise ValueError(f"batch work directory is not empty: {work_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(checkpoint_path, expected)
    return expected


def _validate_checkpoint_config(
    checkpoint: _BatchCheckpoint,
    expected: _BatchCheckpoint,
) -> None:
    fields = (
        "source_path",
        "output_path",
        "task_text",
        "robot_type",
        "motion_speed_threshold",
        "idle_min_duration_s",
        "context_s",
        "episode_ids",
    )
    mismatches = [
        field
        for field in fields
        if getattr(checkpoint, field) != getattr(expected, field)
    ]
    if mismatches:
        raise ValueError(
            "resume configuration does not match checkpoint: " + ", ".join(mismatches)
        )


def _output_root_has_files(output_root: Path) -> bool:
    return output_root.exists() and (
        not output_root.is_dir() or any(output_root.iterdir())
    )


def _batch_result(
    checkpoint: _BatchCheckpoint,
    work_root: Path,
    checkpoint_path: Path,
    report_path: Path,
) -> BatchLeRobotV21Result:
    accepted = _episode_ids_with_status(checkpoint, "accepted")
    needs_review = _episode_ids_with_status(checkpoint, "needs_review")
    rejected = _episode_ids_with_status(checkpoint, "rejected")
    failed = _episode_ids_with_status(checkpoint, "failed")
    return BatchLeRobotV21Result(
        output_path=checkpoint.output_path,
        work_path=str(work_root),
        checkpoint_path=str(checkpoint_path),
        report_path=str(report_path),
        complete=checkpoint.complete,
        processing_complete=(
            len(checkpoint.results) == len(checkpoint.episode_ids) and not failed
        ),
        discovered_episode_count=len(checkpoint.episode_ids),
        accepted_episode_count=len(accepted),
        needs_review_episode_count=len(needs_review),
        rejected_episode_count=len(rejected),
        failed_episode_count=len(failed),
        frame_count=checkpoint.frame_count,
        video_count=checkpoint.video_count,
        accepted_episode_ids=accepted,
        needs_review_episode_ids=needs_review,
        rejected_episode_ids=rejected,
        failed_episode_ids=failed,
        rejected_episode_reports=[
            diagnostic
            for episode_id in rejected
            for diagnostic in checkpoint.results[episode_id].rejection_diagnostics
        ],
    )


_CAMERA_LEADS_RE = re.compile(
    r"camera frame (?P<frame>\d+) precedes state coverage by "
    r"(?P<gap>[0-9.]+)s \(limit (?P<limit>[0-9.]+)s\)"
)
_CAMERA_LAGS_RE = re.compile(
    r"camera frame (?P<frame>\d+) exceeds state coverage by "
    r"(?P<gap>[0-9.]+)s \(limit (?P<limit>[0-9.]+)s\)"
)
_INTERNAL_STATE_GAP_RE = re.compile(
    r"camera frame (?P<frame>\d+) falls inside a (?P<gap>[0-9.]+)s "
    r"state gap \(limit (?P<limit>[0-9.]+)s\)"
)
_SECONDARY_STREAM_GAP_RE = re.compile(
    r"primary camera frame (?P<frame>\d+) differs from nearest "
    r"(?P<stream>\S+) sample by (?P<gap>[0-9.]+)s "
    r"\(limit (?P<limit>[0-9.]+)s; direction=(?P<direction>[a-z_]+)\)"
)


def _diagnose_rejection(
    episode_id: str,
    message: str,
) -> RejectedEpisodeDiagnostic:
    secondary_match = _SECONDARY_STREAM_GAP_RE.search(message)
    if secondary_match is not None:
        return RejectedEpisodeDiagnostic(
            episode_id=episode_id,
            stage="timeline_alignment",
            error_code="SECONDARY_STREAM_ALIGNMENT_OVER_LIMIT",
            message=message,
            details={
                "stream": secondary_match.group("stream"),
                "frame_index": int(secondary_match.group("frame")),
                "direction": secondary_match.group("direction"),
                "observed_gap_s": float(secondary_match.group("gap")),
                "allowed_gap_s": float(secondary_match.group("limit")),
            },
            suggestion=(
                "Check the secondary camera clock and its recording boundaries."
            ),
        )
    alignment_patterns = (
        (
            _CAMERA_LEADS_RE,
            "CAMERA_LEADS_STATE_OVER_LIMIT",
            "camera_leads_state",
            "Check whether camera recording starts before the state stream.",
        ),
        (
            _CAMERA_LAGS_RE,
            "CAMERA_LAGS_STATE_OVER_LIMIT",
            "camera_lags_state",
            "Check whether state recording stops before the camera stream.",
        ),
        (
            _INTERNAL_STATE_GAP_RE,
            "INTERNAL_STATE_GAP_OVER_LIMIT",
            "internal_state_gap",
            "Check the state stream for missing or delayed samples.",
        ),
    )
    for pattern, error_code, direction, suggestion in alignment_patterns:
        match = pattern.search(message)
        if match is None:
            continue
        return RejectedEpisodeDiagnostic(
            episode_id=episode_id,
            stage="timeline_alignment",
            error_code=error_code,
            message=message,
            details={
                "stream": "state",
                "frame_index": int(match.group("frame")),
                "direction": direction,
                "observed_gap_s": float(match.group("gap")),
                "allowed_gap_s": float(match.group("limit")),
            },
            suggestion=suggestion,
        )

    error_code, stage, suggestion = _generic_rejection_metadata(message)
    return RejectedEpisodeDiagnostic(
        episode_id=episode_id,
        stage=stage,
        error_code=error_code,
        message=message,
        suggestion=suggestion,
    )


def _generic_rejection_metadata(message: str) -> tuple[str, str, str]:
    lowered = message.lower()
    if "segment" in lowered or "action_config" in lowered:
        return (
            "INVALID_SUBTASK_RANGE",
            "subtask_validation",
            "Review overlapping or out-of-range subtask labels.",
        )
    if "timeline mapping" in lowered:
        return (
            "INVALID_TIMELINE_MAPPING",
            "timeline_validation",
            "Inspect source timestamps and retained-frame mapping.",
        )
    if "parquet" in lowered or "mp4" in lowered or "video stream" in lowered:
        return (
            "INVALID_LEROBOT_PAYLOAD",
            "source_validation",
            "Verify the declared Parquet and MP4 files and their frame counts.",
        )
    if "s2 reference profile" in lowered or "feature names" in lowered:
        return (
            "INVALID_STATE_SCHEMA",
            "feature_validation",
            "Verify the S2 joint, gripper, and feature schema values.",
        )
    return (
        "EPISODE_QUALITY_REJECTED",
        "quality_validation",
        "Inspect the episode plan error and correct the source data.",
    )


def _episode_ids_with_status(
    checkpoint: _BatchCheckpoint,
    status: BatchEpisodeStatus,
) -> list[str]:
    return [
        episode_id
        for episode_id in checkpoint.episode_ids
        if checkpoint.results.get(episode_id) is not None
        and checkpoint.results[episode_id].status == status
    ]


def _write_plan(path: Path, plan: EpisodePlan) -> None:
    _write_json_atomic(path, plan.model_dump(mode="json"))
    summary_path = path.with_suffix(".summary.md")
    summary_path.write_text(render_episode_plan_summary(plan), encoding="utf-8")


def _write_checkpoint(path: Path, checkpoint: _BatchCheckpoint) -> None:
    _write_json_atomic(path, checkpoint.model_dump(mode="json"))


def _write_batch_report(
    path: Path,
    result: BatchLeRobotV21Result,
    checkpoint: _BatchCheckpoint,
) -> None:
    report = {
        **result.model_dump(mode="json"),
        "episodes": {
            episode_id: checkpoint.results[episode_id].model_dump(mode="json")
            for episode_id in checkpoint.episode_ids
            if episode_id in checkpoint.results
        },
    }
    _write_json_atomic(path, report)


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
