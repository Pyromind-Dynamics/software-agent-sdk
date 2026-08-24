"""Resumable deterministic batch cleaning for self-collected S2 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from openhands.tools.embodied_data.adapters import SelfCollectedAdapter
from openhands.tools.embodied_data.lerobot_v21 import (
    materialize_finalized_lerobot_v21_episode,
    merge_lerobot_v21_datasets,
    validate_lerobot_v21_dataset,
)
from openhands.tools.embodied_data.models import EpisodePlan, QualityStatus
from openhands.tools.embodied_data.planning import (
    finalize_episode_plan,
    plan_episode_cleaning,
)
from openhands.tools.embodied_data.reporting import render_episode_plan_summary


BatchEpisodeStatus = Literal["accepted", "needs_review", "rejected", "failed"]


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


class BatchLeRobotV21Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str
    work_path: str
    checkpoint_path: str
    report_path: str
    complete: bool = False
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


class _BatchCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
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
    """Clean, isolate failures, checkpoint, and merge a self-collected batch."""
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

    adapter = SelfCollectedAdapter(source)
    episode_ids = [path.name for path in adapter.episode_dirs]
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
        if previous is not None and previous.status != "failed":
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
    candidate_path = plan_dir / "episode_plan.json"
    finalized_path = plan_dir / "episode_plan.accepted.json"
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
            finalized_path = plan_dir / "episode_plan.reviewed.json"
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
