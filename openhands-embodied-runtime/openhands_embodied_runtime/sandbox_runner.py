"""Fixed runtime entrypoint for embodied-data sandbox jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openhands_embodied_runtime.adapters import inspect_dataset
from openhands_embodied_runtime.batch import batch_clean_lerobot_v21_dataset
from openhands_embodied_runtime.lerobot_v21 import validate_lerobot_v21_dataset
from openhands_embodied_runtime.planning import plan_episode_cleaning
from openhands_embodied_runtime.reporting import render_episode_plan_summary


def run_plan(
    source: Path,
    run_dir: Path,
    *,
    representative_episode_id: str | None = None,
    motion_speed_threshold: float = 0.02,
    idle_min_duration_s: float = 1.5,
    context_s: float = 0.5,
    runtime_revision: str | None = None,
) -> dict[str, Any]:
    """Inspect a mounted source and persist one representative reversible plan."""
    source = source.resolve()
    run_dir = run_dir.resolve()
    _require_non_overlapping_paths(source, run_dir, "source", "run directory")
    run_dir.mkdir(parents=True, exist_ok=True)

    inspection = inspect_dataset(source, sample_limit=3)
    if inspection.episode_count == 0 or not inspection.sampled_episodes:
        raise ValueError("source dataset contains no inspectable episodes")
    episode_id = representative_episode_id or inspection.sampled_episodes[0].episode_id
    plan = plan_episode_cleaning(
        source,
        episode_id=episode_id,
        motion_speed_threshold=motion_speed_threshold,
        idle_min_duration_s=idle_min_duration_s,
        context_s=context_s,
    )

    inspection_path = run_dir / "inspection.json"
    plan_path = run_dir / "representative_plan.json"
    summary_path = run_dir / "representative_plan.summary.md"
    _write_json_atomic(inspection_path, inspection.model_dump(mode="json"))
    _write_json_atomic(plan_path, plan.model_dump(mode="json"))
    summary_path.write_text(render_episode_plan_summary(plan), encoding="utf-8")

    report = {
        "schema_version": 1,
        "phase": "plan",
        "complete": True,
        "ready_for_confirmation": plan.quality.status.value != "rejected",
        "source_path": _display_path(source),
        "inspection_path": _display_path(inspection_path),
        "representative_episode_id": plan.episode_id,
        "representative_plan_path": _display_path(plan_path),
        "representative_summary_path": _display_path(summary_path),
        "episode_count": inspection.episode_count,
        "quality_status": plan.quality.status.value,
        "runtime_revision": runtime_revision,
        "errors": plan.quality.errors,
        "warnings": [*inspection.warnings, *plan.quality.warnings],
    }
    _write_phase_report(run_dir, "plan", report)
    return report


def run_full(
    source: Path,
    run_dir: Path,
    target: Path,
    *,
    task_text: str,
    confirm_subtasks: bool,
    confirm_derived_action: bool,
    robot_type: str = "s2",
    motion_speed_threshold: float = 0.02,
    idle_min_duration_s: float = 1.5,
    context_s: float = 0.5,
    resume: bool = False,
    runtime_revision: str | None = None,
) -> dict[str, Any]:
    """Batch-clean, validate, and publish one mounted Storage dataset."""
    source = source.resolve()
    run_dir = run_dir.resolve()
    target = target.resolve()
    _require_plan_artifacts(run_dir)
    _require_non_overlapping_paths(source, target, "source", "target dataset")
    _require_non_overlapping_paths(run_dir, target, "run directory", "target dataset")

    staging = run_dir / "merged_lerobot_v21"
    result = batch_clean_lerobot_v21_dataset(
        source,
        staging,
        task_text=task_text,
        confirm_subtasks=confirm_subtasks,
        confirm_derived_action=confirm_derived_action,
        robot_type=robot_type,
        motion_speed_threshold=motion_speed_threshold,
        idle_min_duration_s=idle_min_duration_s,
        context_s=context_s,
        resume=resume,
    )

    validation = None
    published = False
    if result.complete:
        validation = validate_lerobot_v21_dataset(staging)
        if not validation.valid:
            raise RuntimeError(
                "sandbox batch output failed final validation: "
                + "; ".join(validation.errors)
            )
        _publish_validated_dataset(staging, target, run_id=run_dir.name)
        published_validation = validate_lerobot_v21_dataset(target)
        if not published_validation.valid:
            raise RuntimeError(
                "published dataset failed final validation: "
                + "; ".join(published_validation.errors)
            )
        if not _same_dataset(staging, target):
            raise RuntimeError(
                "published dataset does not match validated staging data"
            )
        published = True

    batch_report = _portable_batch_result(result.model_dump(mode="json"))
    report = {
        "schema_version": 1,
        "phase": "resume" if resume else "full",
        **batch_report,
        "complete": result.complete and published,
        "source_path": _display_path(source),
        "target_path": _display_path(target),
        "staging_path": _display_path(staging),
        "published": published,
        "task_text": task_text,
        "action_provenance": "derived/next_state",
        "runtime_revision": runtime_revision,
        "validation": (
            validation.model_dump(mode="json") if validation is not None else None
        ),
    }
    _write_phase_report(run_dir, "resume" if resume else "full", report)
    return report


def _require_plan_artifacts(run_dir: Path) -> None:
    required = (
        run_dir / "inspection.json",
        run_dir / "representative_plan.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "sandbox full run requires a completed plan phase; missing: "
            + ", ".join(missing)
        )


def _require_non_overlapping_paths(
    left: Path,
    right: Path,
    left_name: str,
    right_name: str,
) -> None:
    if left == right or left in right.parents or right in left.parents:
        raise ValueError(f"{left_name} and {right_name} paths must not overlap")


def _portable_batch_result(result: dict[str, Any]) -> dict[str, Any]:
    for name in ("output_path", "work_path", "checkpoint_path", "report_path"):
        value = result.get(name)
        if isinstance(value, str):
            result[name] = _display_path(Path(value))
    return result


def _display_path(path: Path) -> str:
    text = path.as_posix()
    prefix = "/target-workspace"
    if text == prefix:
        return "/"
    if text.startswith(f"{prefix}/"):
        return text.removeprefix(prefix)
    return text


def _publish_validated_dataset(source: Path, target: Path, *, run_id: str) -> None:
    if target.exists():
        validation = validate_lerobot_v21_dataset(target)
        if validation.valid and _same_dataset(source, target):
            return
        raise ValueError(f"target path already exists and is not reusable: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.publish-{run_id}")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    validation = validate_lerobot_v21_dataset(temporary)
    if not validation.valid:
        raise RuntimeError(
            "temporary published dataset is invalid: " + "; ".join(validation.errors)
        )
    temporary.replace(target)


def _same_dataset(left: Path, right: Path) -> bool:
    left_files = _relative_files(left)
    right_files = _relative_files(right)
    if left_files != right_files:
        return False
    return all(
        _file_digest(left / relative) == _file_digest(right / relative)
        for relative in left_files
    )


def _relative_files(root: Path) -> list[Path]:
    return sorted(
        (path.relative_to(root) for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_phase_report(run_dir: Path, phase: str, report: dict[str, Any]) -> None:
    _write_json_atomic(run_dir / f"{phase}_report.json", report)
    _write_json_atomic(run_dir / "report.json", report)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "full", "resume"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--task-text")
    parser.add_argument("--representative-episode-id")
    parser.add_argument("--robot-type", default="s2")
    parser.add_argument("--motion-speed-threshold", type=float, default=0.02)
    parser.add_argument("--idle-min-duration-s", type=float, default=1.5)
    parser.add_argument("--context-s", type=float, default=0.5)
    parser.add_argument("--confirm-subtasks", action="store_true")
    parser.add_argument("--confirm-derived-action", action="store_true")
    parser.add_argument("--runtime-revision")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "plan":
            report = run_plan(
                args.source,
                args.run_dir,
                representative_episode_id=args.representative_episode_id,
                motion_speed_threshold=args.motion_speed_threshold,
                idle_min_duration_s=args.idle_min_duration_s,
                context_s=args.context_s,
                runtime_revision=args.runtime_revision,
            )
        else:
            if args.target is None or args.task_text is None:
                raise ValueError(
                    "full and resume modes require --target and --task-text"
                )
            report = run_full(
                args.source,
                args.run_dir,
                args.target,
                task_text=args.task_text,
                confirm_subtasks=args.confirm_subtasks,
                confirm_derived_action=args.confirm_derived_action,
                robot_type=args.robot_type,
                motion_speed_threshold=args.motion_speed_threshold,
                idle_min_duration_s=args.idle_min_duration_s,
                context_s=args.context_s,
                resume=args.mode == "resume",
                runtime_revision=args.runtime_revision,
            )
    except Exception as exc:
        report = {
            "schema_version": 1,
            "phase": args.mode,
            "complete": False,
            "source_path": _display_path(args.source.resolve()),
            "target_path": (
                _display_path(args.target.resolve())
                if args.target is not None
                else None
            ),
            "runtime_revision": args.runtime_revision,
            "runtime_error": str(exc),
        }
        _write_phase_report(args.run_dir.resolve(), args.mode, report)
        print(json.dumps(report, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {"complete": bool(report.get("complete")), "phase": args.mode},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
