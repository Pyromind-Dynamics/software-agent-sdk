#!/usr/bin/env python3
"""Run deterministic embodied-data operations without an Agent tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openhands_embodied_runtime.adapters import inspect_dataset
from openhands_embodied_runtime.batch import batch_clean_lerobot_v21_dataset
from openhands_embodied_runtime.lerobot_v21 import validate_lerobot_v21_dataset
from openhands_embodied_runtime.planning import plan_episode_cleaning
from pydantic import BaseModel


def _add_cleaning_thresholds(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--motion-speed-threshold", type=float, default=0.02)
    parser.add_argument("--idle-min-duration-s", type=float, default=1.5)
    parser.add_argument("--context-s", type=float, default=0.5)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect",
        help="Detect and summarize a supported local dataset",
    )
    inspect_parser.add_argument("source", type=Path)
    inspect_parser.add_argument("--sample-limit", type=int, default=3)

    plan_parser = commands.add_parser(
        "plan",
        help="Write one reversible episode cleaning plan",
    )
    plan_parser.add_argument("source", type=Path)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--episode-id")
    _add_cleaning_thresholds(plan_parser)

    clean_parser = commands.add_parser(
        "clean",
        help="Run or resume one deterministic cleaning batch",
    )
    clean_parser.add_argument("source", type=Path)
    clean_parser.add_argument("output", type=Path)
    clean_parser.add_argument("--task-text", required=True)
    clean_parser.add_argument("--confirm-subtasks", action="store_true")
    clean_parser.add_argument("--confirm-derived-action", action="store_true")
    clean_parser.add_argument(
        "--mode",
        choices=("full", "resume"),
        default="full",
    )
    clean_parser.add_argument("--robot-type", default="s2")
    _add_cleaning_thresholds(clean_parser)

    validate_parser = commands.add_parser(
        "validate",
        help="Validate a local LeRobotDataset v2.1 directory",
    )
    validate_parser.add_argument("source", type=Path)
    return parser


def _print_model(model: BaseModel) -> None:
    print(json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2))


def _run_inspect(args: argparse.Namespace) -> int:
    inspection = inspect_dataset(args.source, sample_limit=args.sample_limit)
    _print_model(inspection)
    return 0


def _run_plan(args: argparse.Namespace) -> int:
    plan = plan_episode_cleaning(
        args.source,
        episode_id=args.episode_id,
        motion_speed_threshold=args.motion_speed_threshold,
        idle_min_duration_s=args.idle_min_duration_s,
        context_s=args.context_s,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "episode_id": plan.episode_id,
                "output_path": str(output),
                "quality_status": plan.quality.status.value,
                "drop_interval_count": len(plan.drop_intervals),
                "retained_frame_count": len(plan.timeline_mapping),
                "errors": plan.quality.errors,
                "warnings": plan.quality.warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    result = batch_clean_lerobot_v21_dataset(
        args.source,
        args.output,
        task_text=args.task_text,
        confirm_subtasks=args.confirm_subtasks,
        confirm_derived_action=args.confirm_derived_action,
        robot_type=args.robot_type,
        motion_speed_threshold=args.motion_speed_threshold,
        idle_min_duration_s=args.idle_min_duration_s,
        context_s=args.context_s,
        resume=args.mode == "resume",
    )
    _print_model(result)
    return 1 if result.failed_episode_count else 0


def _run_validate(args: argparse.Namespace) -> int:
    report = validate_lerobot_v21_dataset(args.source)
    _print_model(report)
    return 0 if report.valid else 1


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _run_inspect(args)
        if args.command == "plan":
            return _run_plan(args)
        if args.command == "clean":
            return _run_clean(args)
        return _run_validate(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {"command": args.command, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
