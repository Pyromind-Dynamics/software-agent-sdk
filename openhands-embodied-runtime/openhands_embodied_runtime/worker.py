"""Per-episode worker entrypoints for the platform EDP pipeline.

The platform runner executes one manifest record per sandbox. Each record
names one episode; the worker cleans that episode into the shared mounted
work root and reports its outcome through the same reward-file channel tmax
uses, so no orchestrator (runner) changes are required. Merge mode runs as a
separate edp_aggregate-style step over the shared work root.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openhands_embodied_runtime.batch import (
    BatchEpisodeResult,
    clean_one_episode,
)
from openhands_embodied_runtime.lerobot_v21 import (
    merge_lerobot_v21_datasets,
    validate_lerobot_v21_dataset,
)


EPISODES_ROOT = "episodes"
PLANS_ROOT = "plans"
RESULTS_ROOT = "results"
MERGED_ROOT = "merged_lerobot_v21"

_REWARD_BY_STATUS = {
    "accepted": 1.0,
    "needs_review": 0.5,
    "rejected": 0.0,
}
_REWARD_FILE_NAME = "reward.txt"
_RESULT_FILE_NAME = "result.json"


def run_episode(
    source: Path,
    work_root: Path,
    *,
    episode_id: str,
    task_text: str,
    robot_type: str = "s2",
    motion_speed_threshold: float = 0.02,
    idle_min_duration_s: float = 1.5,
    context_s: float = 0.5,
    reward_path: Path | None = None,
) -> dict[str, Any]:
    """Clean exactly one episode; idempotent across sandbox retries."""
    source = source.resolve()
    work_root = work_root.resolve()
    plans_root = work_root / PLANS_ROOT
    episodes_root = work_root / EPISODES_ROOT
    results_root = work_root / RESULTS_ROOT
    results_root.mkdir(parents=True, exist_ok=True)

    try:
        result = clean_one_episode(
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
    except Exception as exc:
        result = BatchEpisodeResult(
            episode_id=episode_id,
            status="failed",
            plan_path=str(plans_root / episode_id),
            errors=[f"{type(exc).__name__}: {exc}"],
        )
    result_path = results_root / f"{episode_id}.json"
    result_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    reward_file = reward_path or Path("/logs/verifier") / _REWARD_FILE_NAME
    reward = _REWARD_BY_STATUS.get(result.status)
    if reward is not None:
        try:
            reward_file.parent.mkdir(parents=True, exist_ok=True)
            reward_file.write_text(f"{reward}\n", encoding="utf-8")
        except OSError:
            # /logs may be absent outside the platform runner; the runner
            # falls back to the worker exit code, so classification survives.
            pass
    return {
        "phase": "episode",
        **result.model_dump(mode="json"),
    }


def run_merge(
    work_root: Path,
    target: Path,
    *,
    task_text: str,
    expected_episode_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Merge per-episode outputs and validate the published LeRobot dataset."""
    work_root = work_root.resolve()
    target = target.resolve()
    results_root = work_root / RESULTS_ROOT
    if not results_root.is_dir():
        raise ValueError(
            f"no episode results under {work_root}: run episode workers first"
        )
    results: dict[str, BatchEpisodeResult] = {}
    for path in sorted(results_root.glob("*.json")):
        if path.name == _RESULT_FILE_NAME:
            continue
        stored = BatchEpisodeResult.model_validate(json.loads(path.read_text()))
        if stored.status != "failed":
            results[stored.episode_id] = stored
    if expected_episode_ids:
        missing = sorted(set(expected_episode_ids) - set(results))
        if missing:
            raise RuntimeError("episode results missing for: " + ", ".join(missing))
    statuses = [result.status for result in results.values()]
    accepted = [result for result in results.values() if result.status == "accepted"]
    if not accepted:
        report = {
            "phase": "merge",
            "complete": statuses.count("failed") == 0,
            "processing_complete": statuses.count("failed") == 0,
            "published": False,
            "all_rejected": True,
            "episode_count": len(results),
            "accepted_episode_count": 0,
            "needs_review_episode_count": statuses.count("needs_review"),
            "rejected_episode_count": statuses.count("rejected"),
            "target_path": str(target),
        }
        json_body = json.dumps(report)
        print(json_body)
        return report
    merged_root = work_root / MERGED_ROOT
    if not merged_root.exists() or not any(merged_root.iterdir()):
        merge_lerobot_v21_datasets(
            [Path(result.dataset_path or "") for result in accepted],
            merged_root,
            task_text=task_text,
        )
    validation = validate_lerobot_v21_dataset(merged_root)
    if not validation.valid:
        raise RuntimeError(
            "merged LeRobot v2.1 dataset is invalid: " + "; ".join(validation.errors)
        )
    if target.resolve() != merged_root:
        if target.exists():
            raise RuntimeError(
                f"refusing to overwrite non-empty publish target {target}"
            )
        merged_root.rename(target)
    published_validation = validate_lerobot_v21_dataset(target)
    if not published_validation.valid:
        raise RuntimeError(
            "published dataset failed final validation: "
            + "; ".join(published_validation.errors)
        )
    report = {
        "phase": "merge",
        "complete": True,
        "processing_complete": True,
        "published": True,
        "all_rejected": False,
        "episode_count": len(results),
        "accepted_episode_count": statuses.count("accepted"),
        "needs_review_episode_count": statuses.count("needs_review"),
        "rejected_episode_count": statuses.count("rejected"),
        "frame_count": published_validation.frame_count,
        "video_count": published_validation.video_count,
        "source_root": str(work_root),
        "target_path": str(target),
    }
    json_body = json.dumps(report)
    print(json_body)
    return report
