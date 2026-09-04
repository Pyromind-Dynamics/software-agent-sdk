import ast
import json
import shutil
from pathlib import Path

import pytest
from openhands_embodied_runtime.lerobot_v21 import validate_lerobot_v21_dataset
from openhands_embodied_runtime.sandbox_runner import _parser, run_full, run_plan


def test_sandbox_runner_plans_batches_validates_and_publishes(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "sandbox-run"
    target = tmp_path / "published_lerobot_v21"

    planned = run_plan(
        self_collected_path,
        run_dir,
        idle_min_duration_s=10,
        runtime_revision="openhands-embodied-runtime==1.29.5",
    )
    completed = run_full(
        self_collected_path,
        run_dir,
        target,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
        runtime_revision="openhands-embodied-runtime==1.29.5",
    )

    assert planned["complete"] is True
    assert planned["alignment_policy"] == {
        "warning_gap_s": 0.1,
        "reject_gap_s": 0.5,
    }
    assert completed["complete"] is True
    assert completed["published"] is True
    assert completed["accepted_episode_count"] == 1
    assert completed["alignment_policy"]["reject_gap_s"] == 0.5
    assert completed["validation"]["valid"] is True
    assert validate_lerobot_v21_dataset(target).valid
    assert not any("plan" in path.name for path in target.rglob("*"))

    resumed = run_full(
        self_collected_path,
        run_dir,
        target,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
        resume=True,
        runtime_revision="openhands-embodied-runtime==1.29.5",
    )
    assert resumed["complete"] is True
    assert resumed["phase"] == "resume"


def test_sandbox_runner_publishes_accepted_subset_and_reports_rejection(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed-source"
    shutil.copytree(self_collected_path, source / "accepted_episode")
    rejected = source / "rejected_episode"
    shutil.copytree(self_collected_path, rejected)
    rows = [
        json.loads(line)
        for line in (rejected / "joints.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["stamp_ns"] += 501_000_000
    (rejected / "joints.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    run_dir = tmp_path / "mixed-run"
    target = tmp_path / "mixed-target"
    run_plan(source, run_dir, idle_min_duration_s=10)

    result = run_full(
        source,
        run_dir,
        target,
        task_text="Pick and place the item on the table",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
    )

    assert result["complete"] is True
    assert result["published"] is True
    assert result["accepted_episode_ids"] == ["accepted_episode"]
    assert result["rejected_episode_ids"] == ["rejected_episode"]
    assert result["rejected_episode_reports"][0]["error_code"] == (
        "CAMERA_LEADS_STATE_OVER_LIMIT"
    )
    assert validate_lerobot_v21_dataset(target).episode_count == 1
    assert not (run_dir / "repair_report.json").exists()


def test_sandbox_runner_cleans_mounted_huggingface_lerobot_source(
    lerobot_v21_online_path: Path,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "hf-sandbox-run"
    target = tmp_path / "hf-published-lerobot-v21"

    planned = run_plan(lerobot_v21_online_path, run_dir, idle_min_duration_s=10)
    completed = run_full(
        lerobot_v21_online_path,
        run_dir,
        target,
        task_text="Pickup items in the supermarket",
        confirm_subtasks=True,
        confirm_derived_action=True,
        idle_min_duration_s=10,
    )

    assert planned["representative_episode_id"] == "648649"
    assert completed["accepted_episode_ids"] == ["648649"]
    assert validate_lerobot_v21_dataset(target).valid
    assert not (target / "annotations.json").exists()


def test_sandbox_runtime_sources_parse_as_python_310() -> None:
    runtime_root = (
        Path(__file__).parents[3]
        / "openhands-embodied-runtime"
        / "openhands_embodied_runtime"
    )
    for source_path in runtime_root.glob("*.py"):
        ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
            feature_version=(3, 10),
        )


def test_full_sandbox_run_requires_plan_artifacts(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires a completed plan phase"):
        run_full(
            self_collected_path,
            tmp_path / "missing-plan-run",
            tmp_path / "target",
            task_text="Pick and place the item on the table",
            confirm_subtasks=True,
            confirm_derived_action=True,
        )


def test_sandbox_runner_has_no_repair_mode() -> None:
    choices = _parser()._option_string_actions["--mode"].choices
    assert choices == ("plan", "full", "resume")
