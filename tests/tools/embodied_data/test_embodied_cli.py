"""Tests for the deterministic embodied-data Skill CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = (
    REPO_ROOT
    / ".agents"
    / "skills"
    / "embodied-data-cleaning"
    / "scripts"
    / "embodied_cli.py"
)


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *arguments],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_embodied_cli_runs_fixed_local_workflow(
    self_collected_path: Path,
    tmp_path: Path,
) -> None:
    inspected = _run_cli("inspect", str(self_collected_path), "--sample-limit", "1")
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout)["source_type"] == "self_collected"

    plan_path = tmp_path / "episode_plan.json"
    planned = _run_cli(
        "plan",
        str(self_collected_path),
        "--output",
        str(plan_path),
        "--idle-min-duration-s",
        "10",
    )
    assert planned.returncode == 0, planned.stderr
    assert json.loads(planned.stdout)["output_path"] == str(plan_path)
    assert plan_path.is_file()

    output_path = tmp_path / "lerobot_v21"
    cleaned = _run_cli(
        "clean",
        str(self_collected_path),
        str(output_path),
        "--task-text",
        "Pick and place the item on the table",
        "--confirm-subtasks",
        "--confirm-derived-action",
        "--idle-min-duration-s",
        "10",
    )
    assert cleaned.returncode == 0, cleaned.stderr
    assert json.loads(cleaned.stdout)["complete"] is True

    validated = _run_cli("validate", str(output_path))
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["valid"] is True
