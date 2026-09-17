"""Clean self-collected S2 data and materialize LeRobotDataset v2.1."""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from openhands_embodied_runtime.adapters import S2_REFERENCE_JOINT_ORDER


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", nargs="?", type=Path)
args = parser.parse_args()

if args.path is None:
    sample_root = Path(tempfile.mkdtemp(prefix="embodied_data_example_"))
    sample_path = sample_root / "episode_000000"
    sample_path.mkdir()
    video_writer = imageio_ffmpeg.write_frames(
        str(sample_path / "head.mp4"),
        (16, 16),
        fps=30,
        codec="mpeg4",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
    )
    video_writer.send(None)
    for frame_index in range(12):
        frame = np.full((16, 16, 3), frame_index * 10, dtype=np.uint8)
        video_writer.send(frame.tobytes())
    video_writer.close()
    with (sample_path / "head.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["frame_index", "wall_time", "stamp_ns"],
        )
        writer.writeheader()
        for frame_index in range(12):
            writer.writerow(
                {
                    "frame_index": frame_index,
                    "wall_time": 100 + frame_index / 10,
                    "stamp_ns": 100_000_000_000 + frame_index * 100_000_000,
                }
            )
    joint_rows = [
        {
            "wall_time": 100 + frame_index / 10,
            "stamp_ns": 100_000_000_000 + frame_index * 100_000_000,
            "joints": {name: frame_index / 20 for name in S2_REFERENCE_JOINT_ORDER},
            "left_ee_states": [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0],
            "right_ee_states": [0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
            "suction_state": int(4 <= frame_index < 9),
            "subtask": "move_object",
        }
        for frame_index in range(12)
    ]
    (sample_path / "joints.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in joint_rows),
        encoding="utf-8",
    )
    (sample_path / "manifest.json").write_text(
        json.dumps(
            {
                "task": "Pick and place the item on the table",
                "steps": [
                    {
                        "step_id": "move_object",
                        "type": "pick_and_place",
                        "started_at": 100.0,
                        "completed_at": 101.1,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
else:
    sample_path = args.path.resolve()

workspace = sample_path.parent
repo_root = Path(__file__).resolve().parents[3]
cli_path = (
    repo_root
    / ".agents"
    / "skills"
    / "embodied-data-cleaning"
    / "scripts"
    / "embodied_cli.py"
)
plan_path = workspace / "episode_plan.json"
output_path = workspace / "lerobot_v21"
commands = [
    ["inspect", str(sample_path), "--sample-limit", "1"],
    [
        "plan",
        str(sample_path),
        "--output",
        str(plan_path),
        "--idle-min-duration-s",
        "10",
    ],
    [
        "clean",
        str(sample_path),
        str(output_path),
        "--task-text",
        "Pick and place the item on the table",
        "--confirm-subtasks",
        "--confirm-derived-action",
        "--idle-min-duration-s",
        "10",
    ],
    ["validate", str(output_path)],
]

for arguments in commands:
    completed = subprocess.run(
        [sys.executable, str(cli_path), *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)

print("EXAMPLE_COST: 0")
