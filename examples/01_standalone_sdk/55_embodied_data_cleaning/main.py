"""Clean self-collected S2 data and materialize LeRobotDataset v2.1."""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool
from openhands.sdk.context import Skill
from openhands.tools.embodied_data import (
    S2_REFERENCE_JOINT_ORDER,
    BatchCleanLeRobotV21Tool,
    BuildEmbodiedEpisodePlanTool,
    InspectEmbodiedDatasetTool,
    ValidateLeRobotV21Tool,
)


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
source_argument = sample_path.name

repo_root = Path(__file__).resolve().parents[3]
skill_path = repo_root / ".agents" / "skills" / "embodied-data-cleaning" / "SKILL.md"
skill = Skill(
    name="embodied-data-cleaning",
    content=skill_path.read_text(encoding="utf-8"),
    source=str(skill_path),
    trigger=None,
)

llm = LLM(
    usage_id="agent",
    model=os.getenv("LLM_MODEL", "gpt-5.5"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)
agent = Agent(
    llm=llm,
    tools=[
        Tool(name=InspectEmbodiedDatasetTool.name),
        Tool(name=BuildEmbodiedEpisodePlanTool.name),
        Tool(name=BatchCleanLeRobotV21Tool.name),
        Tool(name=ValidateLeRobotV21Tool.name),
    ],
    agent_context=AgentContext(skills=[skill]),
)
conversation = Conversation(agent=agent, workspace=str(workspace))
conversation.send_message(
    f"Clean the self-collected dataset at {source_argument!r}. Inspect it and build "
    "one representative EpisodePlan. Then use batch_clean_le_robot_v21 once with "
    "the confirmed shared task 'Pick and place the item on the table'. The source "
    "subtask ranges and reference next-state action convention are explicitly "
    "confirmed. Validate the merged LeRobotDataset v2.1 output at 'lerobot_v21' "
    "once and report the batch counts and output files."
)
conversation.run()

cost = conversation.conversation_stats.get_combined_metrics().accumulated_cost
print(f"EXAMPLE_COST: {cost}")
