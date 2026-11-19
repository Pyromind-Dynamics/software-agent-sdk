"""Verify and load a Project reward in the evaluation worker."""

from __future__ import annotations

import hashlib
import json
import math
import types
from pathlib import Path, PurePosixPath


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_reward(binding: dict, mount: Path = Path("/workspace")):
    reference = PurePosixPath(binding["directory"])
    if not reference.is_absolute() or ".." in reference.parts:
        raise ValueError("invalid frozen Reward Storage path")
    directory = mount / str(reference).lstrip("/")
    manifest_bytes = (directory / "manifest.json").read_bytes()
    if _digest(manifest_bytes) != binding["manifest_sha256"]:
        raise ValueError("frozen Reward manifest digest changed")
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise ValueError("unknown frozen Reward manifest schema")
    expected = {"reward.py", "rubric.json", "task_prompt.md"}
    if manifest["judge_kind"] != "code":
        expected.add("calibration.json")
    members = manifest["files"]
    if not expected <= members.keys() or members.keys() - expected - {
        "studio_entry.py"
    }:
        raise ValueError("invalid frozen Reward members")
    files = {name: (directory / name).read_bytes() for name in members}
    if any(_digest(data) != members[name] for name, data in files.items()):
        raise ValueError("frozen Reward member digest changed")
    rubric = json.loads(files["rubric.json"])
    reward = types.ModuleType("project_reward")
    reward.__file__ = str(directory / "reward.py")
    exec(compile(files["reward.py"], reward.__file__, "exec"), reward.__dict__)

    def parse_record(raw, _context):
        case = reward.parse_record(raw)
        if not isinstance(case, dict):
            raise ValueError("Reward parse_record must return an object")
        return {
            "id": raw.get("id"),
            "reference": {"input": case.get("input", raw.get("input"))},
            "prediction": raw.get(
                "response", raw.get("prediction", raw.get("completion"))
            ),
            "sample": raw,
            **case,
            "_reward_case": case,
        }

    def score_case(case, _spec):
        result = reward.score_case(case["_reward_case"], case.get("prediction"))
        if isinstance(result, dict):
            outcome = dict(result)
            value = outcome.get("overall_score", outcome.get("score"))
        else:
            outcome = {}
            value = result
        if (
            type(value) not in (float, int)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Reward score must be a finite number in [0, 1]")
        outcome["overall_score"] = value
        outcome.setdefault("passed", value == 1)
        return outcome

    adapter = types.SimpleNamespace(
        parse_record=parse_record,
        score_case=score_case,
        script_sha256=members["reward.py"],
    )
    return (
        adapter,
        {"business_rule": rubric["aggregation"]},
        files["task_prompt.md"].decode(),
    )
