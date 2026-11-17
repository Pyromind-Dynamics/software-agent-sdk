"""Validate the local CPU scoring handoff (also runnable in the workspace)."""

import argparse
import json
import math
import re
from pathlib import Path


def _require(condition, field):
    if not condition:
        raise ValueError(f"invalid or missing {field}")


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _path(value):
    return (
        isinstance(value, str)
        and value.startswith("/")
        and ".." not in value.split("/")
        and not any(char in value for char in "?:#\\")
        and value != "/"
    )


def validate_eval_delivery(data):
    _require(isinstance(data, dict), "document")
    _require(
        type(data.get("schema_version")) is int and data["schema_version"] == 1,
        "schema_version",
    )
    _require(
        set(data)
        <= {
            "schema_version",
            "task_id",
            "training_node_id",
            "dataset_role",
            "dataset_sha256",
            "scoring_sha256",
            "report_path",
            "steps",
            "selected_step",
        },
        "document fields",
    )
    for field in ("task_id", "training_node_id"):
        value = data.get(field)
        _require(isinstance(value, str) and bool(value.strip()), field)
    _require(
        data.get("dataset_role") in ("validation", "test", "train"), "dataset_role"
    )
    for field in ("dataset_sha256", "scoring_sha256"):
        _require(_digest(data.get(field)), field)
    _require(_path(data.get("report_path")), "report_path")
    _require(isinstance(data.get("steps"), list) and bool(data["steps"]), "steps")
    seen = set()
    for index, row in enumerate(data["steps"]):
        field = f"steps[{index}]"
        _require(isinstance(row, dict), field)
        _require(
            set(row) == {"step", "checkpoint_path", "prediction_sha256", "metrics"},
            f"{field} fields",
        )
        step = row.get("step")
        _require(type(step) is int and step >= 0 and step not in seen, f"{field}.step")
        seen.add(step)
        _require(_path(row.get("checkpoint_path")), f"{field}.checkpoint_path")
        _require(_digest(row.get("prediction_sha256")), f"{field}.prediction_sha256")
        metrics = row.get("metrics")
        _require(isinstance(metrics, dict) and bool(metrics), f"{field}.metrics")
        for name, value in metrics.items():
            _require(
                bool(name) and type(value) in (int, float) and math.isfinite(value),
                f"{field}.metrics.{name}",
            )
    selected = data.get("selected_step")
    if selected is not None:
        _require(
            data["dataset_role"] == "validation"
            and type(selected) is int
            and selected in seen,
            "selected_step",
        )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    try:
        validate_eval_delivery(json.loads(args.file.read_bytes()))
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(1, f"{exc}\n")
    print("eval_result.json format is valid")


if __name__ == "__main__":
    main()
