"""Platform packaging for CPU-only evaluation of existing predictions."""

from __future__ import annotations

import ast
import json
import re
import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionEvaluationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_manifest_path: str = Field(min_length=1)
    scoring_spec_path: str = Field(min_length=1)
    scoring_script_path: str = Field(min_length=1)


def prepare_definition(
    options: PredictionEvaluationOptions,
    files: Any,
    runtime_hash: str,
    *,
    input_path: str,
    cpu: int,
    memory: int,
) -> dict[str, Any]:
    from openhands.tools.data_preparation.inference import storage_path

    manifest = json.loads(files.read_text(options.source_manifest_path))
    spec = json.loads(files.read_text(options.scoring_spec_path))
    script = files.read_text(options.scoring_script_path)
    if not isinstance(manifest, dict) or not isinstance(spec, dict):
        raise ValueError("manifest and scoring spec must be JSON objects")
    if (
        not isinstance(spec.get("business_rule"), str)
        or not spec["business_rule"].strip()
    ):
        raise ValueError("scoring_spec.business_rule must state the confirmed rule")
    if (
        not isinstance(spec.get("validation_examples"), list)
        or not spec["validation_examples"]
    ):
        raise ValueError("scoring_spec.validation_examples is required")
    functions = {
        node.name
        for node in ast.parse(script).body
        if isinstance(node, ast.FunctionDef)
    }
    if not {"parse_record", "score_case"} <= functions:
        raise ValueError("script must define parse_record and score_case")
    entries = manifest.get("evaluations", [])
    if not entries or not isinstance(entries, list):
        raise ValueError("source manifest requires evaluations")
    steps = set()
    for entry in entries:
        step = entry.get("step")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
            or step in steps
        ):
            raise ValueError("evaluation steps must be unique nonnegative integers")
        steps.add(step)
        entry["path"] = storage_path(entry["path"])
        if (
            manifest.get("schema_version", 1) == 1
            and Path(entry["path"]).name != f"eval_step_{step}.jsonl"
        ):
            # Explicit baseline entries are independently identifiable sources.
            if entry.get("role") != "baseline":
                raise ValueError("evaluation filename must match its step")
        if entry.get("dataset_role", "validation") not in {
            "validation",
            "test",
            "train",
        }:
            raise ValueError("unknown evaluation dataset role")
        if re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) is None:
            raise ValueError("each source requires its SHA-256 fingerprint")
    for checkpoint in manifest.get("checkpoints", []):
        checkpoint["path"] = storage_path(checkpoint["path"])
        if Path(checkpoint["path"]).name != f"checkpoint-{checkpoint['step']}":
            raise ValueError("checkpoint directory must match its step")
    for entry in manifest.get("datasets", {}).values():
        entry["path"] = storage_path(entry["path"])
        if re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) is None:
            raise ValueError("datasets require SHA-256 fingerprints")
    return {
        "kind": "prediction_evaluation",
        "source_manifest": manifest,
        "scoring_spec": spec,
        "scoring_script": script,
        "input_path": storage_path(input_path),
        "dataset_config": {},
        "evaluation_config": {},
        "cpu": cpu,
        "memory": memory,
        "runtime_sha256": runtime_hash,
    }


def build_prediction_evaluation_dsl(
    definition: dict, output_dir: str, run_id: str
) -> str:
    from openhands.tools.data_preparation.inference import storage_path

    output = "/workspace" + storage_path(output_dir)
    command = " ".join(
        shlex.quote(value)
        for value in (
            "python3",
            output + "/evaluate_predictions.py",
            "--run-id",
            run_id,
            "--output-dir",
            output,
        )
    )
    return (
        f"evaluation = CustomCommandCPUNode(id=1, command={command!r}, "
        f"cpu={definition['cpu']}, memory={definition['memory']})\n"
    )
