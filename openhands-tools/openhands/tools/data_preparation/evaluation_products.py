"""Submission contract for independent CPU distribution/review/report jobs."""

import ast
import json
import shlex
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluationProductsOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["profile", "review", "report"]
    manifest_path: str = Field(min_length=1)
    extraction_script_path: str | None = None


def prepare_products(action, files, runtime_hash: str) -> dict:
    from openhands.tools.data_preparation.inference import storage_path

    options = action.products
    manifest = json.loads(files.read_text(options.manifest_path))
    script = None
    if options.operation == "profile":
        if not options.extraction_script_path:
            raise ValueError("profile requires extraction_script_path")
        script = files.read_text(options.extraction_script_path)
        if "parse_record" not in {
            node.name
            for node in ast.parse(script).body
            if isinstance(node, ast.FunctionDef)
        }:
            raise ValueError("extraction script must define parse_record")
    elif options.extraction_script_path:
        raise ValueError("review/report do not execute extraction or scoring scripts")
    if options.operation in {"profile", "review"}:
        if not {"train", "validation"} <= manifest.get("datasets", {}).keys():
            raise ValueError("train and validation datasets are required")
    if options.operation in {"review", "report"}:
        source = manifest["source"]
        if source["kind"] not in {"scoring", "import"}:
            raise ValueError("source.kind must be scoring or import")
        if source["kind"] == "scoring":
            for name in ("scoring_run_id", "scoring_spec_sha256"):
                if not source.get(name):
                    raise ValueError(f"scoring source requires {name}")
            if "scoring_manifest" not in manifest:
                raise ValueError("scoring source requires frozen scoring_manifest")
        if options.operation == "review" and "profile" not in manifest:
            raise ValueError("review requires a frozen, completed profile")
    return {
        "kind": "badcase_review"
        if options.operation == "review"
        else "evaluation_products",
        "products_operation": options.operation,
        "products_manifest": manifest,
        "extraction_script": script,
        "input_path": storage_path(action.input_path),
        "dataset_config": {},
        "evaluation_config": {},
        "cpu": action.cpu,
        "memory": action.memory,
        "runtime_sha256": runtime_hash,
    }


def build_products_dsl(definition: dict, output_dir: str, run_id: str) -> str:
    from openhands.tools.data_preparation.inference import storage_path

    output = "/workspace" + storage_path(output_dir)
    command = " ".join(
        map(
            shlex.quote,
            (
                "python3",
                output + "/evaluation_products.py",
                "--run-id",
                run_id,
                "--output-dir",
                output,
            ),
        )
    )
    return (
        f"products = CustomCommandCPUNode(id=1, command={command!r}, "
        f"cpu={definition['cpu']}, memory={definition['memory']})\n"
    )
