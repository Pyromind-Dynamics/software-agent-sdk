#!/usr/bin/env python3
"""Validate the mandatory command-node Agent Rubric workflow contract."""

from __future__ import annotations

import argparse
import ast
import json
import shlex
from pathlib import Path
from typing import Any


COMMAND_NODES = {"CustomCommandNode", "CustomCommandCPUNode"}
FORBIDDEN_NODES = {
    "MetricsConfigBuilderCustomNode",
    "MetricsConfigBuilderNode",
    "ModelEvalApiNode",
    "VLLMInference",
}
RUBRIC_EVALUATORS = {
    "bbox_iou",
    "exact_match",
    "field_equals",
    "json_valid",
    "list_count_match",
    "non_empty",
    "number_range",
    "required_fields",
}
COMMON_FLAGS = {
    "--dataset-config",
    "--dataset-path",
    "--evaluation-config",
    "--limit",
    "--output-dir",
}
STORAGE_LOGICAL_PREFIX = "/.pyromind-agent/"


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _string_value(expression: ast.expr | None) -> str | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left = _string_value(expression.left)
        right = _string_value(expression.right)
        if left is not None and right is not None:
            return left + right
    return None


def _flag_value(tokens: list[str], flag: str) -> str | None:
    if flag not in tokens:
        return None
    index = tokens.index(flag)
    return tokens[index + 1] if index + 1 < len(tokens) else ""


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not a readable JSON file: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _validate_dataset_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    reference_field = config.get("reference_field")
    if not isinstance(reference_field, str) or not reference_field.strip():
        errors.append("dataset config requires a non-empty reference_field")
    if not config.get("request_builder_entry") and not any(
        isinstance(config.get(name), str) and config[name].strip()
        for name in ("messages_field", "user_prompt_field")
    ):
        errors.append("dataset config requires messages_field or user_prompt_field")
    aliases = {
        "image_field": "media_field",
        "image_root": "media_base_dir",
    }
    for alias, canonical in aliases.items():
        if alias in config:
            errors.append(
                f"dataset config uses legacy {alias}; use {canonical} instead"
            )
    for name in ("media_field", "media_base_dir"):
        value = config.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"dataset config {name} must be a non-empty string")
    image_order = config.get("image_order")
    if image_order is not None and (
        not isinstance(image_order, list)
        or not image_order
        or not all(isinstance(item, str) and item for item in image_order)
    ):
        errors.append("dataset config image_order must contain file names")
    return errors


def _validate_rubric_evaluator(prefix: str, evaluator: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    kind = evaluator.get("type")
    if kind not in RUBRIC_EVALUATORS:
        return [f"{prefix} has an unsupported evaluator type"]
    paths: tuple[str, ...] = ()
    if kind in {"field_equals", "list_count_match", "bbox_iou"}:
        paths = ("prediction_path", "reference_path")
    elif kind in {"non_empty", "number_range"}:
        paths = ("prediction_path",)
    for name in paths:
        if not isinstance(evaluator.get(name), str):
            errors.append(f"{prefix} evaluator requires {name}")
    if kind == "required_fields":
        configured = evaluator.get("prediction_paths")
        if (
            not isinstance(configured, list)
            or not configured
            or not all(isinstance(item, str) and item for item in configured)
        ):
            errors.append(
                f"{prefix} required_fields requires non-empty prediction_paths"
            )
    if kind == "number_range":
        for name in ("min", "max"):
            value = evaluator.get(name)
            if value is not None and not isinstance(value, (int, float)):
                errors.append(f"{prefix} number_range {name} must be numeric")
    if kind == "bbox_iou":
        threshold = evaluator.get("iou_threshold", 0.5)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            errors.append(f"{prefix} bbox_iou iou_threshold must be between 0 and 1")
    return errors


def _validate_evaluation_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("mode") != "agent_rubric":
        errors.append("evaluation config mode must be agent_rubric")
    if any(key.startswith("planner_") for key in config):
        errors.append("evaluation config must not contain runtime planner settings")
    for name in ("pass_threshold", "rubric_pass_threshold"):
        value = config.get(name, 0.7)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            errors.append(f"evaluation config {name} must be between 0 and 1")
    rubrics = config.get("rubrics")
    if not isinstance(rubrics, list) or not 1 <= len(rubrics) <= 20:
        errors.append("evaluation config requires between 1 and 20 rubrics")
        return errors
    names: set[str] = set()
    for index, rubric in enumerate(rubrics, 1):
        prefix = f"rubric {index}"
        if not isinstance(rubric, dict):
            errors.append(f"{prefix} must be an object")
            continue
        name = rubric.get("name")
        criterion = rubric.get("criterion")
        weight = rubric.get("weight")
        evaluator = rubric.get("evaluator")
        required = rubric.get("required", False)
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{prefix} requires a name")
        elif name in names:
            errors.append(f"duplicate rubric name: {name}")
        else:
            names.add(name)
        if not isinstance(criterion, str) or not criterion.strip():
            errors.append(f"{prefix} requires a criterion")
        if not isinstance(weight, (int, float)) or weight <= 0:
            errors.append(f"{prefix} requires a positive weight")
        if not isinstance(required, bool):
            errors.append(f"{prefix} required must be boolean")
        if not isinstance(evaluator, dict):
            errors.append(f"{prefix} requires an evaluator")
        else:
            errors.extend(_validate_rubric_evaluator(prefix, evaluator))
    return errors


def validate_configs(
    dataset_config_path: Path,
    evaluation_config_path: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        errors.extend(
            _validate_dataset_config(
                _json_object(dataset_config_path, "dataset config")
            )
        )
    except ValueError as exc:
        errors.append(str(exc))
    try:
        errors.extend(
            _validate_evaluation_config(
                _json_object(evaluation_config_path, "evaluation config")
            )
        )
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def validate_pipeline(
    workflow_path: Path,
    dataset_config_path: Path,
    evaluation_config_path: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(workflow_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return [f"workflow is not readable Python: {exc}"]

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    calls_by_name: dict[str, list[ast.Call]] = {}
    for call in calls:
        calls_by_name.setdefault(_call_name(call), []).append(call)

    used_forbidden = sorted(FORBIDDEN_NODES.intersection(calls_by_name))
    if used_forbidden:
        errors.append(
            "forbidden benchmark nodes are present: " + ", ".join(used_forbidden)
        )

    command_calls = [
        call for name in COMMAND_NODES for call in calls_by_name.get(name, [])
    ]
    if len(command_calls) != 1:
        errors.append(
            "workflow must contain exactly one CustomCommandNode or "
            "CustomCommandCPUNode"
        )
    else:
        call = command_calls[0]
        node_name = _call_name(call)
        command = _string_value(_keyword(call, "command"))
        if command is None:
            errors.append("command node command must be a static string literal")
        else:
            try:
                tokens = shlex.split(command)
            except ValueError as exc:
                errors.append(f"command is not valid shell syntax: {exc}")
                tokens = []
            if any(token.startswith(STORAGE_LOGICAL_PREFIX) for token in tokens):
                errors.append(
                    "command must map /.pyromind-agent/... Storage paths to "
                    "/workspace/.pyromind-agent/... container paths"
                )
            if not any(token.endswith("/evaluate_inference.py") for token in tokens):
                errors.append("command must execute the staged evaluate_inference.py")
            missing_flags = sorted(flag for flag in COMMON_FLAGS if flag not in tokens)
            if missing_flags:
                errors.append("command is missing flags: " + ", ".join(missing_flags))
            for resource in ("cpu", "memory"):
                if _keyword(call, resource) is None:
                    errors.append(f"{node_name} requires {resource}")
            if node_name == "CustomCommandNode":
                if "--model-path" not in tokens or "--gpu-count" not in tokens:
                    errors.append(
                        "CustomCommandNode requires --model-path and --gpu-count"
                    )
                if "--endpoint" in tokens:
                    errors.append("CustomCommandNode must not use --endpoint")
                for resource in ("gpu_count", "gpu_product"):
                    if _keyword(call, resource) is None:
                        errors.append(f"CustomCommandNode requires {resource}")
                node_gpu_count = _keyword(call, "gpu_count")
                command_gpu_count = _flag_value(tokens, "--gpu-count")
                if (
                    isinstance(node_gpu_count, ast.Constant)
                    and isinstance(node_gpu_count.value, int)
                    and command_gpu_count
                    and command_gpu_count != str(node_gpu_count.value)
                ):
                    errors.append("CustomCommandNode gpu_count must match --gpu-count")
            else:
                if "--endpoint" not in tokens:
                    errors.append("CustomCommandCPUNode requires --endpoint")
                if "--model-path" in tokens or "--gpu-count" in tokens:
                    errors.append(
                        "CustomCommandCPUNode must not use --model-path or --gpu-count"
                    )

    errors.extend(validate_configs(dataset_config_path, evaluation_config_path))

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs-only", action="store_true")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    expected = 2 if args.configs_only else 3
    if len(args.paths) != expected:
        parser.error(f"expected {expected} paths")
    paths = [path.resolve() for path in args.paths]
    errors = (
        validate_configs(paths[0], paths[1])
        if args.configs_only
        else validate_pipeline(paths[0], paths[1], paths[2])
    )
    if errors:
        raise SystemExit(
            "Invalid inference evaluation pipeline:\n- " + "\n- ".join(errors)
        )
    print("Inference evaluation pipeline contract is valid.")


if __name__ == "__main__":
    main()
