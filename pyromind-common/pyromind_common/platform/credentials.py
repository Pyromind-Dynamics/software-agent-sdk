"""Bind execution environment values immediately before a Studio request."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal

from pyromind_common.platform.recipe import (
    RecipeReceipt,
    dataset_gateway_exports,
    dataset_public_profile,
    read_prepared_dataset_recipe,
)


_ENVIRONMENT_NAMES = frozenset(
    {
        "PYROMIND_API_KEY",
        "PYROMIND_BASE_URL",
        "LLM_BASE_URL",
        "LLM_AUTH_TOKEN",
        "LLM_MODEL",
        "DF_API_KEY",
        "DF_API_URL",
        "DF_API_BASE_URL",
        "DF_MODEL_NAME",
    }
)
_BINDING = re.compile(
    r"\{\{pyromind\.runtime:(environment|text|vision):([A-Z_,]+)(?::([a-f0-9]{64}))?\}\}"
)

_DATASET_KEY = re.compile(r"\{\{pyromind\.dataset-key:(text|vision):([a-f0-9]{64})\}\}")


def dataset_key_parameter(
    profile: Literal["text", "vision"], values: Mapping[str, str]
) -> str:
    public = dataset_public_profile(values)
    return (
        "{{pyromind.dataset-key:"
        + profile
        + ":"
        + _public_model_fingerprint(public)
        + "}}"
    )


def _public_model_fingerprint(values: Mapping[str, str]) -> str:
    public = {name: value for name, value in values.items() if name != "DF_API_KEY"}
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


def dataflow_environment(
    profile: Literal["text", "vision"],
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = os.environ if environ is None else environ
    try:
        profiles = json.loads(environment.get("PYROMIND_DATAFLOW_PROFILES", "{}"))
        values = profiles[profile]
        if not isinstance(values, dict) or not all(
            name in _ENVIRONMENT_NAMES and isinstance(value, str) and "\0" not in value
            for name, value in values.items()
        ):
            raise ValueError
        if not all(values.get(name) for name in ("DF_API_URL", "DF_MODEL_NAME")):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            f"The terminal has no usable {profile} model profile"
        ) from None
    return values


def runtime_environment_prefix(
    names: Sequence[str], *, profile: Literal["text", "vision"] | None = None
) -> str:
    selected = tuple(dict.fromkeys(names))
    if not selected or any(name not in _ENVIRONMENT_NAMES for name in selected):
        raise ValueError("Unsupported worker environment names")
    fingerprint = (
        ":" + _public_model_fingerprint(dataflow_environment(profile))
        if profile is not None
        else ""
    )
    return (
        "{{pyromind.runtime:"
        + (profile or "environment")
        + ":"
        + ",".join(selected)
        + fingerprint
        + "}}"
    )


def _bind_dataset_key(
    param: str, recipe: RecipeReceipt, environment: Mapping[str, str]
) -> str:
    match = _DATASET_KEY.fullmatch(param)
    if not match or not recipe.uses_labeling_gateway:
        raise ValueError("Invalid Dataset gateway credential binding")
    profile: Literal["text", "vision"] = "text" if match[1] == "text" else "vision"
    values = dataflow_environment(profile, environment)
    if dataset_key_parameter(profile, values) != param:
        raise ValueError("Model configuration changed after workflow preparation")
    prefix = "\n".join(
        [
            *dataset_gateway_exports(dataset_public_profile(values)),
            'export DF_API_KEY="$param"',
            "unset param",
            "set -eu",
            "",
        ]
    )
    if not recipe.command_template.startswith(
        prefix.replace("{", "{{").replace("}", "}}")
    ):
        raise ValueError("Dataset recipe does not match the selected model profile")
    if not values.get("DF_API_KEY"):
        raise ValueError("Required Dataset gateway credential is unavailable")
    return values["DF_API_KEY"]


def materialize_runtime_environment(
    workflow: dict[str, Any], environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Copy the validated graph; bind only command environment declarations.

    The returned graph is a transport payload, not a persistable artifact.
    Studio receives the same command environment as the existing workers.
    """
    environment = os.environ if environ is None else environ

    def bind(match: re.Match[str]) -> str:
        profile = match[1]
        if profile == "text":
            values = dataflow_environment("text", environment)
        elif profile == "vision":
            values = dataflow_environment("vision", environment)
        else:
            values = environment
        if match[3] and _public_model_fingerprint(values) != match[3]:
            raise ValueError("Model configuration changed after workflow preparation")
        names = match[2].split(",")
        if any(name not in _ENVIRONMENT_NAMES for name in names):
            raise ValueError("Unsupported worker environment names")
        if any(not values.get(name) or "\0" in values[name] for name in names):
            raise ValueError("Required worker environment is unavailable")
        reset = (
            "unset DF_API_KEY DF_API_URL DF_API_BASE_URL DF_MODEL_NAME && "
            if profile in {"text", "vision"}
            else ""
        )
        return (
            "true && "
            + reset
            + " && ".join(
                f"export {name}={shlex.quote(values[name])}" for name in names
            )
            + " &&"
        )

    prepared = deepcopy(workflow)
    for node in prepared.get("nodes", []):
        config = node.get("data", {}).get("config", {})
        command = config.get("command")
        if not isinstance(command, str):
            continue
        param = config.get("param")
        recipe = read_prepared_dataset_recipe(command)
        if recipe is not None:
            if (
                len(prepared["nodes"]) != 1
                or prepared.get("edges")
                or node["data"].get("nodeType") != "CustomCommandCPUNode"
            ):
                raise ValueError("Dataset recipe requires a single CPU node")
            if recipe.uses_labeling_gateway:
                if not isinstance(param, str):
                    raise ValueError("Missing Dataset gateway credential binding")
                config["param"] = _bind_dataset_key(param, recipe, environment)
            elif param not in ("", None):
                raise ValueError("Unexpected Dataset gateway parameter")
            continue
        if isinstance(param, str) and param.startswith("{{pyromind.dataset-key:"):
            raise ValueError("Dataset credential binding requires a prepared recipe")
        config["command"] = _BINDING.sub(bind, command)
        if "{{pyromind.runtime:" in config["command"]:
            raise ValueError("Invalid worker environment declaration")
    return prepared
