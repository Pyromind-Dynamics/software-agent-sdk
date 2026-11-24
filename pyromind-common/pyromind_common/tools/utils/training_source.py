"""Shared, side-effect-free W&B location and credential extraction."""

import re
from typing import Any


RUN_URL_RE = re.compile(r"https://wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([\w-]+)")
RUN_SETUP_RE = re.compile(r"wandb:\s+setting up run\s+(\w+)")
RUN_PATH_RE = re.compile(r"run-\d{8}_\d{6}-(\w+)")
CONFIG_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")


def config_text(text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2)
        for line in text.splitlines()
        if (match := CONFIG_LINE_RE.match(line))
    }


def wandb_environment(output: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}

    def merge(text: str) -> None:
        for key, value in config_text(text).items():
            if value:
                result.setdefault(key.upper(), value)

    config = output.get("wandb_config")
    if isinstance(config, dict):
        result.update({str(k).upper(): str(v) for k, v in config.items()})
    elif isinstance(config, str):
        merge(config)
    for key, value in output.items():
        name = key.upper()
        if name.startswith("WANDB_") and name != "WANDB_CONFIG" and value is not None:
            result[name] = str(value)
    entries = output.get("entries")
    if isinstance(entries, list):
        merge(
            "".join(
                e.get("m", "")
                for e in entries
                if isinstance(e, dict) and isinstance(e.get("m"), str)
            )
        )
    if not result:
        for value in output.values():
            if isinstance(value, str) and value.strip():
                merge(value)
                break
    return {key: value for key, value in result.items() if key.startswith("WANDB_")}


def wandb_run_reference(text: str) -> tuple[str, str, str] | None:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    if match := RUN_URL_RE.search(text):
        return match.group(1), match.group(2), match.group(3)
    if match := RUN_SETUP_RE.search(text) or RUN_PATH_RE.search(text):
        return "", "", match.group(1)
    return None
