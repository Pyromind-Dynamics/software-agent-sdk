"""Decode the platform's terminal-formatted node output API."""

from __future__ import annotations

import json
import re
from typing import Any

import yaml


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SEPARATOR = re.compile(r"(?m)^-{10,}\s*$")
_PORT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(?::[ \t]*)?(.*)$", re.DOTALL)


def strip_terminal_codes(text: str) -> str:
    return _ANSI.sub("", text)


def parse_node_output(value: Any) -> dict[str, Any]:
    """Preserve named output values, including multiline config strings."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return _parse_text(value)
    if not isinstance(value, dict):
        return {}
    for key in ("data", "raw"):
        if isinstance(value.get(key), (dict, str)):
            return parse_node_output(value[key])
    if isinstance(value.get("parameters"), list):
        return {
            item["name"]: item.get("value")
            for item in value["parameters"]
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
    if isinstance(value.get("entries"), list):
        return _parse_text(
            "".join(
                item["m"]
                for item in value["entries"]
                if isinstance(item, dict) and isinstance(item.get("m"), str)
            )
        )
    return value


def _parse_text(text: str) -> dict[str, Any]:
    text = strip_terminal_codes(text).strip()
    if text.startswith("Output Values:"):
        result = {}
        for section in _SEPARATOR.split(text.removeprefix("Output Values:")):
            match = _PORT.fullmatch(section.strip())
            if match:
                result[match[1]] = match[2].strip()
        return result
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    return value if isinstance(value, dict) else {}
