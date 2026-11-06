from __future__ import annotations

import os
from pathlib import PurePosixPath

from pydantic import JsonValue
from pyromind_runtime.domain.content import JsonObject


def sandbox_path_for_storage_uri(uri: str | None) -> str | None:
    if not uri:
        return None
    value = uri.strip()
    if (
        not value
        or "://" in value
        or value == "public_data"
        or value.startswith("public_data/")
        or any(ord(character) < 32 for character in value)
    ):
        return None

    mount = os.getenv("PYROMIND_SANDBOX_MOUNT_PATH", "").strip().rstrip("/")
    mount = mount or "/target-workspace"
    if value == mount or value.startswith(f"{mount}/"):
        return value
    if value == "/workspace":
        return mount
    if value.startswith("/workspace/"):
        value = value.removeprefix("/workspace")
    elif not value.startswith("/"):
        value = f"/{value}"

    path = PurePosixPath(value)
    if ".." in path.parts:
        return None
    return f"{mount}{path.as_posix()}"


def storage_path_contract(node_config: dict[str, object]) -> JsonObject:
    raw_resources = node_config.get("resources", [])
    resources = raw_resources if isinstance(raw_resources, list) else []
    mappings: list[JsonValue] = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        uri = resource.get("uri")
        if not isinstance(uri, str):
            continue
        sandbox_path = sandbox_path_for_storage_uri(uri)
        if sandbox_path is None:
            continue
        resource_id = resource.get("id")
        mapping: JsonObject = {
            "resource_id": resource_id if isinstance(resource_id, str) else None,
            "storage_path": uri,
            "sandbox_path": sandbox_path,
        }
        mappings.append(mapping)
    mount = os.getenv("PYROMIND_SANDBOX_MOUNT_PATH", "").strip().rstrip("/")
    contract: JsonObject = {
        "storage_mount": mount or "/target-workspace",
        "rule": (
            "The mount prefix is additive: Storage path /datasets/a.jsonl is "
            "read in the Sandbox as /target-workspace/datasets/a.jsonl."
        ),
        "resources": mappings,
    }
    return contract
