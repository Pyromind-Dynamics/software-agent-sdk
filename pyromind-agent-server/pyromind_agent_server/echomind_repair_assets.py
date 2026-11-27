from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import PurePosixPath

from pyromind_agent_server.echomind_projects import ProjectNode
from pyromind_runtime.domain.content import JsonObject, JsonValue
from pyromind_runtime.domain.graph_repair import RecipeAssetReplacement
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.ports.storage import EvaluationStorageMissingError


_RECIPE_CHECK = re.compile(
    r'^test "\$\(sha256sum (?P<path>/target-workspace/\.pyromind-agent/'
    r"[\w./-]+/recipe-assets/(?P<digest>[a-f0-9]{64})/"
    r'(?P<role>export|pipeline|prepare|verifier)\.py) \| cut -c1-64\)" = (?P=digest)$',
    re.MULTILINE,
)


def recipe_assets(template: str, conversation_id: str) -> list[JsonObject]:
    assets: list[JsonObject] = []
    root = PurePosixPath("/target-workspace/.pyromind-agent", conversation_id)
    for match in _RECIPE_CHECK.finditer(template):
        path = PurePosixPath(match["path"])
        if not path.is_relative_to(root) or ".." in path.parts:
            continue
        asset: JsonObject = {
            "role": match["role"],
            "sha256": match["digest"],
            "path": "/workspace" + str(path).removeprefix("/target-workspace"),
        }
        if asset not in assets:
            assets.append(asset)
    return assets


def replace_recipe_asset(
    node: ProjectNode,
    conversation_id: str,
    replacement: JsonValue,
    read: Callable[[str], bytes],
) -> str:
    change = RecipeAssetReplacement.model_validate(replacement)
    old, new = change.old_sha256, change.new_sha256
    if old == new:
        raise ValueError("Expected a different SHA-256 for the new recipe asset")
    config = node.config.get(node.executor)
    template = config.get("command_template") if isinstance(config, dict) else None
    if node.executor != "dataset_runtime" or not isinstance(template, str):
        raise ValueError("Target node has no Dataset recipe command")
    matches = [
        a for a in recipe_assets(template, conversation_id) if a["sha256"] == old
    ]
    if len(matches) != 1:
        raise ValueError("No unique matching frozen recipe asset in this conversation")
    source_path = PurePosixPath(str(matches[0]["path"]))
    target_path = source_path.parent.parent / new / source_path.name
    if change.new_asset_path is not None:
        provided = change.new_asset_path
        target_path = PurePosixPath(provided)
        root = PurePosixPath("/workspace/.pyromind-agent", conversation_id)
        if (
            not re.fullmatch(r"[a-zA-Z0-9._/-]+", provided)
            or str(target_path) != provided
            or ".." in target_path.parts
            or not target_path.is_relative_to(root)
            or target_path.parts[-3:] != ("recipe-assets", new, source_path.name)
        ):
            raise ValueError(
                "new_asset_path must be a safe absolute path inside the original "
                "conversation ending in recipe-assets/<new_sha256>/<original filename>"
            )
    try:
        data = read(to_storage_api_path(str(target_path)))
    except (EvaluationStorageMissingError, FileNotFoundError) as exc:
        raise ValueError(
            f"Frozen replacement asset not found at {target_path}. "
            "Supply the existing new_asset_path from the verified replay."
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "Cannot read the replacement asset at its frozen path"
        ) from exc
    if hashlib.sha256(data).hexdigest() != new:
        raise ValueError("Replacement asset bytes do not match the requested SHA-256")
    source = str(source_path).replace("/workspace/", "/target-workspace/", 1)
    target = str(target_path).replace("/workspace/", "/target-workspace/", 1)
    updated = _RECIPE_CHECK.sub(
        lambda match: (
            f'test "$(sha256sum {target} | cut -c1-64)" = {new}'
            if match["path"] == source
            else match[0]
        ),
        template,
    )
    command, count = re.subn(
        r"(?<!\S)" + re.escape(source) + r"(?=\s|$)", target, updated
    )
    if not count:
        raise ValueError("Frozen recipe asset is checked but never consumed")
    return command
