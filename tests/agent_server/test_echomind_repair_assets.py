from hashlib import sha256
from unittest.mock import Mock

import pytest

from pyromind_agent_server.echomind_projects import ProjectNode
from pyromind_agent_server.echomind_repair_assets import replace_recipe_asset
from pyromind_agent_server.echomind_repair_proposals import repair_node_evidence
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.ports.storage import EvaluationStorageMissingError


def test_replace_only_a_verified_recipe_asset_from_the_same_conversation() -> None:
    old = sha256(b"old").hexdigest()
    fixed = b"verified export script"
    new = sha256(fixed).hexdigest()
    prefix = "/.pyromind-agent/owner/preparation/replay/recipe-assets"
    path = f"/target-workspace{prefix}/{old}/export.py"
    template = (
        f'export DF_API_KEY="$param"\ntest "$(sha256sum {path} | cut -c1-64)" = {old}'
        f'\npython3 {path} "$output"\necho "unrelated digest {old}"'
    )
    node = ProjectNode(
        id="dataset",
        title="Export",
        type="dataset",
        executor="dataset_runtime",
        config={"dataset_runtime": {"command_template": template}},
    )
    evidence = repair_node_evidence(node, RequestContext(user_id="42"), "owner")
    assert "command_template" not in evidence
    assert evidence["recipe_assets"] == [
        {"role": "export", "sha256": old, "path": f"/workspace{prefix}/{old}/export.py"}
    ]
    replacement: JsonObject = {"old_sha256": old, "new_sha256": new}
    read = Mock(return_value=fixed)
    command = replace_recipe_asset(node, "owner", replacement, read)
    assert command.splitlines() == [
        'export DF_API_KEY="$param"',
        f'test "$(sha256sum {path.replace(old, new)} | cut -c1-64)" = {new}',
        f'python3 {path.replace(old, new)} "$output"',
        f'echo "unrelated digest {old}"',
    ]
    read.assert_called_once_with(f"{prefix}/{new}/export.py")
    read.reset_mock()
    with pytest.raises(ValueError, match="No unique matching"):
        replace_recipe_asset(node, "other", replacement, read)
    read.assert_not_called()
    with pytest.raises(ValueError, match="bytes do not match"):
        replace_recipe_asset(node, "owner", replacement, lambda _: b"unverified")

    alternate_dir = (
        "/workspace/.pyromind-agent/owner/data_preparation/replay-b/recipe-assets"
    )
    alternate_file = f"{alternate_dir}/{new}/export.py"
    relocated: JsonObject = {**replacement, "new_asset_path": alternate_file}
    revised = replace_recipe_asset(node, "owner", relocated, read)
    assert (
        revised.count(alternate_file.replace("/workspace/", "/target-workspace/")) == 2
    )
    read.assert_called_once_with(alternate_file.removeprefix("/workspace"))
    read.reset_mock()
    for invalid_path in (
        alternate_file.replace("/owner/", "/other/"),
        alternate_file.replace("replay-b", "../replay-b"),
        alternate_file.replace("replay-b", "replay b"),
        alternate_file.replace("replay-b", "replay\nb"),
        alternate_file.replace("replay-b", "$(echo unsafe)"),
        alternate_file.replace("replay-b", "{graph_run_id}"),
        alternate_file.replace(new, old),
        alternate_file.replace("export.py", "prepare.py"),
    ):
        with pytest.raises(ValueError, match="new_asset_path"):
            replace_recipe_asset(
                node, "owner", {**replacement, "new_asset_path": invalid_path}, read
            )
    read.assert_not_called()
    read.side_effect = EvaluationStorageMissingError("missing")
    with pytest.raises(ValueError, match="Frozen replacement asset not found"):
        replace_recipe_asset(node, "owner", relocated, read)
