from pathlib import Path

import pytest

from openhands.tools.utils import PathAccessPolicy, PathRule


def test_policy_resolves_relative_paths_and_checks_permissions(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = PathAccessPolicy((PathRule.create(workspace, "rwx"),))

    assert policy.check(workspace / "nested" / "file.txt", "write")
    assert policy.check(workspace / "nested" / "file.txt", "execute")
    assert not policy.check(workspace / "../outside.txt", "read")


def test_policy_does_not_match_same_prefix_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    sibling = tmp_path / "workspace-cache"
    workspace.mkdir()
    sibling.mkdir()
    policy = PathAccessPolicy((PathRule.create(workspace, "r"),))

    assert policy.check(workspace / "file.txt", "read")
    assert not policy.check(sibling / "file.txt", "read")
    assert not policy.check(workspace / "file.txt", "write")


def test_non_recursive_rule_only_allows_direct_children(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    policy = PathAccessPolicy((PathRule.create(config, "r", recursive=False),))

    assert policy.check(config / "settings.json", "read")
    assert not policy.check(config / "nested" / "settings.json", "read")


def test_require_raises_for_denied_operation(tmp_path: Path) -> None:
    policy = PathAccessPolicy((PathRule.create(tmp_path, "r"),))

    with pytest.raises(PermissionError, match="write"):
        policy.require(tmp_path / "file.txt", "write")
