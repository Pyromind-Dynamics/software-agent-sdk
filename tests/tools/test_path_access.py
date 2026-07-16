from pathlib import Path

import pytest

from openhands.tools.utils import PathAccessPolicy, PathRule


def assert_conversation_policy_shape(
    policy: PathAccessPolicy, conversation_dir: Path
) -> None:
    """Assert the conversation-workspace permission matrix.

    Single source of truth for tool-integration tests. Keep this in sync with
    :data:`openhands.tools.utils.CONVERSATION_READ_WRITE_SUBPATHS` /
    :data:`openhands.tools.utils.CONVERSATION_READ_ONLY_SUBPATHS` and the
    ``exclude_workspace_fallback=True`` semantics.
    """
    workflow_file = conversation_dir / "workflow" / "workflow.py"
    events_file = conversation_dir / "events" / "0001.json"
    canvas_file = conversation_dir / "public_data" / "state.json"
    meta_file = conversation_dir / "meta.json"
    base_state = conversation_dir / "base_state.json"

    assert not policy.check(workflow_file, "read")
    assert not policy.check(workflow_file, "write")

    assert policy.check(events_file, "read")
    assert not policy.check(events_file, "write")

    assert policy.check(canvas_file, "read")
    assert policy.check(canvas_file, "write")

    assert not policy.check(meta_file, "read")
    assert not policy.check(meta_file, "write")
    assert not policy.check(base_state, "read")
    assert not policy.check(base_state, "write")


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


def test_conversation_workspace_restricts_to_subpaths(tmp_path: Path) -> None:
    from openhands.tools.utils import (
        CONVERSATION_READ_ONLY_SUBPATHS,
        CONVERSATION_READ_WRITE_SUBPATHS,
        default_path_access_policy,
    )

    conversation_dir = tmp_path / "conversation"
    conversation_dir.mkdir()
    policy = default_path_access_policy(
        conversation_dir,
        workspace_read_only_subpaths=CONVERSATION_READ_ONLY_SUBPATHS,
        workspace_read_write_subpaths=CONVERSATION_READ_WRITE_SUBPATHS,
        exclude_workspace_fallback=True,
    )
    assert_conversation_policy_shape(policy, conversation_dir)
