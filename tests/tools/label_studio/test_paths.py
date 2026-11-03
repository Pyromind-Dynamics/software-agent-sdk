"""All Label Studio inputs use the same Storage path interpretation."""

from types import SimpleNamespace

import pytest

from openhands.tools.label_studio.paths import resolve_storage_input


@pytest.mark.parametrize(
    "path",
    [
        "/target-workspace/datasets/pcb/defect.jpg",
        "/datasets/pcb/defect.jpg",
        "datasets/pcb/defect.jpg",
    ],
)
def test_storage_alias_mount_and_object_paths_are_equivalent(path):
    workspace = SimpleNamespace(
        working_dir="/target-workspace/.pyromind-agent/conversation",
        storage_path="/target-workspace",
    )
    result = resolve_storage_input(path, workspace, "conversation")
    assert result.original == path
    assert result.resolved == "/datasets/pcb/defect.jpg"
    assert path in result.diagnostic()
    assert "path_type=" in result.diagnostic()


def test_workspace_file_uses_conversation_storage_namespace():
    workspace = SimpleNamespace(
        working_dir="/target-workspace/.pyromind-agent/one",
        storage_path="/target-workspace",
    )
    result = resolve_storage_input("public_data/map.json", workspace, "one")
    assert result.resolved == "/.pyromind-agent/one/public_data/map.json"
    assert result.kind == "workspace"


def test_the_removed_storage_alias_is_rejected():
    """The alias is gone; a stale spelling must not resolve to something else."""

    with pytest.raises(ValueError, match="alias was removed"):
        resolve_storage_input("storage/datasets/pcb/defect.jpg")


def test_storage_directory_named_storage_is_not_removed():
    assert (
        resolve_storage_input("/storage/datasets/a").resolved == "/storage/datasets/a"
    )


@pytest.mark.parametrize(
    "path", ["", "/", ".", "storage/", "../a", "storage/../a", "a\nb"]
)
def test_invalid_paths_are_rejected_before_resolution(path):
    with pytest.raises(ValueError):
        resolve_storage_input(path)
