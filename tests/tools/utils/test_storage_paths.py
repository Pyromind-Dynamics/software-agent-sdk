import pytest

from openhands.tools.utils.storage_paths import (
    StoragePathError,
    to_agent_storage_path,
    to_storage_path,
    to_tool_storage_path,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/workspace/a.csv", "/a.csv"),
        ("/workspace/exports/a.json", "/exports/a.json"),
        ("/datasets/a.csv", "/datasets/a.csv"),
        ("/datasets/nested/tree/a.csv", "/datasets/nested/tree/a.csv"),
        ("/storage/a.csv", "/storage/a.csv"),
        ("  /workspace/datasets/a.csv  ", "/datasets/a.csv"),
    ],
)
def test_to_storage_path_normalizes_public_and_storage_api_paths(
    value: str, expected: str
) -> None:
    assert to_storage_path(value) == expected


def test_to_tool_storage_path_uses_the_same_absolute_contract() -> None:
    assert to_tool_storage_path("/workspace/datasets/a.csv") == "/datasets/a.csv"
    assert to_tool_storage_path("/.pyromind-agent/conv/uploads/a.csv") == (
        "/.pyromind-agent/conv/uploads/a.csv"
    )


@pytest.mark.parametrize("value", ["datasets/a.csv", "./datasets/a.csv", "a.csv"])
def test_relative_storage_paths_are_rejected(value: str) -> None:
    with pytest.raises(StoragePathError, match="must be absolute"):
        to_tool_storage_path(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/datasets/a.csv", "/workspace/datasets/a.csv"),
        ("/workspace/datasets/a.csv", "/workspace/datasets/a.csv"),
    ],
)
def test_to_agent_storage_path_uses_the_public_workspace_root(
    value: str, expected: str
) -> None:
    assert to_agent_storage_path(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "storage/datasets/a.csv",
        "storage/a.csv",
        "storage/",
        "  storage/a.csv  ",
        "./storage/a.csv",
        "storage//datasets//a.csv",
    ],
)
def test_to_storage_path_rejects_the_removed_alias(value: str) -> None:
    """The alias is gone, so a stale spelling fails instead of half-working."""

    with pytest.raises(StoragePathError, match="alias was removed"):
        to_storage_path(value)


def test_the_rejection_names_both_working_spellings() -> None:
    with pytest.raises(StoragePathError) as raised:
        to_storage_path("storage/a.csv")

    message = str(raised.value)
    assert "/workspace/<path>" in message
    assert "'/<path>'" in message


def test_to_storage_path_rejects_the_storage_root() -> None:
    with pytest.raises(StoragePathError, match="must not be the Storage root"):
        to_storage_path("/")


@pytest.mark.parametrize("value", ["", "   "])
def test_to_storage_path_rejects_empty_values(value: str) -> None:
    with pytest.raises(StoragePathError, match="must not be empty"):
        to_storage_path(value)


@pytest.mark.parametrize("value", ["/datasets/../secret.csv", ".."])
def test_to_storage_path_rejects_traversal(value: str) -> None:
    with pytest.raises(StoragePathError):
        to_storage_path(value)


def test_to_storage_path_names_the_field_it_rejected() -> None:
    with pytest.raises(StoragePathError, match="^script_path"):
        to_storage_path("/", field="script_path")


def test_to_storage_path_rejects_a_workspace_path_with_a_fix() -> None:
    """A workspace path must not silently become a Storage path."""

    with pytest.raises(StoragePathError, match="conversation workspace path"):
        to_storage_path("public_data/report.html")


def test_storage_uri_syntax_is_not_a_filesystem_path() -> None:
    with pytest.raises(StoragePathError, match="filesystem path, not a URL"):
        to_storage_path("storage://datasets/a.csv")


def test_to_storage_path_keeps_a_workspace_directory_inside_the_mount() -> None:
    """The workspace is itself a Storage directory, so its absolute form is fine."""

    assert (
        to_storage_path("/.pyromind-agent/conv-1/public_data/report.html")
        == "/.pyromind-agent/conv-1/public_data/report.html"
    )


def test_target_workspace_is_not_a_model_or_tool_path() -> None:
    with pytest.raises(StoragePathError, match="execution-only path"):
        to_storage_path("/target-workspace/datasets/a.csv")
