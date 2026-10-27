import pytest

from openhands.tools.utils.storage_paths import (
    StoragePathError,
    to_storage_path,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Canonical form is already canonical.
        ("/datasets/a.csv", "/datasets/a.csv"),
        ("/datasets/nested/tree/a.csv", "/datasets/nested/tree/a.csv"),
        # An absolute /storage/ is a directory that really sits at the root.
        ("/storage/a.csv", "/storage/a.csv"),
        # The platform's own root name is infrastructure, not user data.
        ("/workspace/a.csv", "/a.csv"),
        ("/workspace/exports/a.json", "/exports/a.json"),
        # A bare relative path has always meant Storage-relative.
        ("datasets/a.csv", "/datasets/a.csv"),
        ("a.csv", "/a.csv"),
        # Noise that used to reach the platform verbatim.
        ("  /datasets/a.csv  ", "/datasets/a.csv"),
        ("./datasets/a.csv", "/datasets/a.csv"),
        ("//datasets/a.csv", "/datasets/a.csv"),
    ],
)
def test_to_storage_path_normalizes_every_accepted_spelling(
    value: str, expected: str
) -> None:
    assert to_storage_path(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/target-workspace/datasets/a.csv", "/datasets/a.csv"),
        ("/target-workspace/a.csv", "/a.csv"),
        # Already Storage-rooted input is untouched by the mount rule.
        ("/datasets/a.csv", "/datasets/a.csv"),
    ],
)
def test_to_storage_path_accepts_a_known_sandbox_mount(
    value: str, expected: str
) -> None:
    assert to_storage_path(value, mount="/target-workspace") == expected


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
        to_storage_path("storage/a.csv", mount="/target-workspace")

    message = str(raised.value)
    assert "/target-workspace/<path>" in message
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
    with pytest.raises(StoragePathError, match="must not contain"):
        to_storage_path(value)


def test_to_storage_path_names_the_field_it_rejected() -> None:
    with pytest.raises(StoragePathError, match="^script_path"):
        to_storage_path("/", field="script_path")


def test_to_storage_path_rejects_a_workspace_path_with_a_fix() -> None:
    """A workspace path must not silently become a Storage path."""

    with pytest.raises(StoragePathError, match="conversation workspace path"):
        to_storage_path("public_data/report.html")


def test_to_storage_path_keeps_a_workspace_directory_inside_the_mount() -> None:
    """The workspace is itself a Storage directory, so its absolute form is fine."""

    assert (
        to_storage_path("/.pyromind-agent/conv-1/public_data/report.html")
        == "/.pyromind-agent/conv-1/public_data/report.html"
    )


def test_the_mount_spelling_needs_no_alias_to_reach_storage() -> None:
    assert to_storage_path(
        "/target-workspace/datasets/a.csv", mount="/target-workspace"
    ) == to_storage_path("/datasets/a.csv")
