import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from openhands.sdk.workspace.models import CommandResult
from openhands.sdk.workspace.workspace import LocalWorkspace
from openhands.tools.utils.workspace_staging import (
    WorkspaceArchiveTruncatedError,
    WorkspaceStagingError,
    resolve_workspace_path,
    stage_workspace_members,
    stage_workspace_path,
)


def _workspace(
    *,
    working_dir: str = "/target-workspace/.pyromind-agent/conv-1",
    storage_path: str | None = "/target-workspace",
) -> SimpleNamespace:
    return SimpleNamespace(working_dir=working_dir, storage_path=storage_path)


def test_resolve_workspace_path_keeps_relative_paths_workspace_scoped() -> None:
    relative, from_storage = resolve_workspace_path(
        _workspace(), "public_data/data-preparation/pipeline.py"
    )

    assert relative == PurePosixPath("public_data/data-preparation/pipeline.py")
    assert from_storage is False


def test_resolve_workspace_path_maps_storage_alias_and_absolute_mount() -> None:
    alias, alias_storage = resolve_workspace_path(
        _workspace(), "storage/datasets/a.jsonl"
    )
    absolute, absolute_storage = resolve_workspace_path(
        _workspace(), "/target-workspace/datasets/a.jsonl"
    )

    assert (alias, alias_storage) == (PurePosixPath("datasets/a.jsonl"), True)
    assert (absolute, absolute_storage) == (PurePosixPath("datasets/a.jsonl"), True)


def test_resolve_workspace_path_maps_absolute_workspace_path() -> None:
    relative, from_storage = resolve_workspace_path(
        _workspace(), "/target-workspace/.pyromind-agent/conv-1/public_data/out.jsonl"
    )

    assert relative == PurePosixPath("public_data/out.jsonl")
    assert from_storage is False


def test_resolve_workspace_path_keeps_storage_plain_without_mount() -> None:
    relative, from_storage = resolve_workspace_path(
        _workspace(storage_path=None), "storage/datasets/a.jsonl"
    )

    assert relative == PurePosixPath("storage/datasets/a.jsonl")
    assert from_storage is False


@pytest.mark.parametrize(
    "path",
    [
        "../outside.jsonl",
        "/somewhere/else.jsonl",
        "public_data/../../escape.jsonl",
    ],
)
def test_resolve_workspace_path_rejects_paths_outside_the_workspace(
    path: str,
) -> None:
    with pytest.raises(WorkspaceStagingError):
        resolve_workspace_path(_workspace(), path)


def test_resolve_workspace_path_accepts_pathlib_input() -> None:
    relative, _ = resolve_workspace_path(_workspace(), Path("public_data/out.jsonl"))

    assert relative == PurePosixPath("public_data/out.jsonl")


class _ArchiveFailingWorkspace:
    """Sandbox double whose archiving step fails with the given stderr."""

    is_remote = True

    def __init__(self, workspace: Any, stderr: str, failures: int) -> None:
        self.working_dir = workspace.working_dir
        self.storage_path = workspace.storage_path
        self.archive_attempts = 0
        self._workspace = workspace
        self._stderr = stderr
        self._failures = failures

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        if command.startswith("tar -czf"):
            self.archive_attempts += 1
            if self.archive_attempts <= self._failures:
                return CommandResult(
                    command=command,
                    exit_code=1,
                    stdout="",
                    stderr=self._stderr,
                    timeout_occurred=False,
                )
        return self._workspace.execute_command(command, cwd=cwd, timeout=timeout)

    def file_download(self, source_path: str | Path, destination_path: str | Path):
        return self._workspace.file_download(source_path, destination_path)


class _CorruptDownloadWorkspace:
    """Sandbox double that returns the same number of unusable archive bytes."""

    is_remote = True

    def __init__(self, workspace: Any) -> None:
        self.working_dir = workspace.working_dir
        self.storage_path = workspace.storage_path
        self._workspace = workspace

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        return self._workspace.execute_command(command, cwd=cwd, timeout=timeout)

    def file_download(self, source_path: str | Path, destination_path: str | Path):
        result = self._workspace.file_download(source_path, destination_path)
        destination = Path(destination_path)
        content = destination.read_bytes()
        # Keep the length so the size check passes and extraction has to fail.
        destination.write_bytes(bytes(len(content)))
        return result


class _UnprobeableSizeWorkspace:
    """Sandbox double whose size probe fails, as a flaky terminal would."""

    is_remote = True

    def __init__(self, workspace: Any) -> None:
        self.working_dir = workspace.working_dir
        self.storage_path = workspace.storage_path
        self._workspace = workspace

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> CommandResult:
        if "wc -c" in command:
            raise RuntimeError("terminal bridge unavailable")
        return self._workspace.execute_command(command, cwd=cwd, timeout=timeout)

    def file_download(self, source_path: str | Path, destination_path: str | Path):
        return self._workspace.file_download(source_path, destination_path)


def test_stage_workspace_path_retries_the_juicefs_file_changed_race(
    sandbox_workspace,
    tmp_path: Path,
) -> None:
    """JuiceFS metadata settles on the first read, so the retry stages the file."""
    source = sandbox_workspace.workspace_dir / "public_data" / "pipeline.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('ok')\n", encoding="utf-8")
    workspace = _ArchiveFailingWorkspace(
        sandbox_workspace,
        "tar: pipeline.py: file changed as we read it",
        failures=1,
    )

    staged = stage_workspace_path(
        workspace,
        "public_data/pipeline.py",
        destination=tmp_path / "stage",
    )

    assert staged.read_text(encoding="utf-8") == "print('ok')\n"
    assert workspace.archive_attempts == 2


def test_stage_workspace_path_reports_other_archive_failures_immediately(
    sandbox_workspace,
    tmp_path: Path,
) -> None:
    """Only the metadata race is retried; a real archive failure still surfaces."""
    source = sandbox_workspace.workspace_dir / "public_data" / "pipeline.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('ok')\n", encoding="utf-8")
    workspace = _ArchiveFailingWorkspace(
        sandbox_workspace,
        "tar: pipeline.py: Cannot open: Permission denied",
        failures=1,
    )

    with pytest.raises(WorkspaceStagingError, match="Permission denied"):
        stage_workspace_path(
            workspace,
            "public_data/pipeline.py",
            destination=tmp_path / "stage",
        )

    assert workspace.archive_attempts == 1


def test_stage_workspace_members_mirrors_the_storage_layout(
    sandbox_workspace,
    tmp_path: Path,
) -> None:
    """Selected members land beside their manifest, not at the staging root."""
    sample = sandbox_workspace.storage_dir / "datasets" / "prelabel_in"
    (sample / "images").mkdir(parents=True)
    (sample / "images" / "b0.bmp").write_bytes(b"bmp")
    (sample / "unused.bmp").write_bytes(b"unused")
    stage = tmp_path / "stage"

    stage_workspace_members(
        sandbox_workspace,
        "storage/datasets/prelabel_in",
        ["images/b0.bmp"],
        destination=stage,
    )

    staged = stage / "storage" / "datasets" / "prelabel_in"
    assert (staged / "images" / "b0.bmp").read_bytes() == b"bmp"
    assert not (staged / "unused.bmp").exists()
    assert not (stage / "storage" / "images").exists()


def test_stage_workspace_members_mirrors_the_workspace_layout(
    tmp_path: Path,
) -> None:
    """The host-backed path keeps the same layout as the remote one."""
    workspace = LocalWorkspace(working_dir=tmp_path)
    sample = tmp_path / "public_data" / "prelabel_in"
    sample.mkdir(parents=True)
    (sample / "b0.bmp").write_bytes(b"bmp")
    stage = tmp_path / "stage"

    stage_workspace_members(
        workspace,
        "public_data/prelabel_in",
        ["b0.bmp"],
        destination=stage,
    )

    assert (stage / "public_data" / "prelabel_in" / "b0.bmp").read_bytes() == b"bmp"


def test_stage_workspace_members_splits_transfers_that_arrive_truncated(
    sandbox_workspace,
    truncated_downloads,
    tmp_path: Path,
) -> None:
    """A cut transfer halves the group until every archive fits the response."""
    sample = sandbox_workspace.storage_dir / "datasets" / "prelabel_in"
    sample.mkdir(parents=True)
    members = []
    for index in range(8):
        name = f"img{index}.bmp"
        # Incompressible content keeps the archive size proportional to members.
        (sample / name).write_bytes(os.urandom(4096))
        members.append(name)
    workspace = truncated_downloads(sandbox_workspace, limit=12000)
    stage = tmp_path / "stage"

    stage_workspace_members(
        workspace,
        "storage/datasets/prelabel_in",
        members,
        destination=stage,
    )

    staged = stage / "storage" / "datasets" / "prelabel_in"
    assert sorted(path.name for path in staged.iterdir()) == sorted(members)
    assert workspace.downloads > 1


def test_stage_workspace_members_reports_a_truncated_single_member(
    sandbox_workspace,
    truncated_downloads,
    tmp_path: Path,
) -> None:
    """One member that cannot be halved further fails with its sizes."""
    sample = sandbox_workspace.storage_dir / "datasets" / "prelabel_in"
    sample.mkdir(parents=True)
    (sample / "big.bmp").write_bytes(os.urandom(8192))
    workspace = truncated_downloads(sandbox_workspace, limit=4096, echo_commands=True)

    with pytest.raises(WorkspaceArchiveTruncatedError, match=r"4096 of \d+ bytes"):
        stage_workspace_members(
            workspace,
            "storage/datasets/prelabel_in",
            ["big.bmp"],
            destination=tmp_path / "stage",
        )


def test_stage_workspace_members_reports_an_unreadable_archive(
    sandbox_workspace,
    tmp_path: Path,
) -> None:
    """A body that is not a usable archive is reported, not leaked as a crash."""
    sample = sandbox_workspace.storage_dir / "datasets" / "prelabel_in"
    sample.mkdir(parents=True)
    (sample / "b0.bmp").write_bytes(b"bmp")
    workspace = _CorruptDownloadWorkspace(sandbox_workspace)

    with pytest.raises(WorkspaceArchiveTruncatedError):
        stage_workspace_members(
            workspace,
            "storage/datasets/prelabel_in",
            ["b0.bmp"],
            destination=tmp_path / "stage",
        )


def test_stage_workspace_members_survives_an_unavailable_size_probe(
    sandbox_workspace,
    tmp_path: Path,
) -> None:
    """A terminal that cannot report sizes leaves extraction as the verdict."""
    sample = sandbox_workspace.storage_dir / "datasets" / "prelabel_in"
    sample.mkdir(parents=True)
    (sample / "b0.bmp").write_bytes(b"bmp")
    workspace = _UnprobeableSizeWorkspace(sandbox_workspace)

    stage_workspace_members(
        workspace,
        "storage/datasets/prelabel_in",
        ["b0.bmp"],
        destination=tmp_path / "stage",
    )

    staged = tmp_path / "stage" / "storage" / "datasets" / "prelabel_in" / "b0.bmp"
    assert staged.read_bytes() == b"bmp"
