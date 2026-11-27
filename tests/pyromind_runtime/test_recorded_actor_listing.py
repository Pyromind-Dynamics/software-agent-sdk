import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from harness_adapter.pi_adapter import actor_replay
from harness_adapter.pi_adapter.actor_replay import RecordedActorReplayReader
from harness_adapter.pi_adapter.actor_workspace import ActorTransport, ActorWorkspace
from pyromind_common.sdk.workspace.models import CommandResult
from pyromind_runtime.ports.storage import (
    EvaluationStorageMissingError,
    EvaluationStorageUnavailableError,
)


@pytest.fixture
def recorded_outputs(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    prefix = "/.pyromind-agent/test/output"
    directory = root / prefix.lstrip("/")
    directory.mkdir(parents=True)

    def mounted(path: str) -> Path:
        return root / Path(path).relative_to("/workspace")

    def execute(self, command, **kwargs):
        args = shlex.split(command)
        args[0] = sys.executable
        args[5] = str(mounted(args[5]))
        process = subprocess.run(args, capture_output=True, text=True, check=False)
        return CommandResult(
            command=command,
            exit_code=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            timeout_occurred=False,
        )

    def read(self, path, *, max_bytes):
        data = mounted(path).read_bytes()
        assert len(data) <= max_bytes
        return data

    monkeypatch.setattr(ActorWorkspace, "execute_command", execute)
    monkeypatch.setattr(ActorTransport, "read_file", read)
    return (
        RecordedActorReplayReader(
            ActorWorkspace(
                working_dir="/workspace/.pyromind-agent/test",
                transport=ActorTransport("https://actor.test", "owner/actor"),
            ),
            max_read_bytes=3,
        ),
        prefix,
        directory,
    )


def test_lists_only_regular_files_and_directories_in_the_recorded_workspace(
    recorded_outputs,
):
    reader, prefix, output = recorded_outputs
    checkpoint = output / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "weights").write_bytes(b"abc")
    (output / "val.jsonl").touch()
    (output / "foreign").symlink_to(output.parent)
    os.mkfifo(output / "pipe")
    assert reader.list("/workspace" + prefix) == {"checkpoint-4", "val.jsonl"}
    assert reader.list(prefix + "/checkpoint-4") == {"weights"}
    assert reader.read(prefix + "/checkpoint-4/weights") == b"abc"
    with pytest.raises(ValueError):
        reader.list("/.pyromind-agent/different")
    with pytest.raises(EvaluationStorageUnavailableError):
        reader.list(prefix + "/foreign")
    with pytest.raises(EvaluationStorageUnavailableError):
        reader.list(prefix + "/foreign/output")
    with pytest.raises(EvaluationStorageMissingError):
        reader.list(prefix + "/missing")


def test_lists_and_file_reads_are_bounded(recorded_outputs, monkeypatch):
    reader, prefix, output = recorded_outputs
    (output / "weights").write_bytes(b"abcd")
    with pytest.raises(EvaluationStorageUnavailableError):
        reader.read(prefix + "/weights")
    monkeypatch.setattr(actor_replay, "_MAX_ENTRIES", 1)
    (output / "config.json").touch()
    with pytest.raises(EvaluationStorageUnavailableError):
        reader.list(prefix)
    monkeypatch.setattr(actor_replay, "_MAX_ENTRIES", 2)
    monkeypatch.setattr(actor_replay, "_MAX_LIST_BYTES", 8)
    with pytest.raises(EvaluationStorageUnavailableError):
        reader.list(prefix)
