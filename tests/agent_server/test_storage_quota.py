from types import SimpleNamespace
from uuid import UUID

import pytest

from openhands.agent_server.storage_quota import (
    OH_CONVERSATION_STORAGE_QUOTA_ENV,
    ConversationStorageQuota,
    parse_storage_size,
    project_id_for,
    quota_from_env,
)


def test_parse_storage_size():
    assert parse_storage_size("50M") == 50 * 1024 * 1024
    assert parse_storage_size("1G") == 1024**3
    assert parse_storage_size("512") == 512
    assert parse_storage_size("2t") == 2 * 1024**4

    with pytest.raises(ValueError, match="invalid storage size"):
        parse_storage_size("10X")


def test_project_id_is_stable_and_positive():
    conversation_id = UUID("af3258e4-6542-408a-ba43-1c6f8ee37821")
    assert project_id_for(conversation_id) == project_id_for(conversation_id)
    assert 0 < project_id_for(conversation_id) < (1 << 31)


def test_quota_from_env(monkeypatch):
    monkeypatch.delenv(OH_CONVERSATION_STORAGE_QUOTA_ENV, raising=False)
    assert quota_from_env().limit_bytes is None

    monkeypatch.setenv(OH_CONVERSATION_STORAGE_QUOTA_ENV, "500M")
    assert quota_from_env().limit_bytes == 500 * 1024 * 1024


def _make_running_quota(monkeypatch, calls, tmp_path, limit):
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.shutil.which",
        lambda _: "/usr/bin/xfs_quota",
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._mountpoint_for", lambda _: tmp_path
    )

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("openhands.agent_server.storage_quota.subprocess.run", fake_run)
    return ConversationStorageQuota(limit)


def test_apply_runs_project_and_limit_commands(monkeypatch, tmp_path):
    directory = tmp_path / "conv"
    directory.mkdir()
    calls = []
    quota = _make_running_quota(monkeypatch, calls, tmp_path, 50 * 1024 * 1024)
    conversation_id = UUID("af3258e4-6542-408a-ba43-1c6f8ee37821")

    assert quota.apply(directory, conversation_id)

    project_id = project_id_for(conversation_id)
    assert calls == [
        [
            "xfs_quota",
            "-x",
            "-c",
            f"project -s -p {directory} {project_id}",
            str(tmp_path),
        ],
        [
            "xfs_quota",
            "-x",
            "-c",
            f"limit -p bhard={50 * 1024 * 1024} {project_id}",
            str(tmp_path),
        ],
    ]


def test_disabled_quota_skips_commands(monkeypatch, tmp_path):
    directory = tmp_path / "conv"
    directory.mkdir()
    calls = []
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.subprocess.run",
        lambda *args, **kwargs: calls.append(args) and SimpleNamespace(returncode=0),
    )

    quota = ConversationStorageQuota(None)
    assert not quota.apply(directory, UUID(int=1))
    assert not quota.remove(directory, UUID(int=1))
    assert calls == []


def test_quota_remove_runs_project_remove(monkeypatch, tmp_path):
    directory = tmp_path / "conv"
    directory.mkdir()
    calls = []
    quota = _make_running_quota(monkeypatch, calls, tmp_path, 10 * 1024)
    conversation_id = UUID("af3258e4-6542-408a-ba43-1c6f8ee37821")
    assert quota.apply(directory, conversation_id)
    assert quota.remove(directory, conversation_id)

    project_id = project_id_for(conversation_id)
    assert calls[-1] == [
        "xfs_quota",
        "-x",
        "-c",
        f"project -c {project_id}",
        str(tmp_path),
    ]


def test_quota_command_failure_returns_false(monkeypatch, tmp_path):
    directory = tmp_path / "conv"
    directory.mkdir()
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.shutil.which",
        lambda _: "/usr/bin/xfs_quota",
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._mountpoint_for", lambda _: tmp_path
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="operation not permitted"
        ),
    )

    quota = ConversationStorageQuota(10 * 1024)
    assert not quota.apply(directory, UUID(int=2))
