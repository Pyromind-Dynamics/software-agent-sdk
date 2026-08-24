from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from openhands.agent_server.storage_quota import (
    OH_CONVERSATION_STORAGE_QUOTA_ENV,
    OH_STORAGE_QUOTA_DEVICE_ENV,
    OH_STORAGE_QUOTA_REQUIRED_ENV,
    ConversationStorageQuota,
    parse_storage_size,
    preflight_storage_quota,
    project_id_for,
    quota_from_env,
    storage_quota_required,
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


def _fake_device_for(mount_point, devices):
    return Path(str(mount_point) + "-dev") if devices else None


def test_run_uses_device_node_when_available(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.shutil.which",
        lambda _: "/usr/bin/xfs_quota",
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._mountpoint_for", lambda _: tmp_path
    )
    dev = tmp_path / "dev-node"
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._device_node_for",
        lambda _: dev,
    )

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("openhands.agent_server.storage_quota.subprocess.run", fake_run)
    quota = ConversationStorageQuota(10)
    assert quota.apply(tmp_path, UUID("af3258e4-6542-408a-ba43-1c6f8ee37821")) is True
    assert calls and all(str(dev) in args for args in calls)


def test_run_falls_back_to_mountpoint_without_device(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.shutil.which",
        lambda _: "/usr/bin/xfs_quota",
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._mountpoint_for", lambda _: tmp_path
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._device_node_for", lambda _: None
    )

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("openhands.agent_server.storage_quota.subprocess.run", fake_run)
    quota = ConversationStorageQuota(10)
    assert quota.apply(tmp_path, UUID("af3258e4-6542-408a-ba43-1c6f8ee37821")) is True
    assert calls and str(tmp_path) in calls[0]


def test_device_node_parses_mountinfo_major_minor(monkeypatch, tmp_path):

    mountinfo = "23 21 259:5 / /workspace rw,relatime - xfs /dev/nvme1n1 rw,prjquota"
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.Path.read_text",
        lambda self: mountinfo,
    )
    made = []
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.os.mknod",
        lambda path, mode, dev: made.append((path, mode, dev)),
    )
    from openhands.agent_server.storage_quota import _device_node_for

    node = _device_node_for(Path("/workspace"))
    assert node is not None


def test_device_node_prefers_existing_source_device(monkeypatch, tmp_path):
    device = tmp_path / "nvme1n1"
    device.write_text("block device")
    mountinfo = f"23 21 259:5 / /workspace rw,relatime - xfs {device} rw,prjquota"
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.Path.read_text",
        lambda self: mountinfo,
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.stat.S_ISBLK",
        lambda _mode: True,
    )
    from openhands.agent_server.storage_quota import _device_node_for

    assert _device_node_for(Path("/workspace")) == device


def test_storage_quota_required_env(monkeypatch):
    monkeypatch.delenv(OH_STORAGE_QUOTA_REQUIRED_ENV, raising=False)
    assert not storage_quota_required()

    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(OH_STORAGE_QUOTA_REQUIRED_ENV, value)
        assert storage_quota_required()

    for value in ("0", "false", "off", ""):
        monkeypatch.setenv(OH_STORAGE_QUOTA_REQUIRED_ENV, value)
        assert not storage_quota_required()


def test_quota_from_env_parses_device_override(monkeypatch):
    monkeypatch.delenv(OH_STORAGE_QUOTA_DEVICE_ENV, raising=False)
    quota = ConversationStorageQuota(10)
    assert quota.device is None

    monkeypatch.setenv(OH_STORAGE_QUOTA_DEVICE_ENV, "/dev/nvme1n1")
    monkeypatch.setenv(OH_CONVERSATION_STORAGE_QUOTA_ENV, "500M")
    parsed = quota_from_env()
    assert parsed.device == Path("/dev/nvme1n1")


def test_apply_uses_configured_device(monkeypatch, tmp_path):
    directory = tmp_path / "conv"
    directory.mkdir()
    calls = []
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.shutil.which",
        lambda _: "/usr/bin/xfs_quota",
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._mountpoint_for", lambda _: tmp_path
    )
    dev = tmp_path / "real-device"

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("openhands.agent_server.storage_quota.subprocess.run", fake_run)
    quota = ConversationStorageQuota(10, device=dev)
    assert quota.apply(directory, UUID(int=3))
    assert calls and all(str(dev) in args for args in calls)


def test_apply_records_last_error_on_mountpoint_failure(monkeypatch, tmp_path):
    directory = tmp_path / "conv"
    directory.mkdir()
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._mountpoint_for", lambda _: None
    )
    quota = ConversationStorageQuota(10)
    assert not quota.apply(directory, UUID(int=4))
    assert "cannot resolve mountpoint" in (quota.last_error or "")


def test_run_records_command_failure_detail(monkeypatch, tmp_path):
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
        "openhands.agent_server.storage_quota._device_node_for", lambda _: None
    )

    def failing_run(args, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="", stderr="No such device or address"
        )

    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.subprocess.run", failing_run
    )
    quota = ConversationStorageQuota(10)
    assert not quota.apply(directory, UUID(int=5))
    assert quota.last_error is not None
    assert "No such device or address" in quota.last_error


def test_preflight_disabled_quota_returns_empty(monkeypatch):
    monkeypatch.delenv(OH_CONVERSATION_STORAGE_QUOTA_ENV, raising=False)
    assert preflight_storage_quota() == []


def _preflight_mount(monkeypatch, tmp_path, fs_type, options, report_ok=True):
    monkeypatch.setenv(OH_CONVERSATION_STORAGE_QUOTA_ENV, "500M")
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._mountpoint_for", lambda _: tmp_path
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._fs_type_and_options",
        lambda _: (fs_type, options),
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota.shutil.which",
        lambda _: "/usr/bin/xfs_quota",
    )
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._device_node_for", lambda _: None
    )

    def runner(*args, **kwargs):
        status = 0 if report_ok else 1
        stderr = "" if report_ok else "No such device or address"
        return SimpleNamespace(returncode=status, stdout="", stderr=stderr)

    monkeypatch.setattr("openhands.agent_server.storage_quota.subprocess.run", runner)


def test_preflight_reports_filesystem_problems(monkeypatch, tmp_path):
    _preflight_mount(
        monkeypatch, tmp_path, fs_type="ext4", options=("rw",), report_ok=False
    )
    problems = preflight_storage_quota(workspace_root=tmp_path)
    assert any("XFS" in problem for problem in problems)
    assert any("prjquota" in problem for problem in problems)
    assert any("No such device or address" in problem for problem in problems)


def test_preflight_reports_unopenable_device(monkeypatch, tmp_path):
    _preflight_mount(
        monkeypatch, tmp_path, fs_type="xfs", options=("prjquota",), report_ok=False
    )
    problems = preflight_storage_quota(workspace_root=tmp_path)
    assert any("cannot open device" in problem for problem in problems)


def test_preflight_ok_returns_empty(monkeypatch, tmp_path):
    _preflight_mount(monkeypatch, tmp_path, fs_type="xfs", options=("prjquota",))
    assert preflight_storage_quota(workspace_root=tmp_path) == []


def test_preflight_unknown_mountpoint(monkeypatch, tmp_path):
    monkeypatch.setenv(OH_CONVERSATION_STORAGE_QUOTA_ENV, "500M")
    monkeypatch.setattr(
        "openhands.agent_server.storage_quota._mountpoint_for", lambda _: None
    )
    problems = preflight_storage_quota(workspace_root=tmp_path)
    assert problems and "mountpoint" in problems[0]
