import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openhands.tools.terminal.sandbox import (
    APPARMOR_PROFILE_NAME,
    SandboxMemoryCgroup,
    TerminalSandbox,
    _is_bwrap_usable,
    parse_memory_limit,
    sandbox_memory_cgroup_from_env,
    terminal_sandbox_enabled,
    terminal_sandbox_mode,
)


@pytest.fixture(autouse=True)
def _clear_bwrap_cache():
    _is_bwrap_usable.cache_clear()
    yield
    _is_bwrap_usable.cache_clear()


def _option_index(args: list[str], option: str, value: str) -> int:
    for index, arg in enumerate(args[:-1]):
        if arg == option and args[index + 1] == value:
            return index
    raise AssertionError(f"{option} {value} not found in {args}")


def test_parse_memory_limit_accepts_human_sizes():
    assert parse_memory_limit("512M") == 512 * 1024 * 1024
    assert parse_memory_limit("1g") == 1024**3
    assert parse_memory_limit("300") == 300


def test_parse_memory_limit_rejects_invalid_value():
    with pytest.raises(ValueError):
        parse_memory_limit("banana")


def test_memory_cgroup_defaults_to_500m(monkeypatch: pytest.MonkeyPatch):
    captured: list[int] = []

    class FakeMemoryCgroup:
        def __init__(self, limit_bytes: int, *, root: Path = Path("/sys/fs/cgroup")):
            captured.append(limit_bytes)

        def create(self) -> bool:
            return True

    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.SandboxMemoryCgroup", FakeMemoryCgroup
    )
    monkeypatch.delenv("OH_SANDBOX_MEMORY_LIMIT", raising=False)
    assert sandbox_memory_cgroup_from_env() is not None
    assert captured == [500 * 1024 * 1024]


def test_memory_cgroup_empty_env_disables_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OH_SANDBOX_MEMORY_LIMIT", "")
    assert sandbox_memory_cgroup_from_env() is None


def test_memory_cgroup_create_sets_limits_in_root(tmp_path: Path):
    memory_cgroup = SandboxMemoryCgroup(64 * 1024 * 1024, root=tmp_path)
    assert memory_cgroup.create()
    assert memory_cgroup.path is not None
    assert memory_cgroup.path.parent == tmp_path
    assert (memory_cgroup.path / "memory.max").read_text() == str(64 * 1024 * 1024)
    assert (memory_cgroup.path / "memory.high").read_text() == str(
        64 * 1024 * 1024 * 3 // 4
    )
    assert (memory_cgroup.path / "memory.swap.max").read_text() == "0"


def test_memory_cgroup_create_degrades_on_unwritable_root(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    memory_cgroup = SandboxMemoryCgroup(1024, root=blocker)
    assert not memory_cgroup.create()
    assert memory_cgroup.path is None


def test_memory_cgroup_attach_moves_pid_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    memory_cgroup = SandboxMemoryCgroup(1024, root=tmp_path)
    assert memory_cgroup.create()
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._descendant_pids", lambda _pid: [100, 200]
    )
    memory_cgroup.attach(42)
    path = memory_cgroup.path
    assert path is not None
    procs = (path / "cgroup.procs").read_text().split()
    assert procs == ["42", "100", "200"]


def test_memory_cgroup_cleanup_removes_directory(tmp_path: Path):
    memory_cgroup = SandboxMemoryCgroup(1024, root=tmp_path)
    assert memory_cgroup.create()
    path = memory_cgroup.path
    assert path is not None and path.exists()
    memory_cgroup.cleanup()
    assert not path.exists()
    assert memory_cgroup.path is None


def test_terminal_sandbox_attaches_spawned_pid_to_memory_cgroup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    memory_cgroup = SandboxMemoryCgroup(1024, root=tmp_path)
    assert memory_cgroup.create()
    sandbox = TerminalSandbox(str(tmp_path), "off")
    sandbox._memory_cgroup = memory_cgroup
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._descendant_pids", lambda _pid: []
    )
    sandbox.attach_memory_cgroup(1234)
    path = memory_cgroup.path
    assert path is not None
    assert (path / "cgroup.procs").read_text().split() == ["1234"]


def test_terminal_sandbox_mode_rejects_unknown_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OH_TERMINAL_SANDBOX", "invalid")

    with pytest.raises(ValueError, match="must be one of"):
        terminal_sandbox_mode()


def test_terminal_sandbox_prepare_creates_private_tmp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    sandbox = TerminalSandbox(str(tmp_path), "auto")

    sandbox.prepare()

    assert (tmp_path / ".openhands-tmp").is_dir()
    assert (tmp_path / ".openhands-tmp").stat().st_mode & 0o777 == 0o700


def test_terminal_sandbox_applies_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: False
    )

    class FakeLandlock:
        def __init__(self, *, strict: bool):
            assert strict

        def __getattr__(self, name: str):
            return lambda *paths: self

    monkeypatch.setitem(
        __import__("sys").modules,
        "py_landlock",
        SimpleNamespace(Landlock=FakeLandlock),
    )
    sandbox = TerminalSandbox(str(tmp_path), "required")
    sandbox.prepare()

    assert sandbox._landlock_wrapper is not None
    wrapper = sandbox._landlock_wrapper.read_text()
    assert wrapper.startswith("#!/usr/bin/env python3\n")
    assert "from py_landlock import Landlock" in wrapper
    assert "os.execv(parent_python" in wrapper
    assert (
        str(tmp_path / ".openhands-tmp" / ".openhands-landlock-policy.json") in wrapper
    )

    policy = json.loads(
        (tmp_path / ".openhands-tmp" / ".openhands-landlock-policy.json").read_text()
    )
    assert str(tmp_path) in policy["read_write_paths"]
    assert any(p in {"/usr", "/bin", "/sbin"} for p in policy["executable_paths"])


def test_terminal_sandbox_uses_explicit_read_and_write_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: False
    )

    class FakeLandlock:
        def __init__(self, *, strict: bool):
            assert strict

        def __getattr__(self, name: str):
            return lambda *paths: self

    monkeypatch.setitem(
        __import__("sys").modules,
        "py_landlock",
        SimpleNamespace(Landlock=FakeLandlock),
    )
    workflow_dir = tmp_path / "workflow"
    events_dir = tmp_path / "events"
    workflow_dir.mkdir()
    events_dir.mkdir()
    sandbox = TerminalSandbox(
        str(workflow_dir),
        "required",
        read_only_paths=(str(events_dir),),
        read_write_paths=(str(workflow_dir),),
    )
    sandbox.prepare()

    policy = json.loads(
        (
            workflow_dir / ".openhands-tmp" / ".openhands-landlock-policy.json"
        ).read_text()
    )
    assert str(workflow_dir) in policy["read_write_paths"]
    assert str(events_dir) in policy["read_only_paths"]


def test_terminal_sandbox_off_does_not_enable_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )

    assert not terminal_sandbox_enabled("off")


def test_seatbelt_profile_denies_sibling_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Darwin"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.shutil.which",
        lambda _: "/usr/bin/sandbox-exec",
    )
    current = tmp_path / "current"
    current.mkdir()
    sandbox = TerminalSandbox(str(current), "required")

    sandbox.prepare()

    assert sandbox._seatbelt_profile is not None
    profile = sandbox._seatbelt_profile.read_text()
    assert f'(deny file-read* (subpath "{tmp_path}"))' in profile
    assert f'(allow file-read-metadata (literal "{tmp_path}"))' in profile
    assert f'(allow file-read-metadata (literal "{current}"))' in profile
    assert f'(allow file-read* (subpath "{current}"))' in profile
    assert '(allow file-read* (subpath "/agent-server/knowledge"))' in profile
    assert '(allow file-read* (subpath "/agent-server/.agents/skills"))' in profile
    assert '(allow file-write* (literal "/dev/null"))' in profile
    assert '(allow file-write* (literal "/dev/tty"))' in profile
    assert sandbox.wrap_command(["/bin/bash", "-i"])[:3] == [
        "/usr/bin/sandbox-exec",
        "-f",
        str(sandbox._seatbelt_profile),
    ]


def test_seatbelt_profile_blocks_meta_json_in_conversation_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Darwin"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.shutil.which",
        lambda _: "/usr/bin/sandbox-exec",
    )
    conv_dir = tmp_path / "conversations" / "abc123"
    conv_dir.mkdir(parents=True)
    (conv_dir / "events").mkdir()
    (conv_dir / "public_data").mkdir()
    (conv_dir / "meta.json").write_text("{}")

    sandbox = TerminalSandbox(
        str(conv_dir),
        "required",
        read_only_paths=("events",),
        read_write_paths=("public_data",),
    )
    sandbox.prepare()

    assert sandbox._seatbelt_profile is not None
    profile = sandbox._seatbelt_profile.read_text()
    events_path = str(conv_dir / "events")
    public_data_path = str(conv_dir / "public_data")
    assert f'(deny file-read* (subpath "{tmp_path / "conversations"}"))' in profile
    assert (
        f'(allow file-read-metadata (literal "{tmp_path / "conversations"}"))'
        in profile
    )
    assert f'(allow file-read-metadata (literal "{conv_dir}"))' in profile
    assert f'(allow file-read* (subpath "{events_path}"))' in profile
    assert f'(allow file-read* (subpath "{public_data_path}"))' in profile
    meta_path = str(conv_dir / "meta.json")
    assert f'(allow file-read* (subpath "{meta_path}"))' not in profile
    assert f'(allow file-read* (subpath "{conv_dir}"))' not in profile


@pytest.mark.skipif(
    platform.system() != "Darwin" or shutil.which("sandbox-exec") is None,
    reason="requires macOS sandbox-exec",
)
def test_seatbelt_allows_persistent_shell_paths_but_denies_siblings(
    tmp_path: Path,
) -> None:
    conversations = tmp_path / "conversations"
    conv_dir = conversations / "current"
    public_data = conv_dir / "public_data"
    events = conv_dir / "events"
    sibling = conversations / "sibling"
    public_data.mkdir(parents=True)
    events.mkdir()
    sibling.mkdir()
    (sibling / "secret.txt").write_text("secret")
    sandbox = TerminalSandbox(
        str(conv_dir),
        "required",
        read_only_paths=("events",),
        read_write_paths=("public_data",),
    )
    sandbox.prepare()
    try:
        allowed = subprocess.run(
            sandbox.wrap_command(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    (f'pwd; echo ok 2>/dev/null; mkdir -p "{public_data / "nested"}"'),
                ]
            ),
            cwd=conv_dir,
            capture_output=True,
            text=True,
        )
        if (
            allowed.returncode == 71
            and "sandbox_apply: Operation not permitted" in allowed.stderr
        ):
            pytest.skip("current process cannot create a nested Seatbelt sandbox")
        assert allowed.returncode == 0, allowed.stderr
        assert str(conv_dir) in allowed.stdout
        assert (public_data / "nested").is_dir()

        denied = subprocess.run(
            sandbox.wrap_command(
                [
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    f'cat "{sibling / "secret.txt"}"',
                ]
            ),
            cwd=conv_dir,
            capture_output=True,
            text=True,
        )
        assert denied.returncode != 0
        assert "secret" not in denied.stdout
    finally:
        sandbox.cleanup()


def test_sandbox_resolves_relative_subpaths_against_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conv_dir = tmp_path / "workspace"
    conv_dir.mkdir()
    sandbox = TerminalSandbox(
        str(conv_dir),
        "auto",
        read_only_paths=("events",),
        read_write_paths=("public_data",),
    )
    assert sandbox.read_only_paths == (conv_dir / "events",)
    assert sandbox.read_write_paths == (conv_dir / "public_data",)


def test_apparmor_wrap_command_prefixes_aa_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: True
    )
    sandbox = TerminalSandbox(str(tmp_path), "required")
    sandbox.prepare()

    wrapped = sandbox.wrap_command(["/bin/bash", "-i"])

    assert wrapped == [
        "aa-exec",
        "-p",
        APPARMOR_PROFILE_NAME,
        "--",
        "/bin/bash",
        "-i",
    ]
    assert sandbox._backend == "apparmor"


def test_apparmor_takes_priority_over_bwrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: True
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_bwrap_usable", lambda: True
    )
    sandbox = TerminalSandbox(str(tmp_path), "required")

    sandbox.prepare()

    assert sandbox._backend == "apparmor"


def test_required_mode_error_mentions_all_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: False
    )
    monkeypatch.setattr("openhands.tools.terminal.sandbox.shutil.which", lambda _: None)
    import builtins

    real_import = builtins.__import__

    def _raise(name, *a, **kw):
        if name == "py_landlock":
            raise ImportError("no py-landlock")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _raise)

    sandbox = TerminalSandbox(str(tmp_path), "required")
    with pytest.raises(RuntimeError, match="AppArmor profile not loaded"):
        sandbox.prepare()


def test_conversation_policy_prefers_bwrap_over_landlock_and_apparmor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: True
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_bwrap_usable", lambda: True
    )

    class FakeLandlock:
        def __init__(self, *, strict: bool):
            pass

        def __getattr__(self, name: str):
            return lambda *a, **kw: self

    monkeypatch.setitem(
        __import__("sys").modules,
        "py_landlock",
        SimpleNamespace(Landlock=FakeLandlock),
    )

    events_dir = tmp_path / "events"
    public_data_dir = tmp_path / "public_data"
    events_dir.mkdir()
    public_data_dir.mkdir()
    sandbox = TerminalSandbox(
        str(tmp_path),
        "required",
        read_only_paths=(str(events_dir),),
        read_write_paths=(str(public_data_dir),),
    )
    sandbox.prepare()

    assert sandbox._backend == "bwrap"
    wrapped = sandbox.wrap_command(["/bin/bash", "-i"])
    assert wrapped[:3] == ["bwrap", "--unshare-ipc", "--unshare-uts"]
    assert _option_index(wrapped, "--bind", str(public_data_dir)) < _option_index(
        wrapped, "--ro-bind", str(events_dir)
    )


def test_bwrap_tmp_is_bound_to_disk_not_tmpfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: False
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_bwrap_usable", lambda: True
    )

    sandbox = TerminalSandbox(str(tmp_path), "required")
    sandbox.prepare()

    assert sandbox._backend == "bwrap"
    wrapped = sandbox.wrap_command(["/bin/bash", "-i"])
    assert "--tmpfs" not in wrapped
    tmp_index = _option_index(wrapped, "--bind", str(sandbox._sandbox_tmp))
    assert wrapped[tmp_index + 2] == "/tmp"
    assert sandbox._sandbox_tmp.is_dir()


def test_conversation_policy_bwrap_binds_container_root_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: False
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_bwrap_usable", lambda: True
    )

    events_dir = tmp_path / "events"
    public_data_dir = tmp_path / "public_data"
    events_dir.mkdir()
    public_data_dir.mkdir()
    sandbox = TerminalSandbox(
        str(tmp_path),
        "required",
        read_only_paths=(str(events_dir),),
        read_write_paths=(str(public_data_dir),),
    )
    sandbox.prepare()

    assert sandbox._backend == "bwrap"
    wrapped = sandbox.wrap_command(["/bin/bash", "-i"])
    # The container root is mounted read-only before the declared rw path so
    # untracked paths (e.g. a sibling /workspace/models) are denied instead of
    # landing on an implicit writable sandbox root.
    assert _option_index(wrapped, "--ro-bind", "/") < _option_index(
        wrapped, "--bind", str(public_data_dir)
    )
    assert "--tmpfs" not in wrapped
    private_binds = [
        wrapped[i + 1]
        for i in range(len(wrapped) - 1)
        if wrapped[i] == "--bind"
        and wrapped[i + 1] in (str(sandbox._tmp_dir), str(sandbox._sandbox_tmp))
    ]
    assert private_binds == []


def test_without_conversation_policy_keeps_sandbox_root_writable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: False
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_bwrap_usable", lambda: True
    )

    sandbox = TerminalSandbox(str(tmp_path), "required")
    sandbox.prepare()

    assert sandbox._backend == "bwrap"
    wrapped = sandbox.wrap_command(["/bin/bash", "-i"])
    assert ("--ro-bind", "/") not in [
        (wrapped[i], wrapped[i + 1]) for i in range(len(wrapped) - 1)
    ]


def test_conversation_policy_uses_landlock_when_bwrap_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: True
    )
    monkeypatch.setattr("openhands.tools.terminal.sandbox.shutil.which", lambda _: None)

    class FakeLandlock:
        def __init__(self, *, strict: bool):
            pass

        def __getattr__(self, name: str):
            return lambda *a, **kw: self

    monkeypatch.setitem(
        __import__("sys").modules,
        "py_landlock",
        SimpleNamespace(Landlock=FakeLandlock),
    )

    events_dir = tmp_path / "events"
    public_data_dir = tmp_path / "public_data"
    events_dir.mkdir()
    public_data_dir.mkdir()
    sandbox = TerminalSandbox(
        str(tmp_path),
        "required",
        read_only_paths=(str(events_dir),),
        read_write_paths=(str(public_data_dir),),
    )
    sandbox.prepare()

    assert sandbox._backend == "landlock"
    wrapped = sandbox.wrap_command(["/bin/bash", "-i"])
    assert wrapped[0] == str(sandbox._landlock_wrapper)
    assert wrapped[1] == sys.executable
    assert wrapped[2:5] == ["aa-exec", "-p", APPARMOR_PROFILE_NAME]
    assert wrapped[-2:] == ["/bin/bash", "-i"]


def test_conversation_policy_falls_back_to_apparmor_when_no_landlock_or_bwrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: True
    )
    monkeypatch.setattr("openhands.tools.terminal.sandbox.shutil.which", lambda _: None)
    import builtins

    real_import = builtins.__import__

    def _raise(name, *a, **kw):
        if name == "py_landlock":
            raise ImportError("no py-landlock")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _raise)

    events_dir = tmp_path / "events"
    events_dir.mkdir()
    sandbox = TerminalSandbox(
        str(tmp_path),
        "required",
        read_only_paths=(str(events_dir),),
    )
    sandbox.prepare()

    assert sandbox._backend == "apparmor"
    wrapped = sandbox.wrap_command(["/bin/bash", "-i"])
    assert wrapped[:3] == ["aa-exec", "-p", APPARMOR_PROFILE_NAME]


def test_landlock_skipped_in_pyinstaller_frozen_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: False
    )
    monkeypatch.setattr("openhands.tools.terminal.sandbox.shutil.which", lambda _: None)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    class FakeLandlock:
        def __init__(self, *, strict: bool):
            pass

        def __getattr__(self, name: str):
            return lambda *a, **kw: self

    monkeypatch.setitem(
        sys.modules,
        "py_landlock",
        SimpleNamespace(Landlock=FakeLandlock),
    )

    required_sandbox = TerminalSandbox(str(tmp_path), "required")
    with pytest.raises(RuntimeError, match="PyInstaller frozen mode"):
        required_sandbox.prepare()
    assert required_sandbox._backend is None
    assert required_sandbox._landlock_wrapper is None

    auto_sandbox = TerminalSandbox(str(tmp_path), "auto")
    auto_sandbox.prepare()
    assert auto_sandbox._backend is None
    assert auto_sandbox._landlock_wrapper is None


def test_landlock_skipped_in_pyinstaller_frozen_mode_falls_back_to_apparmor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: True
    )
    monkeypatch.setattr("openhands.tools.terminal.sandbox.shutil.which", lambda _: None)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    class FakeLandlock:
        def __init__(self, *, strict: bool):
            pass

        def __getattr__(self, name: str):
            return lambda *a, **kw: self

    monkeypatch.setitem(
        sys.modules,
        "py_landlock",
        SimpleNamespace(Landlock=FakeLandlock),
    )

    sandbox = TerminalSandbox(str(tmp_path), "required")
    sandbox.prepare()
    assert sandbox._backend == "apparmor"
    assert sandbox._landlock_wrapper is None


def test_bwrap_smoke_test_failure_falls_back_to_apparmor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: True
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.shutil.which",
        lambda name: "/usr/bin/bwrap" if name == "bwrap" else None,
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_bwrap_usable", lambda: False
    )

    sandbox = TerminalSandbox(str(tmp_path), "required")
    sandbox.prepare()

    assert sandbox._backend == "apparmor"
