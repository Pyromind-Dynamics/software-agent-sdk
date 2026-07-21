from pathlib import Path
from types import SimpleNamespace

import pytest

from openhands.tools.terminal.sandbox import (
    APPARMOR_PROFILE_NAME,
    TerminalSandbox,
    terminal_sandbox_enabled,
    terminal_sandbox_mode,
)


def _option_index(args: list[str], option: str, value: str) -> int:
    for index, arg in enumerate(args[:-1]):
        if arg == option and args[index + 1] == value:
            return index
    raise AssertionError(f"{option} {value} not found in {args}")


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
    calls: list[tuple[str, tuple[str, ...]]] = []

    class FakeLandlock:
        def __init__(self, *, strict: bool):
            assert strict

        def __getattr__(self, name: str):
            def record(*paths: str):
                calls.append((name, paths))
                return self

            return record

    monkeypatch.setitem(
        __import__("sys").modules,
        "py_landlock",
        SimpleNamespace(Landlock=FakeLandlock),
    )
    sandbox = TerminalSandbox(str(tmp_path), "required")
    sandbox.prepare()

    sandbox.apply()

    assert any(
        name == "allow_read_write" and str(tmp_path) in paths for name, paths in calls
    )
    assert any(name == "allow_execute" for name, _ in calls)


def test_terminal_sandbox_uses_explicit_read_and_write_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "openhands.tools.terminal.sandbox._is_apparmor_available", lambda: False
    )
    calls: list[tuple[str, tuple[str, ...]]] = []

    class FakeLandlock:
        def __init__(self, *, strict: bool):
            assert strict

        def __getattr__(self, name: str):
            def record(*paths: str):
                calls.append((name, paths))
                return self

            return record

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
    sandbox.apply()

    read_write_calls = [paths for name, paths in calls if name == "allow_read_write"]
    read_calls = [paths for name, paths in calls if name == "allow_read"]
    assert any(str(workflow_dir) in paths for paths in read_write_calls)
    assert any(str(events_dir) in paths for paths in read_calls)


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
    assert f'(allow file-read* (subpath "{current}"))' in profile
    assert '(allow file-read* (subpath "/agent-server/knowledge"))' in profile
    assert '(allow file-read* (subpath "/agent-server/.agents/skills"))' in profile
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
    assert f'(allow file-read* (subpath "{events_path}"))' in profile
    assert f'(allow file-read* (subpath "{public_data_path}"))' in profile
    meta_path = str(conv_dir / "meta.json")
    assert f'(allow file-read* (subpath "{meta_path}"))' not in profile
    assert f'(allow file-read* (subpath "{conv_dir}"))' not in profile


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
        "openhands.tools.terminal.sandbox.shutil.which",
        lambda name: "/usr/bin/bwrap" if name == "bwrap" else None,
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
        "openhands.tools.terminal.sandbox.shutil.which",
        lambda name: "/usr/bin/bwrap" if name == "bwrap" else None,
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
    assert wrapped[:3] == ["aa-exec", "-p", APPARMOR_PROFILE_NAME]
    assert wrapped[-2:] == ["--", "/bin/bash", "-i"][-2:]


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
