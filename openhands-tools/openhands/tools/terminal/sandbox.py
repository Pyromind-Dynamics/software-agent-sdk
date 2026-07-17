"""Kernel-backed filesystem isolation for terminal child processes."""

from __future__ import annotations

import os
import platform
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from openhands.sdk.logger import get_logger
from openhands.tools.utils import resolve_workspace_subpath


logger = get_logger(__name__)

TerminalSandboxMode = Literal["off", "auto", "required"]
TERMINAL_SANDBOX_ENV = "OH_TERMINAL_SANDBOX"
PUBLIC_READ_ROOTS = (
    "/agent-server/knowledge",
    "/agent-server/.agents/skills",
)


def terminal_sandbox_mode() -> TerminalSandboxMode:
    """Return the configured terminal sandbox mode.

    Linux and macOS use a required kernel-backed sandbox by default. Windows
    remains disabled because this module has no Windows backend.
    """
    default: TerminalSandboxMode = (
        "required" if platform.system() in {"Linux", "Darwin"} else "off"
    )
    value = os.environ.get(TERMINAL_SANDBOX_ENV, default).lower()
    if value not in {"off", "auto", "required"}:
        raise ValueError(
            f"{TERMINAL_SANDBOX_ENV} must be one of: off, auto, required; got {value!r}"
        )
    return cast(TerminalSandboxMode, value)


def terminal_sandbox_enabled(mode: TerminalSandboxMode) -> bool:
    """Return whether this mode requires an isolated Unix terminal backend."""
    return mode != "off" and platform.system() in {"Linux", "Darwin"}


class TerminalSandbox:
    """Apply a platform-specific kernel policy before starting a shell.

    Backend selection (Linux):
      1. ``bwrap`` (Bubblewrap) — user-namespace-based, no kernel feature required
      2. ``py_landlock`` — Landlock LSM (Linux 5.13+ kernel support needed)

    macOS always uses ``sandbox-exec`` (Seatbelt).
    """

    _backend: Literal["bwrap", "landlock", "seatbelt"] | None

    def __init__(
        self,
        work_dir: str,
        mode: TerminalSandboxMode,
        *,
        read_only_paths: tuple[str, ...] = (),
        read_write_paths: tuple[str, ...] | None = None,
    ):
        resolved_work_dir = Path(work_dir).resolve()
        self.work_dir = resolved_work_dir
        self.mode: TerminalSandboxMode = mode
        self._tmp_dir = resolved_work_dir / ".openhands-tmp"
        self.read_only_paths = tuple(
            resolve_workspace_subpath(p, resolved_work_dir) for p in read_only_paths
        )
        rw_paths = (
            read_write_paths if read_write_paths is not None else (resolved_work_dir,)
        )
        self.read_write_paths = tuple(
            resolve_workspace_subpath(p, resolved_work_dir) for p in rw_paths
        )
        self._backend = None
        self._landlock_factory: Any | None = None
        self._seatbelt_profile: Path | None = None

    def prepare(self) -> None:
        """Probe available sandbox backends and prepare the chosen one."""
        if not terminal_sandbox_enabled(self.mode):
            return
        self._tmp_dir.mkdir(mode=0o700, exist_ok=True)
        if platform.system() == "Darwin":
            sandbox_exec = shutil.which("sandbox-exec")
            if sandbox_exec is None:
                if self.mode == "required":
                    raise RuntimeError(
                        "Terminal sandbox is required, but sandbox-exec is unavailable"
                    )
                logger.warning(
                    "sandbox-exec is unavailable; terminal sandbox is disabled"
                )
                return
            self._seatbelt_profile = self._tmp_dir / ".openhands-seatbelt.sb"
            self._seatbelt_profile.write_text(self._build_seatbelt_profile())
            self._backend = "seatbelt"
            return
        bwrap_path = shutil.which("bwrap")
        if bwrap_path is not None:
            self._backend = "bwrap"
            return
        try:
            landlock_module: Any = import_module("py_landlock")
            self._landlock_factory = landlock_module.Landlock
            self._backend = "landlock"
        except ImportError as exc:
            if self.mode == "required":
                raise RuntimeError(
                    "Terminal sandbox is required, but neither bwrap "
                    "nor py-landlock is available"
                ) from exc
            logger.warning("no sandbox backend available")

    def apply(self) -> None:
        """Apply the policy in the child process, failing closed when required."""
        if self._backend != "landlock":
            return
        if not terminal_sandbox_enabled(self.mode):
            return
        landlock = self._landlock_factory
        if landlock is None:
            if self.mode == "required":
                raise RuntimeError("Terminal sandbox was not prepared")
            return

        try:
            system_read_paths = tuple(
                path
                for path in ("/usr", "/etc", "/lib", "/lib64", "/bin", "/sbin", "/dev")
                if Path(path).exists()
            )
            public_read_paths = tuple(
                path for path in PUBLIC_READ_ROOTS if Path(path).exists()
            )
            executable_paths = tuple(
                path for path in ("/usr", "/bin", "/sbin") if Path(path).exists()
            )
            (
                landlock(strict=True)
                .allow_read(*system_read_paths)
                .allow_read(*public_read_paths, *map(str, self.read_only_paths))
                .allow_write("/dev/null", "/dev/tty")
                .allow_execute(*executable_paths)
                .allow_read_write(str(self._tmp_dir), *map(str, self.read_write_paths))
                .apply()
            )
        except Exception as exc:
            if self.mode == "required":
                raise RuntimeError(
                    "Failed to apply the terminal Landlock policy"
                ) from exc
            logger.warning("Failed to apply terminal Landlock policy: %s", exc)

    def wrap_command(self, command: list[str]) -> list[str]:
        """Wrap a command with the platform-specific sandbox launcher."""
        if self._backend == "bwrap":
            return self._build_bwrap_args() + command
        if self._backend == "seatbelt":
            sandbox_exec = shutil.which("sandbox-exec")
            if sandbox_exec is None:
                raise RuntimeError(
                    "sandbox-exec became unavailable after profile creation"
                )
            return [
                sandbox_exec,
                "-f",
                str(self._seatbelt_profile),
                "--",
                *command,
            ]
        return command

    def cleanup(self) -> None:
        """Remove the generated macOS profile after the shell exits."""
        if self._seatbelt_profile is not None:
            self._seatbelt_profile.unlink(missing_ok=True)

    def _build_bwrap_args(self) -> list[str]:
        args = ["bwrap", "--unshare-ipc", "--unshare-uts"]
        for path in ("/usr", "/etc", "/lib", "/lib64", "/bin", "/sbin"):
            if Path(path).exists():
                args.extend(["--ro-bind", path, path])
        for path in PUBLIC_READ_ROOTS:
            if Path(path).exists():
                args.extend(["--ro-bind", path, path])
        for path in self.read_only_paths:
            if path.exists():
                args.extend(["--ro-bind", str(path), str(path)])
        args.extend(["--bind", str(self._tmp_dir), str(self._tmp_dir)])
        for path in self.read_write_paths:
            args.extend(["--bind", str(path), str(path)])
        args.extend(["--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"])
        return args

    def _build_seatbelt_profile(self) -> str:
        parent = self._seatbelt_path(self.work_dir.parent)
        return "\n".join(
            [
                "(version 1)",
                "(allow default)",
                f'(deny file-read* (subpath "{parent}"))',
                *(
                    f'(allow file-read* (subpath "{self._seatbelt_path(path)}"))'
                    for path in (*self.read_write_paths, *self.read_only_paths)
                ),
                *(
                    f'(allow file-read* (subpath "{self._seatbelt_path(Path(path))}"))'
                    for path in PUBLIC_READ_ROOTS
                ),
                "(deny file-write*)",
                *(
                    f'(allow file-write* (subpath "{self._seatbelt_path(path)}"))'
                    for path in self.read_write_paths
                ),
                "",
            ]
        )

    @staticmethod
    def _seatbelt_path(path: Path) -> str:
        return str(path).replace("\\", "\\\\").replace('"', '\\"')
