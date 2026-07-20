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

# Name of the AppArmor profile loaded into the kernel at image build time
# by `apparmor_parser -r -W /etc/apparmor.d/openhands-agent-terminal`.
# The profile file ships under openhands/tools/terminal/apparmor/ and is
# copied into /etc/apparmor.d/ by the Dockerfile.
APPARMOR_PROFILE_NAME = "openhands-agent-terminal"


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


def _is_apparmor_available() -> bool:
    """Return whether the AppArmor terminal sandbox backend is usable.

    Requires three conditions, all checkable without privilege:
      1. ``aa-exec`` is on PATH (ships with ``apparmor-utils``).
      2. The kernel has the AppArmor LSM active (``/sys/kernel/security/lsm``
         contains ``apparmor``).
      3. The ``openhands-agent-terminal`` profile file is installed under
         ``/etc/apparmor.d/``. The Dockerfile loads it into the kernel at
         image build time via ``apparmor_parser``; we do not reload at
         runtime to avoid requiring CAP_MAC_ADMIN on the pod.
    """
    if platform.system() != "Linux":
        return False
    if shutil.which("aa-exec") is None:
        return False
    try:
        lsm = Path("/sys/kernel/security/lsm").read_text(
            encoding="utf-8", errors="ignore"
        )
    except OSError:
        return False
    if "apparmor" not in lsm.split(","):
        return False
    profile_path = Path(f"/etc/apparmor.d/{APPARMOR_PROFILE_NAME}")
    return profile_path.is_file()


class TerminalSandbox:
    """Apply a platform-specific kernel policy before starting a shell.

    Backend selection (Linux):
      1. ``apparmor`` — LSM-based, no capability/namespace required. Requires
         ``aa-exec`` and a pre-loaded ``openhands-agent-terminal`` profile
         (the Dockerfile loads it at image build time via ``apparmor_parser``).
      2. ``bwrap`` (Bubblewrap) — user-namespace-based, no kernel feature required
      3. ``py_landlock`` — Landlock LSM (Linux 5.13+ kernel support needed)

    macOS always uses ``sandbox-exec`` (Seatbelt).
    """

    _backend: Literal["apparmor", "bwrap", "landlock", "seatbelt"] | None

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
        self._apparmor_available: bool = False

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
        # Linux backend selection.
        #
        # When the caller passed conversation-scoped ``read_only_paths`` or
        # ``read_write_paths``, the sandbox is meant to enforce a per-conversation
        # PathAccessPolicy. Landlock and bwrap can express that by allowing
        # ``events/`` read-only and ``public_data/`` read-write. AppArmor uses a
        # single global denylist loaded at image build time, so use it only as a
        # fallback when neither per-conversation backend is available.
        #
        # Without a per-conversation policy, the default order stands:
        #   AppArmor (no capability / namespace) > bwrap > Landlock.
        has_conversation_policy = bool(
            self.read_only_paths or self.read_write_paths != (self.work_dir,)
        )
        backend_chosen = False

        # Conversation-scoped policy → prefer Landlock (per-conversation
        # semantics) over AppArmor (global denylist).  Stack AppArmor on top
        # when both are available for defense-in-depth.
        if has_conversation_policy:
            self._apparmor_available = _is_apparmor_available()
            try:
                landlock_module = import_module("py_landlock")
                self._landlock_factory = landlock_module.Landlock
                self._backend = "landlock"
                backend_chosen = True
                logger.info(
                    "Using Landlock terminal sandbox for per-conversation policy%s",
                    " (+ AppArmor as defense-in-depth)"
                    if self._apparmor_available
                    else "",
                )
            except ImportError:
                logger.warning(
                    "Per-conversation policy requested but py-landlock is not "
                    "importable; falling back to bwrap / AppArmor"
                )
                if shutil.which("bwrap") is None and self._apparmor_available:
                    self._backend = "apparmor"
                    backend_chosen = True
                    logger.info(
                        "Using AppArmor terminal sandbox (profile=%s) as fallback",
                        APPARMOR_PROFILE_NAME,
                    )

        # No conversation policy → keep the legacy order: AppArmor first.
        if not backend_chosen and not has_conversation_policy:
            if _is_apparmor_available():
                self._backend = "apparmor"
                self._apparmor_available = True
                backend_chosen = True
                logger.info(
                    "Using AppArmor terminal sandbox (profile=%s)",
                    APPARMOR_PROFILE_NAME,
                )

        if not backend_chosen:
            bwrap_path = shutil.which("bwrap")
            if bwrap_path is not None:
                self._backend = "bwrap"
                backend_chosen = True

        if not backend_chosen:
            try:
                landlock_module = import_module("py_landlock")
                self._landlock_factory = landlock_module.Landlock
                self._backend = "landlock"
                backend_chosen = True
            except ImportError as exc:
                if self.mode == "required":
                    raise RuntimeError(
                        "Terminal sandbox is required, but no backend is available "
                        "(AppArmor profile not loaded, bwrap not on PATH, "
                        "py-landlock not importable)"
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
        # Landlock is applied via ``apply()`` in the child preexec; it does not
        # need a wrapper process.  But when AppArmor is also loaded on the host,
        # we additionally prepend ``aa-exec`` so the child runs under both LSMs
        # (Landlock for the per-conversation policy, AppArmor for the broader
        # denylist of high-value paths and privilege-escalation tools).
        if self._backend == "landlock":
            if self._apparmor_available:
                return [
                    "aa-exec",
                    "-p",
                    APPARMOR_PROFILE_NAME,
                    "--",
                    *command,
                ]
            return command
        if self._backend == "apparmor":
            return [
                "aa-exec",
                "-p",
                APPARMOR_PROFILE_NAME,
                "--",
                *command,
            ]
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
        args.extend(["--bind", str(self._tmp_dir), str(self._tmp_dir)])
        for path in self.read_write_paths:
            args.extend(["--bind", str(path), str(path)])
        for path in self.read_only_paths:
            if path.exists():
                args.extend(["--ro-bind", str(path), str(path)])
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
