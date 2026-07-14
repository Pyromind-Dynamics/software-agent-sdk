"""Shared utilities."""

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


def configured_public_read_roots(
    read_only_roots: list[str] | None = None,
) -> tuple[Path, ...]:
    """Return configured read-only roots without exposing them to the model."""
    if read_only_roots is not None:
        roots = read_only_roots
    else:
        roots = [
            os.environ.get("PYROMIND_KNOWLEDGE_BASE_PATH", ""),
            *os.environ.get("PYROMIND_PUBLIC_READ_PATHS", "").split(os.pathsep),
        ]
    return tuple(Path(root).resolve() for root in roots if root)


def resolve_public_read_alias(
    path: str,
    roots: tuple[Path, ...],
) -> Path | None:
    """Resolve the ``knowledge/`` alias to the configured root."""
    candidate = Path(path)
    if candidate.is_absolute() or not candidate.parts:
        return None
    if candidate.parts[0] != "knowledge" or not roots:
        return None
    return (roots[0] / Path(*candidate.parts[1:])).resolve()


def logical_public_read_path(path: Path, roots: tuple[Path, ...]) -> str:
    """Return a model-safe alias for a path under a configured public root."""
    resolved = path.resolve()
    if roots and resolved.is_relative_to(roots[0]):
        return str(Path("knowledge") / resolved.relative_to(roots[0]))
    return str(resolved)


def _check_command_available(
    command: str,
    probe_args: Sequence[str] | None = ("--version",),
) -> bool:
    """Check if a command is available and optionally responds to a probe."""

    try:
        if shutil.which(command) is None:
            return False
        if probe_args is None:
            return True
        result = subprocess.run(
            [command, *probe_args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is available on the system."""

    return _check_command_available("rg")


def _check_grep_available() -> bool:
    """Check if grep is available on the system."""

    return _check_command_available("grep", probe_args=None)


def _log_ripgrep_fallback_warning(tool_name: str, fallback_method: str) -> None:
    """Log a warning about falling back from ripgrep to alternative method.

    Args:
        tool_name: Name of the tool (e.g., "glob", "grep")
        fallback_method: Description of the fallback method being used
    """
    logger.warning(
        f"{tool_name}: ripgrep (rg) not available. "
        f"Falling back to {fallback_method}. "
        f"For better performance, consider installing ripgrep: "
        f"https://github.com/BurntSushi/ripgrep#installation"
    )
