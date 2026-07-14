"""Shared utilities."""

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

PUBLIC_READ_ALIASES: tuple[tuple[str, str, str, str | None], ...] = (
    ("knowledge", "PYROMIND_KNOWLEDGE_BASE_PATH", "knowledge", None),
    (".agents/skills", "PYROMIND_SKILLS_PATH", "skills", ".agents"),
)

_PUBLIC_READ_ALIAS_PATTERN = re.compile(
    r"(?<![\w.-])(?:knowledge|\.agents/skills)(?=(?:/|\b))"
)


def terminal_public_read_block_reason(command: str) -> str | None:
    """Return a safe redirect when terminal targets a public read-only root."""
    normalized_command = command.replace("\\", "/")
    if _PUBLIC_READ_ALIAS_PATTERN.search(normalized_command):
        return (
            "Public knowledge and skill documents are read-only. Use `grep` "
            "with the logical `knowledge/` or `.agents/skills/` path, then "
            'use `file_editor` with `command="view"`. Do not use terminal '
            "or a host filesystem path."
        )

    for root in configured_public_read_roots():
        if str(root).replace("\\", "/") in normalized_command:
            return (
                "Public knowledge and skill documents must be accessed through "
                "the logical `knowledge/` or `.agents/skills/` path with `grep` "
                "or `file_editor.view`; host filesystem paths are not exposed."
            )
    return None


def configured_public_read_roots(
    read_only_roots: list[str] | None = None,
) -> tuple[Path, ...]:
    """Return configured read-only roots without exposing them to the model."""
    if read_only_roots is not None:
        roots = read_only_roots
    else:
        roots = [
            *(
                os.environ.get(environment_variable, "")
                for _, environment_variable, _, _ in PUBLIC_READ_ALIASES
            ),
            *os.environ.get("PYROMIND_PUBLIC_READ_PATHS", "").split(os.pathsep),
        ]
    resolved_roots = (Path(root).resolve() for root in roots if root)
    return tuple(dict.fromkeys(resolved_roots))


def _public_alias_root(alias: str, roots: tuple[Path, ...]) -> Path | None:
    for configured_alias, _, root_name, parent_name in PUBLIC_READ_ALIASES:
        if alias != configured_alias:
            continue
        for root in roots:
            if root.name == root_name and (
                parent_name is None or root.parent.name == parent_name
            ):
                return root
    return None


def resolve_public_read_alias(
    path: str,
    roots: tuple[Path, ...],
) -> Path | None:
    """Resolve a logical public-read alias to its configured root."""
    candidate = Path(path)
    if candidate.is_absolute() or not candidate.parts:
        return None

    for alias, _, _, _ in PUBLIC_READ_ALIASES:
        alias_parts = Path(alias).parts
        if candidate.parts[: len(alias_parts)] != alias_parts:
            continue
        root = _public_alias_root(alias, roots)
        if root is not None:
            return (root / Path(*candidate.parts[len(alias_parts) :])).resolve()
    return None


def logical_public_read_path(path: Path, roots: tuple[Path, ...]) -> str:
    """Return a model-safe alias for a path under a configured public root."""
    resolved = path.resolve()
    for alias, _, _, _ in PUBLIC_READ_ALIASES:
        root = _public_alias_root(alias, roots)
        if root is not None and resolved.is_relative_to(root):
            return str(Path(alias) / resolved.relative_to(root))
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
