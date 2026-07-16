"""Shared utilities."""

import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

PUBLIC_READ_ALIASES: tuple[tuple[str, str, str, str | None], ...] = (
    ("knowledge", "PYROMIND_KNOWLEDGE_BASE_PATH", "knowledge", None),
    (".agents/skills", "PYROMIND_SKILLS_PATH", "skills", ".agents"),
)

PathOperation = Literal["read", "write", "execute"]


@dataclass(frozen=True)
class PathRule:
    """A filesystem root and the operations permitted below it."""

    path: Path
    perm: frozenset[PathOperation]
    recursive: bool = True

    @classmethod
    def create(
        cls,
        path: str | Path,
        perm: str,
        recursive: bool = True,
    ) -> "PathRule":
        invalid = set(perm) - {"r", "w", "x"}
        if invalid:
            raise ValueError(f"Unsupported path permissions: {sorted(invalid)}")
        allowed: set[PathOperation] = set()
        permission_markers: tuple[tuple[PathOperation, str], ...] = (
            ("read", "r"),
            ("write", "w"),
            ("execute", "x"),
        )
        for operation, marker in permission_markers:
            if marker in perm:
                allowed.add(operation)
        return cls(
            path=Path(path).expanduser().resolve(),
            perm=frozenset(allowed),
            recursive=recursive,
        )


class PathAccessPolicy:
    """Resolve and authorize filesystem paths against ordered allow rules."""

    def __init__(self, rules: Sequence[PathRule]):
        self.rules = tuple(rules)

    def check(self, target_path: str | Path, operation: PathOperation) -> bool:
        target = Path(target_path).expanduser().resolve()
        for rule in self.rules:
            if rule.recursive:
                in_rule = target == rule.path or target.is_relative_to(rule.path)
            else:
                in_rule = target == rule.path or target.parent == rule.path
            if in_rule:
                return operation in rule.perm
        return False

    def require(self, target_path: str | Path, operation: PathOperation) -> Path:
        target = Path(target_path).expanduser().resolve()
        if not self.check(target, operation):
            raise PermissionError(f"Path is not allowed for {operation}: {target}")
        return target


def resolve_workspace_subpath(subpath: str | Path, workspace_root: Path) -> Path:
    """Resolve *subpath* against *workspace_root*.

    Absolute paths are resolved directly. Relative paths are joined with
    *workspace_root* before resolving.
    """
    p = Path(subpath)
    return (p if p.is_absolute() else workspace_root / p).resolve()


def default_path_access_policy(
    workspace_dir: str | Path,
    read_only_roots: Sequence[str | Path] = (),
    workspace_read_only_subpaths: Sequence[str | Path] = (),
    workspace_read_write_subpaths: Sequence[str | Path] = (),
    exclude_workspace_fallback: bool = False,
) -> PathAccessPolicy:
    """Build the standard workspace plus public-read path policy.

    ``read_only_roots`` are absolute roots outside the workspace that the agent
    may read (knowledge base, skills, etc.).

    ``workspace_read_only_subpaths`` / ``workspace_read_write_subpaths`` are
    paths relative to ``workspace_dir`` registered with ``r`` / ``rw``
    permissions before the workspace ``rwx`` fallback rule. Use these together
    with ``exclude_workspace_fallback=True`` to model a conversation workspace
    where only a handful of subdirectories are exposed to the agent (for
    example, ``public_data/`` is read-write and ``events/`` is read-only,
    while ``meta.json`` / ``base_state.json` are off-limits).

    When ``exclude_workspace_fallback`` is True, the workspace ``rwx`` rule is
    omitted entirely; paths outside any declared subpath will be denied by
    :class:`PathAccessPolicy` (first-match, no-match = deny).

    When the caller does not pass explicit subpaths or ``exclude_workspace_fallback``,
    the function auto-detects conversation workspaces (via
    :func:`is_conversation_workspace`) and applies the standard conversation
    permission matrix automatically. This ensures all tools get correct
    path restrictions without per-tool conversation detection.
    """
    caller_specified = (
        workspace_read_only_subpaths
        or workspace_read_write_subpaths
        or exclude_workspace_fallback
    )
    if not caller_specified and is_conversation_workspace(workspace_dir):
        workspace_read_only_subpaths = CONVERSATION_READ_ONLY_SUBPATHS
        workspace_read_write_subpaths = CONVERSATION_READ_WRITE_SUBPATHS
        exclude_workspace_fallback = True

    rules: list[PathRule] = [PathRule.create(root, "r") for root in read_only_roots]
    workspace_root = Path(workspace_dir).expanduser().resolve()
    for subpath, perm in (
        *((s, "r") for s in workspace_read_only_subpaths),
        *((s, "rw") for s in workspace_read_write_subpaths),
    ):
        target = resolve_workspace_subpath(subpath, workspace_root)
        rules.append(PathRule.create(target, perm))
    if not exclude_workspace_fallback:
        rules.append(PathRule.create(workspace_root, "rwx"))
    return PathAccessPolicy(rules)


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


WORKFLOW_SUBPATH: Final[str] = "workflow"
EVENTS_SUBPATH: Final[str] = "events"
PUBLIC_DATA_SUBPATH: Final[str] = "public_data"

CONVERSATION_READ_WRITE_SUBPATHS: tuple[str, ...] = (PUBLIC_DATA_SUBPATH,)
CONVERSATION_READ_ONLY_SUBPATHS: tuple[str, ...] = (EVENTS_SUBPATH,)


def is_conversation_workspace(workspace_dir: str | Path) -> bool:
    """Return True when ``workspace_dir`` looks like a conversation workspace.

    Detection uses two heuristics (either is sufficient):
    1. The directory path matches ``…/workspace/conversations/<id>`` — this
       works even before the conversation starts and ``events/`` is created.
    2. An ``events/`` subdirectory exists — this works for already-running
       conversations regardless of path layout.
    """
    resolved = Path(workspace_dir).resolve()
    if (resolved / EVENTS_SUBPATH).is_dir():
        return True
    return resolved.parent.name == "conversations" and bool(resolved.name)


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
