"""Grep tool executor implementation."""

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolExecutor
from openhands.sdk.utils import sanitized_env


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
from openhands.tools.grep.definition import GrepAction, GrepMatch, GrepObservation
from openhands.tools.utils import (
    _check_grep_available,
    _check_ripgrep_available,
    _log_ripgrep_fallback_warning,
)


logger = get_logger(__name__)


class GrepExecutor(ToolExecutor[GrepAction, GrepObservation]):
    """Executor for grep content search operations.

    This implementation prefers ripgrep for performance, falls back to the
    system grep binary when available, and finally uses a Python recursive
    search when no grep binary is installed.
    """

    _MAX_MATCHES = 100

    def __init__(self, working_dir: str):
        """Initialize the grep executor.

        Args:
            working_dir: The working directory to use as the base for searches
        """
        self.working_dir: Path = Path(working_dir).resolve()
        self._search_backend = self._select_search_backend()

        if self._search_backend == "grep":
            _log_ripgrep_fallback_warning("grep", "system grep")
        elif self._search_backend == "python":
            _log_ripgrep_fallback_warning("grep", "system grep, then Python search")

    def _select_search_backend(self) -> str:
        if _check_ripgrep_available():
            return "ripgrep"
        if _check_grep_available():
            return "grep"
        return "python"

    def __call__(
        self,
        action: GrepAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> GrepObservation:
        """Execute grep content search using the best available backend."""
        try:
            if action.path:
                search_path = Path(action.path).resolve()
                if not search_path.is_dir():
                    return GrepObservation.from_text(
                        text=f"Search path '{action.path}' is not a valid directory",
                        matches=[],
                        pattern=action.pattern,
                        search_path=str(search_path),
                        include_pattern=action.include,
                        is_error=True,
                    )
            else:
                search_path = self.working_dir

            try:
                regex = re.compile(action.pattern, re.IGNORECASE)
            except re.error as e:
                return GrepObservation.from_text(
                    text=f"Invalid regex pattern: {e}",
                    matches=[],
                    pattern=action.pattern,
                    search_path=str(search_path),
                    include_pattern=action.include,
                    is_error=True,
                )

            if self._search_backend == "ripgrep":
                return self._execute_with_ripgrep(action, search_path)
            if self._search_backend == "grep":
                return self._execute_with_system_grep(action, search_path)
            return self._execute_with_python_search(action, search_path, regex)

        except Exception as e:
            try:
                if action.path:
                    error_search_path = str(Path(action.path).resolve())
                else:
                    error_search_path = str(self.working_dir)
            except Exception:
                error_search_path = "unknown"

            return GrepObservation.from_text(
                text=str(e),
                matches=[],
                pattern=action.pattern,
                search_path=error_search_path,
                include_pattern=action.include,
                is_error=True,
            )

    def _format_output(
        self,
        matches: list[GrepMatch],
        pattern: str,
        search_path: str,
        include_pattern: str | None,
        truncated: bool,
    ) -> str:
        """Format the grep observation output message."""
        include_info = (
            f" (filtered by '{include_pattern}')" if include_pattern else ""
        )
        if not matches:
            return (
                f"No matches found for pattern '{pattern}' "
                f"in directory '{search_path}'{include_info}"
            )

        match_lines = "\n".join(
            f"{m.file_path}:{m.line_number}: {m.line}" for m in matches
        )
        output = (
            f"Found {len(matches)} match(es) for pattern "
            f"'{pattern}' in '{search_path}'{include_info}:\n{match_lines}"
        )
        if truncated:
            output += (
                "\n\n[Results truncated to the first 100 matches. "
                "Consider using a more specific pattern.]"
            )
        return output

    def _path_matches_filters(
        self,
        path: Path,
        search_path: Path,
        include_pattern: str | None,
    ) -> bool:
        """Return whether a matched path should be surfaced to the user."""
        try:
            relative_parts = path.resolve().relative_to(search_path.resolve()).parts
        except ValueError:
            relative_parts = (path.name,)

        if any(part.startswith(".") for part in relative_parts[:-1]):
            return False

        filename = relative_parts[-1] if relative_parts else path.name
        if include_pattern:
            return fnmatch.fnmatch(filename, include_pattern)
        return not filename.startswith(".")

    def _match_mtime(self, path: Path) -> float:
        """Return a sortable modification time for matched paths."""
        try:
            return path.stat().st_mtime
        except OSError:
            return float("-inf")

    def _finalize_matches(
        self,
        matches: list[GrepMatch],
        search_path: Path,
        include_pattern: str | None,
    ) -> tuple[list[GrepMatch], bool]:
        """Filter, deduplicate, sort, and truncate raw line matches."""
        unique_matches: dict[tuple[str, int], GrepMatch] = {}
        for match in matches:
            try:
                resolved = Path(match.file_path).resolve()
            except OSError:
                continue
            if not self._path_matches_filters(resolved, search_path, include_pattern):
                continue
            key = (str(resolved), match.line_number)
            if key in unique_matches:
                continue
            unique_matches[key] = GrepMatch(
                file_path=str(resolved),
                line_number=match.line_number,
                line=match.line,
            )

        # Sort by file modification time (newest first), then line number ascending.
        sorted_matches = sorted(
            unique_matches.values(),
            key=lambda m: (-self._match_mtime(Path(m.file_path)), m.line_number),
        )
        truncated = len(sorted_matches) > self._MAX_MATCHES
        return sorted_matches[: self._MAX_MATCHES], truncated

    def _build_observation(
        self,
        action: GrepAction,
        search_path: Path,
        matches: list[GrepMatch],
    ) -> GrepObservation:
        finalized_matches, truncated = self._finalize_matches(
            matches,
            search_path,
            action.include,
        )
        output = self._format_output(
            matches=finalized_matches,
            pattern=action.pattern,
            search_path=str(search_path),
            include_pattern=action.include,
            truncated=truncated,
        )
        return GrepObservation.from_text(
            text=output,
            matches=finalized_matches,
            pattern=action.pattern,
            search_path=str(search_path),
            include_pattern=action.include,
            truncated=truncated,
        )

    def _parse_grep_lines(self, stdout: str) -> list[GrepMatch]:
        """Parse ``path:line_number:content`` output into GrepMatch entries."""
        matches: list[GrepMatch] = []
        if not stdout:
            return matches
        for raw_line in stdout.splitlines():
            if not raw_line:
                continue
            parts = raw_line.split(":", 2)
            if len(parts) < 3:
                continue
            file_path, line_no_str, content = parts
            try:
                line_number = int(line_no_str)
            except ValueError:
                continue
            matches.append(
                GrepMatch(
                    file_path=file_path,
                    line_number=line_number,
                    line=content,
                )
            )
        return matches

    def _execute_with_ripgrep(
        self, action: GrepAction, search_path: Path
    ) -> GrepObservation:
        """Execute grep content search using ripgrep."""
        cmd = [
            "rg",
            "--line-number",
            "--no-heading",
            "--with-filename",
            "--color=never",
            "-i",
            action.pattern,
            str(search_path),
            "--sortr=modified",
        ]
        if action.include:
            cmd.extend(["-g", action.include])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=sanitized_env(),
        )

        matches = self._parse_grep_lines(result.stdout)
        return self._build_observation(action, search_path, matches)

    def _execute_with_system_grep(
        self, action: GrepAction, search_path: Path
    ) -> GrepObservation:
        """Execute grep content search using the system grep binary."""
        result = subprocess.run(
            ["grep", "-R", "-I", "-n", "-i", action.pattern, str(search_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=sanitized_env(),
        )
        if result.returncode not in (0, 1):
            logger.warning(
                "grep backend failed with exit code %s; falling back to Python search",
                result.returncode,
            )
            return self._execute_with_python_search(action, search_path)

        matches = self._parse_grep_lines(result.stdout)
        return self._build_observation(action, search_path, matches)

    def _execute_with_python_search(
        self,
        action: GrepAction,
        search_path: Path,
        regex: re.Pattern[str] | None = None,
    ) -> GrepObservation:
        """Execute grep content search using Python file walking."""
        compiled_regex = regex or re.compile(action.pattern, re.IGNORECASE)
        matches: list[GrepMatch] = []
        for root, dirs, files in os.walk(search_path):
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            for filename in files:
                file_path = Path(root) / filename
                if not self._path_matches_filters(
                    file_path, search_path, action.include
                ):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for line_number, line in enumerate(content.splitlines(), start=1):
                    if compiled_regex.search(line):
                        matches.append(
                            GrepMatch(
                                file_path=str(file_path),
                                line_number=line_number,
                                line=line,
                            )
                        )

        return self._build_observation(action, search_path, matches)
