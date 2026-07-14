"""Grep tool implementation for fast content search."""

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)
from openhands.tools.utils import configured_public_read_roots


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class GrepAction(Action):
    """Schema for grep content search operations."""

    pattern: str = Field(description="The regex pattern to search for in file contents")
    path: str | None = Field(
        default=None,
        description=(
            "The directory or exact file to search. Absolute paths are accepted; "
            "relative paths resolve from the current working directory. Defaults "
            "to the current working directory."
        ),
    )
    include: str | None = Field(
        default=None,
        description=(
            "Optional file pattern to filter which files to search "
            '(e.g., "*.js", "*.{ts,tsx}")'
        ),
    )


class GrepMatch(BaseModel):
    """A single matching line found by grep."""

    model_config = ConfigDict(frozen=True)

    file_path: str = Field(description="Absolute path of the file containing the match")
    line_number: int = Field(
        description="1-based line number of the matching line within the file"
    )
    line: str = Field(description="The full text of the matching line")


class GrepObservation(Observation):
    """Observation from grep content search operations."""

    matches: list[GrepMatch] = Field(
        default_factory=list,
        description=(
            "Matching lines, each with its file path, 1-based line number, "
            "and line text"
        ),
    )
    pattern: str = Field(description="The regex pattern that was used")
    search_path: str = Field(description="The file or directory that was searched")
    include_pattern: str | None = Field(
        default=None, description="The file pattern filter that was used"
    )
    truncated: bool = Field(
        default=False,
        description="Whether results were truncated to the first 100 matches",
    )
    searched_files: int | None = Field(
        default=None,
        description=(
            "Number of candidate files searched. Populated for zero-match searches "
            "that use include, so an overly narrow glob is visible."
        ),
    )

    @field_validator("matches", mode="before")
    @classmethod
    def _coerce_legacy_matches(cls, value: object) -> object:
        """Coerce legacy string matches into GrepMatch objects.

        Older persisted observations stored ``matches`` as a list of file-path
        strings (before per-line matches were introduced). Map each such string
        to a GrepMatch so historical conversations remain loadable.
        """
        if not isinstance(value, list):
            return value
        coerced: list[object] = []
        for item in value:
            if isinstance(item, str):
                coerced.append({"file_path": item, "line_number": 0, "line": ""})
            else:
                coerced.append(item)
        return coerced


TOOL_DESCRIPTION = """Fast content search tool.
* Searches file contents using regular expressions
* Supports full regex syntax (eg. "log.*Error", "function\\s+\\w+", etc.)
* `path` accepts a directory or an exact file. Pyromind documentation uses both
  `.md` and `.mdx`; use `include="*.mdx"` for Studio/basic/sdk pages, or omit
  `include` when uncertain.
* Filter files by pattern with the include parameter (eg. "*.mdx", "*.js")
* Returns each matching line with its file path and 1-based line number.
* Only the first 100 matches are returned. Narrow your search with a stricter regex pattern or the include/path parameters if you need more results.
* A no-match result means only that this query found no lines; it does not prove
  the topic is absent.
* Use this tool when you need to find where specific patterns occur in files.
"""  # noqa


class GrepTool(ToolDefinition[GrepAction, GrepObservation]):
    """A ToolDefinition subclass that automatically initializes a GrepExecutor."""

    def declared_resources(self, action: Action) -> DeclaredResources:
        """Declare resource usage for parallel execution.

        All grep backends are stateless and safe to run lock-free in parallel:
        ripgrep and system grep spawn independent subprocesses, and the Python
        fallback only performs local file reads.
        """
        if not isinstance(action, GrepAction):
            raise TypeError(f"Expected GrepAction, got {type(action).__name__}")
        return DeclaredResources(keys=(), declared=True)

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        read_only_roots: list[str] | None = None,
    ) -> Sequence["GrepTool"]:
        """Initialize GrepTool with a GrepExecutor.

        Args:
            conv_state: Conversation state to get working directory from.
                         If provided, working_dir will be taken from
                         conv_state.workspace
        """
        # Import here to avoid circular imports
        from openhands.tools.grep.impl import GrepExecutor

        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        # Initialize the executor
        configured_roots = read_only_roots
        if configured_roots is None:
            configured_roots = [str(root) for root in configured_public_read_roots()]
        executor = GrepExecutor(
            working_dir=working_dir, read_only_roots=configured_roots
        )

        # Add working directory information to the tool description
        enhanced_description = (
            f"{TOOL_DESCRIPTION}\n\n"
            f"Your current working directory is: {working_dir}\n"
            f"When searching for content, searches are performed in this directory."
        )

        # Initialize the parent ToolDefinition with the executor
        return [
            cls(
                description=enhanced_description,
                action_type=GrepAction,
                observation_type=GrepObservation,
                annotations=ToolAnnotations(
                    title="grep",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# Automatically register the tool when this module is imported
register_tool(GrepTool.name, GrepTool)
