"""ApplyPatch ToolDefinition and executor integrating the cookbook implementation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.utils import default_path_access_policy
from openhands.tools.workflow.definition import (
    WORKFLOW_RELATIVE_PATH,
    mark_pyromind_workflow_dirty,
)

from .core import Commit, DiffError, process_patch


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class ApplyPatchAction(Action):
    """Tool action schema specifying the patch to apply.

    The patch must follow the exact text format described in the OpenAI
    Cookbook's GPT-5.1 prompting guide. The executor parses this patch and
    applies changes relative to the current workspace root.
    """

    patch: str = Field(
        description=(
            "The full patch text, starting with '*** Begin Patch' and ending "
            "with '*** End Patch'. Pass it as a plain string (newlines escaped "
            "as \\n inside the JSON argument); do not wrap it in a code fence."
        ),
    )


class ApplyPatchObservation(Observation):
    """Result of applying a patch.

    - message: human-readable summary of the changes or error
    - fuzz: number of lines of fuzz used when applying hunks (0 means exact)
    - commit: structured summary of the applied operations
    """

    message: str = ""
    fuzz: int = 0
    commit: Commit | None = None


class ApplyPatchExecutor(ToolExecutor[ApplyPatchAction, ApplyPatchObservation]):
    """Executor that applies unified text patches within the workspace.

    Uses the pure functions in core.py for parsing and applying patches. All
    filesystem access is constrained to the agent's workspace_root.
    """

    def __init__(self, workspace_root: str):
        """Initialize executor with a workspace root.

        Args:
            workspace_root: Base directory relative to which all patch paths are
                resolved. Absolute or path-escaping references are rejected.
        """
        self.workspace_root = Path(workspace_root).resolve()
        self.path_policy = default_path_access_policy(self.workspace_root)

    def _resolve_path(self, p: str) -> Path:
        """Resolve a file path into the workspace, disallowing escapes."""
        pth = (
            (self.workspace_root / p).resolve()
            if not p.startswith("/")
            else Path(p).resolve()
        )
        if not pth.is_relative_to(self.workspace_root):
            raise DiffError("Absolute or escaping paths are not allowed")
        return pth

    def __call__(
        self,
        action: ApplyPatchAction,
        conversation=None,
    ) -> ApplyPatchObservation:
        """Execute the patch application and return an observation."""

        def open_file(path: str) -> str:
            fp = self._resolve_path(path)
            self.path_policy.require(fp, "read")
            with open(fp, encoding="utf-8") as f:
                return f.read()

        def write_file(path: str, content: str) -> None:
            fp = self._resolve_path(path)
            self.path_policy.require(fp, "write")
            fp.parent.mkdir(parents=True, exist_ok=True)
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)

        def remove_file(path: str) -> None:
            fp = self._resolve_path(path)
            self.path_policy.require(fp, "write")
            fp.unlink(missing_ok=False)

        try:
            msg, fuzz, commit = process_patch(
                action.patch, open_file, write_file, remove_file
            )
            self._mark_workflow_dirty_if_changed(commit, conversation)
            # Include a human-readable summary in content so Responses API sees
            # a function_call_output payload paired with the function_call.
            obs = ApplyPatchObservation(
                message=msg,
                fuzz=fuzz,
                commit=commit,
            )
            if msg:
                # Use Observation.from_text to populate content field correctly
                obs = ApplyPatchObservation.from_text(
                    text=msg,
                    message=msg,
                    fuzz=fuzz,
                    commit=commit,
                    is_error=False,
                )
            return obs
        except DiffError as e:
            return ApplyPatchObservation.from_text(text=str(e), is_error=True)

    def _mark_workflow_dirty_if_changed(
        self,
        commit: Commit,
        conversation,
    ) -> None:
        workflow_path = (self.workspace_root / WORKFLOW_RELATIVE_PATH).resolve()
        for path in commit.changes:
            if self._resolve_path(path) == workflow_path:
                mark_pyromind_workflow_dirty(conversation)
                return


# Adapted from codex's apply_patch tool instructions
# (codex-rs/prompts/templates/apply_patch_tool_instructions.md), reworked for
# a JSON function tool that takes the patch text in a single `patch` argument.
_DESCRIPTION = """Use the `apply_patch` tool to create, delete, or edit files.

The `patch` argument is a stripped-down, file-oriented diff format:

*** Begin Patch
[ one or more file sections ]
*** End Patch

Each file section starts with exactly one of three headers:

*** Add File: <path> - create a new file. Every following line is a + line
(the initial contents).
*** Delete File: <path> - remove an existing file. Nothing follows.
*** Update File: <path> - patch an existing file in place. May be immediately
followed by *** Move to: <new path> to rename the file. Then one or more hunks,
each introduced by @@ (optionally followed by a class/function header). Within
a hunk each line starts with ' ' (context), '-' (remove), or '+' (add).

For Update File hunks:
- Show 3 lines of context immediately above and below each change. If a change
  is within 3 lines of a previous change, do NOT duplicate context lines
  between hunks.
- If 3 lines of context is insufficient to uniquely locate the snippet, use
  the @@ operator to name the enclosing class or function, e.g.
  `@@ class BaseClass` or `@@ def method():`. Multiple @@ statements may be
  stacked to narrow down further.

Full grammar:
Patch := Begin { FileOp } End
Begin := "*** Begin Patch" NEWLINE
End := "*** End Patch" NEWLINE
FileOp := AddFile | DeleteFile | UpdateFile
AddFile := "*** Add File: " path NEWLINE { "+" line NEWLINE }
DeleteFile := "*** Delete File: " path NEWLINE
UpdateFile := "*** Update File: " path NEWLINE [ MoveTo ] { Hunk }
MoveTo := "*** Move to: " newPath NEWLINE
Hunk := "@@" [ header ] NEWLINE { HunkLine } [ "*** End of File" NEWLINE ]
HunkLine := (" " | "-" | "+") text NEWLINE

Example combining several operations:

*** Begin Patch
*** Add File: hello.txt
+Hello world
*** Update File: src/app.py
*** Move to: src/main.py
@@ def greet():
-print("Hi")
+print("Hello, world!")
*** Delete File: obsolete.txt
*** End Patch

Remember:
- Every file section must have an Add/Delete/Update header.
- Prefix every new content line with `+`, even when creating a new file.
- The `+` prefix applies ONLY to file content lines. Never prefix `***` marker
  lines: the patch must end with the bare line `*** End Patch`, not `+*** End
  Patch`.
- File paths must be relative, NEVER ABSOLUTE."""


class ApplyPatchTool(ToolDefinition[ApplyPatchAction, ApplyPatchObservation]):
    """ToolDefinition for applying unified text patches.

    Creates an ApplyPatchExecutor bound to the current workspace and supplies
    the full patch-format instructions (adapted from codex) as the tool
    description, so models that don't natively know this format can still
    produce valid patches.
    """

    @classmethod
    def create(cls, conv_state: ConversationState) -> Sequence[ApplyPatchTool]:
        """Initialize the tool for the active conversation state."""
        executor = ApplyPatchExecutor(workspace_root=conv_state.workspace.working_dir)
        return [
            cls(
                description=_DESCRIPTION,
                action_type=ApplyPatchAction,
                observation_type=ApplyPatchObservation,
                annotations=ToolAnnotations(
                    title="apply_patch",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


register_tool(ApplyPatchTool.name, ApplyPatchTool)
