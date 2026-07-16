from pathlib import Path
from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor
from openhands.sdk.utils.path import is_host_absolute_path


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
from openhands.tools.file_editor.definition import (
    CommandLiteral,
    FileEditorAction,
    FileEditorObservation,
)
from openhands.tools.file_editor.editor import FileEditor
from openhands.tools.file_editor.exceptions import ToolError
from openhands.tools.utils import (
    configured_public_read_roots,
    default_path_access_policy,
    logical_public_read_path,
    resolve_public_read_alias,
)
from openhands.tools.workflow.definition import (
    WORKFLOW_RELATIVE_PATH,
    mark_pyromind_workflow_dirty,
)


# Module-global editor instance (lazily initialized in file_editor)
_GLOBAL_EDITOR: FileEditor | None = None


class FileEditorExecutor(ToolExecutor):
    """File editor executor with configurable file restrictions."""

    def __init__(
        self,
        workspace_root: str | None = None,
        allowed_edits_files: list[str] | None = None,
        read_only_roots: list[str] | None = None,
    ):
        self.workspace_root = (
            Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
        )
        self.editor: FileEditor = FileEditor(workspace_root=str(self.workspace_root))
        self.read_only_roots = configured_public_read_roots(read_only_roots)
        self.path_policy = default_path_access_policy(
            self.workspace_root, self.read_only_roots
        )
        self.read_only_editors = {
            root: FileEditor(workspace_root=str(root)) for root in self.read_only_roots
        }
        self.allowed_edits_files: set[Path] | None = (
            {Path(self.normalize_path(f)).resolve() for f in allowed_edits_files}
            if allowed_edits_files
            else None
        )

    def normalize_path(self, path: str) -> str:
        """Return a host-absolute path for editor operations.

        The underlying FileEditor validates host-absolute paths only. The model
        should be able to pass short workspace-relative paths such as
        ``workflow.py``; this adapter resolves those paths in code before the
        validation layer runs.
        """
        action_path = Path(path)

        aliased = resolve_public_read_alias(path, self.read_only_roots)
        if aliased is not None:
            return str(aliased)

        if is_host_absolute_path(action_path):
            if str(action_path).startswith("/workspace/"):
                cwd_candidate = (Path.cwd() / str(action_path).lstrip("/")).resolve()
                if cwd_candidate.exists() or cwd_candidate.parent.exists():
                    return str(cwd_candidate)
            return str(action_path.resolve())

        workspace_candidate = (self.workspace_root / action_path).resolve()
        if workspace_candidate.exists():
            return str(workspace_candidate)

        if str(action_path).startswith("workspace/"):
            cwd_candidate = (Path.cwd() / action_path).resolve()
            if cwd_candidate.exists() or cwd_candidate.parent.exists():
                return str(cwd_candidate)

        return str(workspace_candidate)

    def __call__(
        self,
        action: FileEditorAction,
        conversation: "LocalConversation | None" = None,
    ) -> FileEditorObservation:
        normalized_path = self.normalize_path(action.path)

        operation = "read" if action.command == "view" else "write"
        try:
            self.path_policy.require(normalized_path, operation)
        except PermissionError as error:
            return FileEditorObservation.from_text(
                text=str(error), command=action.command, is_error=True
            )

        # Enforce allowed_edits_files restrictions
        if self.allowed_edits_files is not None and action.command != "view":
            action_path = Path(normalized_path).resolve()
            if action_path not in self.allowed_edits_files:
                return FileEditorObservation.from_text(
                    text=(
                        f"Operation '{action.command}' is not allowed "
                        f"on file '{action_path}'. "
                        f"Only the following files can be edited: "
                        f"{sorted(str(p) for p in self.allowed_edits_files)}"
                    ),
                    command=action.command,
                    is_error=True,
                )

        result: FileEditorObservation | None = None
        try:
            editor = self._editor_for_view(normalized_path, action.command)
            result = editor(
                command=action.command,
                path=normalized_path,
                file_text=action.file_text,
                view_range=action.view_range,
                old_str=action.old_str,
                new_str=action.new_str,
                insert_line=action.insert_line,
            )
        except ToolError as e:
            result = FileEditorObservation.from_text(
                text=e.message, command=action.command, is_error=True
            )
        assert result is not None, "file_editor should always return a result"
        if not result.is_error:
            public_path = logical_public_read_path(
                Path(normalized_path), self.read_only_roots
            )
            if public_path != normalized_path:
                result = result.model_copy(update={"path": public_path})
        if not result.is_error and action.command != "view":
            self._mark_workflow_dirty_if_target(normalized_path, conversation)
        return result

    def _editor_for_view(self, path: str, command: CommandLiteral) -> FileEditor:
        if command == "view":
            resolved = Path(path).resolve()
            for root, editor in self.read_only_editors.items():
                if resolved.is_relative_to(root):
                    return editor
        return self.editor

    def _mark_workflow_dirty_if_target(
        self,
        path: str,
        conversation: "LocalConversation | None",
    ) -> None:
        target_path = Path(path).resolve()
        workflow_path = (self.workspace_root / WORKFLOW_RELATIVE_PATH).resolve()
        if target_path != workflow_path:
            return
        mark_pyromind_workflow_dirty(conversation)


def file_editor(
    command: CommandLiteral,
    path: str,
    file_text: str | None = None,
    view_range: list[int] | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
    insert_line: int | None = None,
) -> FileEditorObservation:
    """A global FileEditor instance to be used by the tool."""

    global _GLOBAL_EDITOR
    if _GLOBAL_EDITOR is None:
        _GLOBAL_EDITOR = FileEditor()

    result: FileEditorObservation | None = None
    try:
        result = _GLOBAL_EDITOR(
            command=command,
            path=path,
            file_text=file_text,
            view_range=view_range,
            old_str=old_str,
            new_str=new_str,
            insert_line=insert_line,
        )
    except ToolError as e:
        result = FileEditorObservation.from_text(
            text=e.message, command=command, is_error=True
        )
    assert result is not None, "file_editor should always return a result"
    return result
