import os
from pathlib import Path

import pytest

from openhands.tools.apply_patch.definition import ApplyPatchAction, ApplyPatchExecutor
from openhands.tools.workflow.definition import PYROMIND_WORKFLOW_DIRTY_KEY


class _DirtyState:
    def __init__(self) -> None:
        self.agent_state: dict[str, object] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def owned(self) -> bool:
        return True


class _DirtyConversation:
    def __init__(self) -> None:
        self._state = _DirtyState()
        self._step_holds_state_lock = False


@pytest.fixture()
def tmp_ws(tmp_path: Path) -> Path:
    # match other tool tests: use pytest tmp_path as a workspace root
    return tmp_path


def run_exec(ws: Path, patch: str, conversation=None):
    ex = ApplyPatchExecutor(workspace_root=str(ws))
    return ex(ApplyPatchAction(patch=patch), conversation=conversation)


def test_create_modify_delete(tmp_ws: Path):
    # 1) create FACTS.txt
    patch1 = (
        "*** Begin Patch\n"
        "*** Add File: FACTS.txt\n"
        "+OpenHands SDK integrates tools.\n"
        "*** End Patch"
    )
    obs1 = run_exec(tmp_ws, patch1)
    assert not obs1.is_error
    fp = tmp_ws / "FACTS.txt"
    assert fp.exists()
    assert fp.read_text().rstrip("\n") == "OpenHands SDK integrates tools."

    # 2) append a second line
    patch2 = (
        "*** Begin Patch\n"
        "*** Update File: FACTS.txt\n"
        "@@\n"
        " OpenHands SDK integrates tools.\n"
        "+ApplyPatch works.\n"
        "*** End Patch"
    )
    obs2 = run_exec(tmp_ws, patch2)
    assert not obs2.is_error
    assert fp.read_text() == ("OpenHands SDK integrates tools.\nApplyPatch works.")

    # 3) delete
    patch3 = "*** Begin Patch\n*** Delete File: FACTS.txt\n*** End Patch"
    obs3 = run_exec(tmp_ws, patch3)
    assert not obs3.is_error
    assert not fp.exists()


def test_apply_patch_marks_workflow_dirty(tmp_ws: Path):
    patch = (
        "*** Begin Patch\n"
        "*** Add File: public_data/workflow_canvas/workflow.py\n"
        "+# workflow: Patch Demo\n"
        "+limit = 20\n"
        "*** End Patch"
    )
    conversation = _DirtyConversation()

    obs = run_exec(tmp_ws, patch, conversation=conversation)

    assert not obs.is_error
    assert conversation._state.agent_state[PYROMIND_WORKFLOW_DIRTY_KEY] is True


def test_apply_patch_non_workflow_file_does_not_mark_dirty(tmp_ws: Path):
    patch = (
        "*** Begin Patch\n*** Add File: notes.py\n+print('not workflow')\n*** End Patch"
    )
    conversation = _DirtyConversation()

    obs = run_exec(tmp_ws, patch, conversation=conversation)

    assert not obs.is_error
    assert PYROMIND_WORKFLOW_DIRTY_KEY not in conversation._state.agent_state


def test_reject_absolute_path(tmp_ws: Path):
    # refuse escape/absolute paths
    patch = (
        "*** Begin Patch\n"
        f"*** Add File: {os.path.abspath('/etc/passwd')}\n"
        "+x\n"
        "*** End Patch"
    )
    obs = run_exec(tmp_ws, patch)
    assert obs.is_error
    assert "Absolute or escaping paths" in obs.text


@pytest.mark.parametrize(
    ("patch", "expected_error"),
    [
        (
            "*** Add File: FACTS.txt\n+x\n*** End Patch",
            "PATCH_BEGIN_MISSING",
        ),
        (
            "*** Begin Patch\n*** Add File: FACTS.txt\n+x",
            "PATCH_END_MISSING",
        ),
        (
            "*** Begin Patch\n*** Bogus File: FACTS.txt\n*** End Patch",
            "PATCH_SECTION_INVALID at line 2",
        ),
        (
            "*** Begin Patch\n*** Add File: FACTS.txt\nx\n*** End Patch",
            "PATCH_ADD_LINE_INVALID at line 3",
        ),
    ],
)
def test_patch_parse_errors_are_actionable(
    tmp_ws: Path,
    patch: str,
    expected_error: str,
):
    observation = run_exec(tmp_ws, patch)

    assert observation.is_error
    assert expected_error in observation.text


@pytest.mark.parametrize(
    "patch",
    [
        # Whitespace around the envelope markers (codex trims boundary lines).
        ("*** Begin Patch \n*** Add File: lenient.txt\n+content\n  *** End Patch  "),
        # Whole patch wrapped in a markdown code fence.
        (
            "```\n"
            "*** Begin Patch\n"
            "*** Add File: lenient.txt\n"
            "+content\n"
            "*** End Patch\n"
            "```"
        ),
        # Shell heredoc wrapper (codex lenient mode).
        (
            "<<'EOF'\n"
            "*** Begin Patch\n"
            "*** Add File: lenient.txt\n"
            "+content\n"
            "*** End Patch\n"
            "EOF"
        ),
        # Trailing prose after the closing marker.
        (
            "*** Begin Patch\n"
            "*** Add File: lenient.txt\n"
            "+content\n"
            "*** End Patch\n"
            "Done! The file has been created."
        ),
        # Spurious '+' prefix on the closing marker (over-applied content rule).
        ("*** Begin Patch\n*** Add File: lenient.txt\n+content\n+*** End Patch"),
    ],
)
def test_lenient_envelope_parsing(tmp_ws: Path, patch: str):
    obs = run_exec(tmp_ws, patch)
    assert not obs.is_error
    assert (tmp_ws / "lenient.txt").read_text() == "content"


def test_missing_end_marker_still_rejected(tmp_ws: Path):
    # A patch without any closing marker is likely truncated output and must
    # not be silently applied.
    patch = "*** Begin Patch\n*** Add File: trunc.txt\n+partial content"
    obs = run_exec(tmp_ws, patch)
    assert obs.is_error
    assert "PATCH_END_MISSING" in obs.text
    assert not (tmp_ws / "trunc.txt").exists()


def test_multi_hunk_success_single_file(tmp_ws: Path):
    fp = tmp_ws / "multi_success.txt"
    fp.write_text("a1\na2\na3\na4\na5\n")

    patch = (
        "*** Begin Patch\n"
        "*** Update File: multi_success.txt\n"
        "@@\n"
        " a1\n"
        "-a2\n"
        "+A2\n"
        " a3\n"
        " a4\n"
        "-a5\n"
        "+A5\n"
        "*** End Patch"
    )

    obs = run_exec(tmp_ws, patch)
    assert not obs.is_error
    assert fp.read_text() == "a1\nA2\na3\na4\nA5\n"


def test_multi_file_update_single_patch(tmp_ws: Path):
    fp1 = tmp_ws / "file1.txt"
    fp2 = tmp_ws / "file2.txt"
    fp1.write_text("x1\nx2\n")
    fp2.write_text("y1\ny2\n")

    patch = (
        "*** Begin Patch\n"
        "*** Update File: file1.txt\n"
        "@@\n"
        " x1\n"
        "-x2\n"
        "+X2\n"
        "*** Update File: file2.txt\n"
        "@@\n"
        " y1\n"
        "-y2\n"
        "+Y2\n"
        "*** End Patch"
    )

    obs = run_exec(tmp_ws, patch)
    assert not obs.is_error
    assert fp1.read_text() == "x1\nX2\n"
    assert fp2.read_text() == "y1\nY2\n"


def test_multi_file_add_update_delete_single_patch(tmp_ws: Path):
    existing = tmp_ws / "existing.txt"
    to_delete = tmp_ws / "delete_me.txt"
    existing.write_text("base\n")
    to_delete.write_text("gone soon\n")

    patch = (
        "*** Begin Patch\n"
        "*** Add File: added.txt\n"
        "+new content\n"
        "*** Update File: existing.txt\n"
        "@@\n"
        " base\n"
        "+more\n"
        "*** Delete File: delete_me.txt\n"
        "*** End Patch"
    )

    obs = run_exec(tmp_ws, patch)
    assert not obs.is_error

    added = tmp_ws / "added.txt"
    assert added.exists()
    assert added.read_text() == "new content"

    assert existing.read_text() == "base\nmore\n"
    assert not to_delete.exists()


def test_multi_hunk_invalid_context_error(tmp_ws: Path):
    fp = tmp_ws / "multi.txt"
    fp.write_text("line1\nline2\nline3\nline4\n")

    patch = (
        "*** Begin Patch\n"
        "*** Update File: multi.txt\n"
        "@@\n"
        " line1\n"
        "-line2\n"
        "+line2a\n"
        " line3\n"
        "@@\n"
        " line3\n"
        "+line3a\n"
        " line4\n"
        "*** End Patch"
    )

    obs = run_exec(tmp_ws, patch)
    assert obs.is_error
    assert "Invalid Context" in obs.text


def test_fuzz_matching_trailing_spaces(tmp_ws: Path):
    fp = tmp_ws / "fuzz.txt"
    fp.write_text("a\ncontext line   \nend\n")

    patch = (
        "*** Begin Patch\n"
        "*** Update File: fuzz.txt\n"
        "@@\n"
        " context line\n"
        "-end\n"
        "+END\n"
        "*** End Patch"
    )

    obs = run_exec(tmp_ws, patch)
    assert not obs.is_error
    # fuzz should be > 0 because whitespace-stripped context is used
    assert obs.fuzz > 0
    assert fp.read_text() == "a\ncontext line   \nEND\n"


def test_delete_missing_file_expected_differror(tmp_ws: Path):
    """Delete of a missing file should surface as a structured DiffError.

    The reference implementation would bubble a FileNotFoundError from
    load_files/open_fn; our SDK adapts this by converting it into a
    "Delete File Error: Missing File" DiffError so the tool can return a
    clean error observation instead of crashing.
    """
    patch = "*** Begin Patch\n*** Delete File: missing.txt\n*** End Patch"
    obs = run_exec(tmp_ws, patch)
    # Intentionally assert the idealized behavior we *would* like to see.
    assert obs.is_error
    assert "Missing File" in obs.text


def test_duplicate_add_file_error(tmp_ws: Path):
    patch = (
        "*** Begin Patch\n"
        "*** Add File: dup.txt\n"
        "+one\n"
        "*** Add File: dup.txt\n"
        "+two\n"
        "*** End Patch"
    )
    obs = run_exec(tmp_ws, patch)
    assert obs.is_error
    assert "Add File Error: Duplicate Path" in obs.text


def test_path_escape_with_parent_directory(tmp_ws: Path):
    patch = "*** Begin Patch\n*** Add File: ../escape.txt\n+x\n*** End Patch"
    obs = run_exec(tmp_ws, patch)
    assert obs.is_error
    assert "Absolute or escaping paths" in obs.text


def test_reject_sibling_prefix_path_escape(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "workspace-escape" / "owned.txt"

    patch = (
        "*** Begin Patch\n"
        "*** Add File: ../workspace-escape/owned.txt\n"
        "+x\n"
        "*** End Patch"
    )
    obs = run_exec(workspace, patch)

    assert obs.is_error
    assert "Absolute or escaping paths" in obs.text
    assert not outside.exists()


def test_malformed_patch_header_returns_differror(tmp_ws: Path):
    """A patch missing '*** Begin Patch' must return a structured error, not crash.

    Before the assert→DiffError fix, process_patch() used assert to validate
    the header. Python disables assert with -O (optimized mode, used in Docker
    production images), so a bad header would silently pass and corrupt state.
    Now it raises DiffError which is caught and returned as is_error=True.
    """
    obs = run_exec(tmp_ws, "INVALID HEADER\n*** End Patch")
    assert obs.is_error
    assert "Begin Patch" in obs.text


def test_invalid_move_to_path_returns_differror(tmp_ws: Path):
    """A '*** Move to:' with '..' components must be rejected as a structured error.

    The original Parser.parse() had a TODO acknowledging this check was missing.
    An unvalidated move_to path could allow the agent to move files outside the
    workspace.
    """
    fp = tmp_ws / "source.txt"
    fp.write_text("hello\n")
    patch = (
        "*** Begin Patch\n"
        "*** Update File: source.txt\n"
        "*** Move to: ../outside.txt\n"
        "@@\n"
        " hello\n"
        "*** End Patch"
    )
    obs = run_exec(tmp_ws, patch)
    assert obs.is_error
    assert "Invalid move path" in obs.text
