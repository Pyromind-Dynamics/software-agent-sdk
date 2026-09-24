from pathlib import Path
from types import SimpleNamespace

from openhands.sdk.workspace.workspace import LocalWorkspace
from openhands.tools.utils.conversation_dirs import conversation_state_dir


def test_conversation_state_dir_prefers_host_conversation_root(tmp_path: Path) -> None:
    host_root = tmp_path / "conversations" / "conversation-1"
    conversation = SimpleNamespace(
        workspace_root=host_root,
        workspace=LocalWorkspace(working_dir="/target-workspace/conversation-1"),
    )

    assert (
        conversation_state_dir(conversation, "tasks")
        == tmp_path / "conversations" / "tasks"
    )


def test_conversation_state_dir_falls_back_to_local_workspace(tmp_path: Path) -> None:
    root = tmp_path / "conversations" / "conversation-1"
    root.mkdir(parents=True)
    conversation = SimpleNamespace(workspace=LocalWorkspace(working_dir=root))

    assert (
        conversation_state_dir(conversation, "tasks")
        == tmp_path / "conversations" / "tasks"
    )


def test_conversation_state_dir_returns_none_without_host_path() -> None:
    conversation = SimpleNamespace(workspace=SimpleNamespace(working_dir=None))

    assert conversation_state_dir(conversation, "tasks") is None
