"""Host-side locations of per-conversation control-plane state.

Tools that keep task records beside the conversation directory need a host
path, which the SDK resolves from ``workspace.working_dir``. A remote (sandbox)
workspace only has a container path, so the host conversation root carried on
the conversation object takes precedence there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def conversation_state_dir(conversation: Any, dirname: str) -> Path | None:
    """Host directory holding ``dirname`` state for ``conversation``.

    Returns None when the conversation exposes neither a host conversation root
    nor a workspace directory.
    """
    root = _conversation_root(conversation)
    if root is None:
        return None
    return root.parent / dirname


def _conversation_root(conversation: Any) -> Path | None:
    host_root = getattr(conversation, "workspace_root", None)
    if isinstance(host_root, (str, Path)):
        return Path(host_root).resolve()
    workspace = getattr(conversation, "workspace", None)
    working_dir = getattr(workspace, "working_dir", None)
    if isinstance(working_dir, (str, Path)):
        return Path(working_dir).resolve()
    return None
