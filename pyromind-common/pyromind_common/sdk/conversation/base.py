"""The tool host contract, independent of agent execution or checkpoints."""

from typing import Any, Protocol

from pyromind_common.sdk.workspace.base import BaseWorkspace


class ToolConversation(Protocol):
    id: str
    workspace: BaseWorkspace
    state: Any
    persistence_dir: str | None

    def register_active_long_task(self, task: Any) -> None: ...
    def send_agent_message(self, message: Any) -> None: ...
