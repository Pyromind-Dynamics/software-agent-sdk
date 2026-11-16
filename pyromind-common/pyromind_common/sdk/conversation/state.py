from typing import Any, Protocol

from pydantic import Field

from pyromind_common.sdk.utils.models import OpenHandsModel


class ActiveLongTask(OpenHandsModel):
    """A long-running platform task submitted from a conversation.

    Tracked on ``ToolState.active_long_tasks`` so the presentation
    layer can keep the conversation in the running state while any task is
    in flight, and so interrupting the conversation can stop the tasks too.
    """

    task_id: str = Field(description="Platform task id.")
    kind: str = Field(
        description=("Task kind, e.g. 'data_preparation' or 'data_cleaning'.")
    )
    status: str = Field(
        default="Pending",
        description="Latest known platform status snapshot for display.",
    )


class ToolState(Protocol):
    persistence_dir: str | None
    workspace: Any
    agent_state: dict[str, Any]
    secret_registry: Any
    agent: Any
