"""将平台状态映射后分发给当前 Product Runtime。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal


WorkflowStatus = Literal[
    "Succeeded", "Pending", "Running", "Failed", "Error", "Terminated"
]
CallbackOutcome = Literal[
    "delivered_async",
    "unknown_task",
    "unknown_conversation",
    "ignored_non_terminal",
    "duplicate_terminal",
]


_PLATFORM_STATUS_MAP: dict[str, WorkflowStatus] = {
    "succeeded": "Succeeded",
    "success": "Succeeded",
    "pending": "Pending",
    "running": "Running",
    "failed": "Failed",
    "error": "Error",
    "terminated": "Terminated",
    "stopped": "Terminated",
}


@dataclass(frozen=True)
class WorkflowCallbackResult:
    """单次回调执行结果。"""

    outcome: CallbackOutcome
    task_id: str
    normalized_status: WorkflowStatus | None
    conversation_id: str | None


def normalize_platform_status(raw: str) -> WorkflowStatus:
    """将平台或 Kafka 的状态字符串映射为 WorkflowStatus 枚举值。"""
    normalized = _PLATFORM_STATUS_MAP.get(raw.strip().lower())
    if normalized is not None:
        return normalized
    for candidate in (
        "Succeeded",
        "Pending",
        "Running",
        "Failed",
        "Error",
        "Terminated",
    ):
        if raw.strip().lower() == candidate.lower():
            return candidate  # type: ignore[return-value]
    raise ValueError(f"Unsupported workflow status: {raw!r}")


_status_dispatcher: Callable[..., Awaitable[WorkflowCallbackResult]] | None = None


def set_workflow_status_dispatcher(
    dispatcher: Callable[..., Awaitable[WorkflowCallbackResult]] | None,
) -> None:
    global _status_dispatcher
    _status_dispatcher = dispatcher


async def dispatch_workflow_status(**kwargs) -> WorkflowCallbackResult:
    if _status_dispatcher is not None:
        return await _status_dispatcher(**kwargs)
    return WorkflowCallbackResult(
        outcome="unknown_conversation"
        if kwargs.get("conversation_id")
        else "unknown_task",
        task_id=kwargs["task_id"],
        normalized_status=normalize_platform_status(kwargs["status"]),
        conversation_id=kwargs.get("conversation_id"),
    )
