"""Deliver Pyromind workflow terminal status updates back to conversations.

Agent-server callback layer for asynchronous Studio workflows. An external
Kafka consumer (or HTTP webhook) calls :func:`deliver_run_workflow_status`
when the platform reports a workflow status change. The callback uses the
``conversation_id`` written to the task at submission time
(``TrainingTaskCreateRequest.out_id``) to locate the original conversation,
injects a ``<system_reminder>`` for the LLM, and optionally triggers
``auto_run`` so the agent continues without a new user message.

Pyromind run_workflow 异步终态回调（agent-server 层）。

外部 Kafka consumer（或 HTTP webhook）在工作流状态变更时调用
:func:`deliver_run_workflow_status`。回调通过提交 task 时写入的
``conversation_id``（``TrainingTaskCreateRequest.out_id``）定位原会话，
向 LLM 注入 ``<system_reminder>``，并可选 ``auto_run`` 自动继续 Agent。

Module layout / 模块结构
------------------------
1. **Types & constants** — status enums, terminal set, platform status map
2. **Text builders** — user-facing submission text and LLM system reminders
3. **ID helpers** — parse ``conversation_id`` from Kafka / task metadata
4. **Broker bridge (optional)** — lazy import of ``run_workflow_broker`` for
   Debug ``wait_mode=block`` and in-process registry fallback
5. **Idempotency** — process-wide dedup of terminal deliveries per ``task_id``
6. **Conversation delivery** — inject hidden environment context
7. **Main entry** — :func:`deliver_run_workflow_status` orchestrates the above
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from openhands.agent_server.conversation_service import (
    ConversationService,
    get_default_conversation_service,
)
from openhands.agent_server.event_service import EventService
from openhands.sdk.llm import TextContent
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. Types & constants / 类型与常量
# ---------------------------------------------------------------------------

RunWorkflowStatus = Literal[
    "Succeeded", "Pending", "Running", "Failed", "Error", "Terminated"
]
"""Normalized workflow status aligned with RunWorkflowObservation."""

CallbackOutcome = Literal[
    "resolved_blocked",  # Debug path: broker.wait was woken
    "delivered_async",  # Async path: conversation updated + auto_run
    "unknown_task",  # No conversation_id and no broker registration
    "unknown_conversation",  # Invalid id or conversation not on this server
    "duplicate_terminal",  # Terminal status already delivered for task_id
    "ignored_non_terminal",  # Pending/Running — no agent restart
]

# Terminal statuses trigger conversation delivery; non-terminal are ignored.
# 终态才会投递并重启 Agent；非终态（Pending/Running）直接忽略。
TERMINAL_STATUSES: frozenset[RunWorkflowStatus] = frozenset(
    {"Succeeded", "Failed", "Error", "Terminated"}
)

# Lowercase platform / Kafka strings → canonical RunWorkflowStatus.
# 平台 / Kafka 原始字符串 → 规范枚举值。
_PLATFORM_STATUS_MAP: dict[str, RunWorkflowStatus] = {
    "succeeded": "Succeeded",
    "success": "Succeeded",
    "pending": "Pending",
    "running": "Running",
    "failed": "Failed",
    "error": "Error",
    "terminated": "Terminated",
    "stopped": "Terminated",
}

# In-process dedup for async terminal delivery (single agent-server instance).
# 进程内终态去重，避免 Kafka 重复消息多次 auto_run。
_delivered_terminal_lock = threading.Lock()
_delivered_terminal_task_ids: set[str] = set()


@dataclass(frozen=True)
class RunWorkflowCallbackResult:
    """Result of one callback invocation / 单次回调执行结果。"""

    outcome: CallbackOutcome
    task_id: str
    normalized_status: RunWorkflowStatus | None
    conversation_id: str | None


# ---------------------------------------------------------------------------
# 2. Text builders / 文案构造
# ---------------------------------------------------------------------------


def normalize_platform_status(raw: str) -> RunWorkflowStatus:
    """Map platform / Kafka status to RunWorkflowStatus.

    将平台 / Kafka 状态字符串映射为规范枚举值。
    """
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


def build_run_workflow_terminal_reminder(
    *,
    task_id: str,
    status: RunWorkflowStatus,
    error_log: str | None = None,
) -> str:
    """Build ``<system_reminder>`` for LLM context (via extended_content).

    构造注入 LLM 上下文的 ``<system_reminder>``（通过 extended_content）。
    """
    lines = [
        "<system_reminder>",
        f"Pyromind workflow task {task_id} completed with status {status}.",
        (
            "Resume the tool invocation associated with this task and follow "
            "that tool's result contract when inspecting outputs or errors."
        ),
        (
            "Respond in the language of the user's most recent non-empty "
            "visible message."
        ),
    ]
    if error_log:
        lines.append("Runtime error log:")
        lines.append(error_log)
    lines.append("</system_reminder>")
    return "\n".join(lines)


def build_run_workflow_submission_user_text(
    *,
    task_id: str,
    conversation_id: str,
    status: RunWorkflowStatus,
) -> str:
    """User-visible text after async submission (for RunWorkflowObservation).

    异步提交成功后给用户看的 observation 文案（供 run_workflow 工具复用）。
    """
    return (
        "工作流已提交到 Pyromind 平台，正在运行中。\n\n"
        f"- task_id: {task_id}\n"
        f"- conversation_id: {conversation_id}\n"
        f"- status: {status}\n"
        "- 当前对话本轮已结束；工作流完成后 Agent 将自动在本会话继续。\n\n"
        "请勿关闭页面，界面将保持锁定直至运行结束。"
    )


# ---------------------------------------------------------------------------
# 3. ID helpers / 会话 ID 解析
# ---------------------------------------------------------------------------


def parse_conversation_id(raw: str) -> UUID:
    """Parse conversation id from Kafka payload or task out_id.

    Accepts standard UUID strings and 32-char hex (no hyphens).

    解析 Kafka 或 task out_id 中的 conversation_id；支持标准 UUID 与 32 位 hex。
    """
    cleaned = raw.strip()
    try:
        return UUID(cleaned)
    except ValueError:
        return UUID(hex=cleaned)


# ---------------------------------------------------------------------------
# 4. Broker bridge (optional) / Broker 可选桥接
#
# run_workflow_broker lives in openhands-tools and may not be installed yet.
# Lazy import keeps this module usable before the broker is implemented.
# Debug wait_mode=block uses broker.resolve(); async path uses broker.lookup()
# only when conversation_id is missing from the Kafka message.
#
# run_workflow_broker 位于 openhands-tools，可能尚未实现；延迟导入保证本模块
# 可独立使用。Debug 阻塞模式走 broker.resolve()；async 模式仅在 Kafka 未带
# conversation_id 时用 broker.lookup() 作 fallback。
# ---------------------------------------------------------------------------


def _get_run_workflow_broker_module():
    import importlib

    return importlib.import_module("openhands.tools.workflow.run_workflow_broker")


def _lookup_conversation_id_from_broker(task_id: str) -> str | None:
    """Fallback: resolve conversation_id from in-process registry by task_id."""
    try:
        broker_module = _get_run_workflow_broker_module()
    except ImportError:
        return None

    registration = broker_module.get_run_workflow_result_broker().lookup(task_id)
    if registration is None:
        return None
    return registration.conversation_id


def _try_resolve_blocked_waiter(
    *,
    task_id: str,
    status: RunWorkflowStatus,
    error_log: str | None,
) -> bool:
    """Wake a Debug-path broker.wait() thread if one is registered."""
    try:
        broker_module = _get_run_workflow_broker_module()
    except ImportError:
        return False

    return broker_module.get_run_workflow_result_broker().resolve(
        task_id,
        status=status,
        error_log=error_log,
    )


# ---------------------------------------------------------------------------
# 5. Idempotency / 终态去重
# ---------------------------------------------------------------------------


def _mark_terminal_delivered(task_id: str) -> bool:
    """Record terminal delivery; return False if this task_id was already handled."""
    with _delivered_terminal_lock:
        if task_id in _delivered_terminal_task_ids:
            return False
        _delivered_terminal_task_ids.add(task_id)
        return True


def _release_terminal_delivery(task_id: str) -> None:
    """Allow a failed delivery attempt to be retried."""
    with _delivered_terminal_lock:
        _delivered_terminal_task_ids.discard(task_id)


# ---------------------------------------------------------------------------
# 6. Conversation delivery / 会话投递
# ---------------------------------------------------------------------------


async def resume_conversation_after_workflow(
    *,
    event_service: EventService,
    task_id: str,
    status: RunWorkflowStatus,
    error_log: str | None = None,
    auto_run: bool = True,
) -> None:
    """Inject terminal status into a conversation and optionally start the agent.

    Uses hidden environment context so the LLM receives the workflow status
    without creating a user-authored chat event.

    通过隐藏的 environment 上下文注入终态并可选启动 Agent，不会伪造用户消息。
    """
    reminder = build_run_workflow_terminal_reminder(
        task_id=task_id,
        status=status,
        error_log=error_log,
    )
    await event_service.send_internal_context(
        [TextContent(text=reminder)],
        run=auto_run,
    )


# ---------------------------------------------------------------------------
# 7. Main entry / 主入口
# ---------------------------------------------------------------------------


async def deliver_run_workflow_status(
    *,
    task_id: str,
    status: str,
    error_log: str | None = None,
    conversation_id: str | None = None,
    updated_at: datetime | None = None,
    auto_run: bool = True,
    conversation_service: ConversationService | None = None,
) -> RunWorkflowCallbackResult:
    """Handle one workflow status update from Kafka or HTTP webhook.

    Processing pipeline / 处理流水线:
    1. Normalize status
    2. Try Debug broker resolve → ``resolved_blocked``
    3. Skip non-terminal → ``ignored_non_terminal``
    4. Resolve conversation_id (Kafka > broker fallback)
    5. Locate EventService on this agent-server
    6. Dedup terminal → ``duplicate_terminal``
    7. Inject reminder + auto_run → ``delivered_async``

    Args:
        task_id: Platform task id from ``studio.create()``.
        status: Raw platform / Kafka status string.
        error_log: Runtime log when the workflow failed or errored.
        conversation_id: Usually from task ``out_id`` / Kafka message.
        updated_at: Reserved for future ordering / idempotency.
        auto_run: Restart agent after injecting terminal reminder (default True).
        conversation_service: Override for tests; else process singleton.
    """
    del updated_at  # reserved for future idempotency / ordering

    deliver_t0 = time.monotonic()
    normalized_status = normalize_platform_status(status)
    service = conversation_service or get_default_conversation_service()

    # Step 2: Debug block path — wake broker.wait(), skip async delivery.
    if _try_resolve_blocked_waiter(
        task_id=task_id,
        status=normalized_status,
        error_log=error_log,
    ):
        logger.info(
            "Resolved blocked run_workflow waiter for task_id=%s status=%s",
            task_id,
            normalized_status,
        )
        return RunWorkflowCallbackResult(
            outcome="resolved_blocked",
            task_id=task_id,
            normalized_status=normalized_status,
            conversation_id=conversation_id,
        )

    # Step 3: Pending/Running — no agent restart for async path.
    if normalized_status not in TERMINAL_STATUSES:
        logger.debug(
            "Ignoring non-terminal run_workflow status task_id=%s status=%s",
            task_id,
            normalized_status,
        )
        return RunWorkflowCallbackResult(
            outcome="ignored_non_terminal",
            task_id=task_id,
            normalized_status=normalized_status,
            conversation_id=conversation_id,
        )

    # Step 4: conversation_id from the callback or broker registry.
    resolved_conversation_id = conversation_id
    if resolved_conversation_id is None:
        resolved_conversation_id = _lookup_conversation_id_from_broker(task_id)
    if resolved_conversation_id is None:
        logger.warning(
            "No conversation_id for run_workflow callback task_id=%s", task_id
        )
        return RunWorkflowCallbackResult(
            outcome="unknown_task",
            task_id=task_id,
            normalized_status=normalized_status,
            conversation_id=None,
        )

    try:
        conversation_uuid = parse_conversation_id(resolved_conversation_id)
    except ValueError:
        logger.warning(
            "Invalid conversation_id=%r for run_workflow task_id=%s",
            resolved_conversation_id,
            task_id,
        )
        return RunWorkflowCallbackResult(
            outcome="unknown_conversation",
            task_id=task_id,
            normalized_status=normalized_status,
            conversation_id=resolved_conversation_id,
        )

    # Step 5: Conversation must be loaded on this agent-server instance.
    # Under Kafka broadcast (per-pod consumer group), other pods normally miss
    # the conversation — return unknown_conversation without raising so the
    # consumer skips (no retry/DLQ). Only the pod that holds the session delivers.
    event_service = await service.get_event_service(conversation_uuid)
    if event_service is None:
        logger.info(
            "Skip run_workflow callback: conversation %s not on this pod "
            "(expected under Kafka broadcast). task_id=%s",
            conversation_uuid,
            task_id,
        )
        return RunWorkflowCallbackResult(
            outcome="unknown_conversation",
            task_id=task_id,
            normalized_status=normalized_status,
            conversation_id=str(conversation_uuid),
        )

    # Step 6: Reserve this terminal delivery after all routing data is available.
    if not _mark_terminal_delivered(task_id):
        logger.info(
            "Ignoring duplicate terminal run_workflow status task_id=%s status=%s",
            task_id,
            normalized_status,
        )
        return RunWorkflowCallbackResult(
            outcome="duplicate_terminal",
            task_id=task_id,
            normalized_status=normalized_status,
            conversation_id=str(conversation_uuid),
        )

    # Step 7: Deliver to conversation and auto_run.
    try:
        resume_t0 = time.monotonic()
        await resume_conversation_after_workflow(
            event_service=event_service,
            task_id=task_id,
            status=normalized_status,
            error_log=error_log,
            auto_run=auto_run,
        )
        resume_ms = (time.monotonic() - resume_t0) * 1000
    except Exception:
        _release_terminal_delivery(task_id)
        raise
    logger.info(
        "Delivered run_workflow terminal status task_id=%s status=%s "
        "conversation_id=%s auto_run=%s",
        task_id,
        normalized_status,
        conversation_uuid,
        auto_run,
    )
    logger.info(
        "[perf] run_workflow_callback task_id=%s conversation_id=%s "
        "resume_ms=%.1f total_ms=%.1f auto_run=%s status=%s",
        task_id,
        conversation_uuid,
        resume_ms,
        (time.monotonic() - deliver_t0) * 1000,
        auto_run,
        normalized_status,
    )
    return RunWorkflowCallbackResult(
        outcome="delivered_async",
        task_id=task_id,
        normalized_status=normalized_status,
        conversation_id=str(conversation_uuid),
    )
