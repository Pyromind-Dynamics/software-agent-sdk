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

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID

from openhands.agent_server.conversation_service import (
    ConversationService,
    get_default_conversation_service,
)
from openhands.agent_server.event_service import EventService
from openhands.sdk.llm import LLM, Message, TextContent
from openhands.sdk.logger import get_logger


if TYPE_CHECKING:
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState
    from openhands.tools.node_signature.definition import NodeSignatureObservation


logger = get_logger(__name__)

# Matches `<var> = <NodeType>(...` lines of the generated workflow DSL; the
# line anchor keeps parameter values (which may contain ``=`` or ``(``) out.
_NODE_DEFINITION_PATTERN = re.compile(r"^(\w+)\s*=\s*([A-Za-z_]\w*)\s*\(", re.MULTILINE)

# Full DSL assignment lines (including the parameter list) so failed node ids
# can be matched against their ``id=`` keyword, pairing with
# ``_NODE_DEFINITION_PATTERN`` which only extracts type names.
_DSL_NODE_LINE_PATTERN = re.compile(
    r"^(\w+)\s*=\s*([A-Za-z_]\w*)\s*\(([^)]*)\)", re.MULTILINE
)

# `--- <node_code> ---` group headers in the decoded error log; the middleware
# groups per-node failure logs by node_code, which is the DSL node id.
_ERROR_LOG_NODE_HEADER_PATTERN = re.compile(r"^---\s*(\S+?)\s*---\s*$", re.MULTILINE)

# Instructs the LLM to condense raw node signatures into concise fix guidance.
_NODE_SIGNATURE_SUMMARY_PROMPT = """\
你是一个工作流调试助手。下面是一次失败的 Pyromind 工作流运行所使用的节点的原始
函数签名、docstring、参数规格与源码。请为每个节点产出一段简洁的说明，帮助 AI
智能体修复 workflow DSL：列出必填参数、关键可选参数及默认值，以及调用时必须遵守
的约束（入口、类型、跨节点连线）。不要复述源码，不要使用 markdown 标题或前言，
直接用中文输出精简内容。"""

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
    from_workflow_debug: bool = False,
    node_signature_guidance: str | None = None,
) -> str:
    """Build ``<system_reminder>`` for LLM context (via extended_content).

    构造注入 LLM 上下文的 ``<system_reminder>``（通过 extended_content）。

    When ``from_workflow_debug`` is True (Kafka out_id tagged
    ``agent1#debug#...``), success/failure guidance is specific to the
    debug/test loop. Production ``run_workflow`` callbacks keep the generic
    resume wording.

    ``node_signature_guidance`` is appended after the runtime error log: it is
    either an LLM-condensed summary of the workflow nodes' signatures or the
    raw signature text when summarization failed (see
    :func:`build_node_signature_guidance`).
    """
    lines = [
        "<system_reminder>",
        f"Pyromind workflow task {task_id} completed with status {status}.",
    ]
    if from_workflow_debug and status == "Succeeded":
        lines.append(
            "This terminal status is from a workflow_debug (test) run that "
            "passed. Briefly tell the user that this test workflow succeeded, "
            "then wait for their next message. Do not call workflow_debug "
            "again unless they ask."
        )
    elif from_workflow_debug and status in ("Failed", "Error"):
        lines.append(
            "This terminal status is from a workflow_debug (test) run that "
            "failed. The workflow DSL may be wrong. Read the runtime error "
            "below, regenerate or fix "
            "public_data/workflow_canvas/workflow.py accordingly (validate if "
            "needed), then call workflow_debug again to continue testing."
        )
    elif from_workflow_debug and status == "Terminated":
        lines.append(
            "This terminal status is from a workflow_debug (test) run that "
            "was terminated. Briefly explain that to the user. Use any error "
            "log below to decide whether to fix the DSL and call "
            "workflow_debug again, or wait for the user's next message."
        )
    elif status == "Terminated":
        lines.append(
            "This task was terminated by the platform (cancelled or aborted). "
            "Briefly explain this to the user and ask whether they want to "
            "resubmit. Do NOT automatically resubmit the task."
        )
    else:
        lines.append(
            "Resume the tool invocation associated with this task and follow "
            "that tool's result contract when inspecting outputs or errors."
        )
    lines.append(
        "Respond in the language of the user's most recent non-empty visible message."
    )

    if error_log:
        lines.append("")
        lines.append("Runtime error log:")
        lines.append(error_log)
    if node_signature_guidance:
        lines.append("")
        lines.append("Node signature guidance:")
        lines.append(node_signature_guidance)
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


def _reset_workflow_attempt_counter(event_service: EventService) -> None:
    """Reset the conversation's round submission counter after a successful debug round.

    A successful debug/test round ends here (Succeeded terminal status), so the
    next debug session starts with a fresh attempt budget. Kept in sync with
    ``openhands.tools.workflow.run_workflow.WORKFLOW_ATTEMPT_STATE_KEY``.
    """
    try:
        conversation = event_service.get_conversation()
    except ValueError:
        return
    import importlib

    run_workflow_module = importlib.import_module(
        "openhands.tools.workflow.run_workflow"
    )
    state = conversation.state
    state.agent_state = {
        **state.agent_state,
        run_workflow_module.WORKFLOW_ATTEMPT_STATE_KEY: 0,
    }


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


def _extract_node_names_from_dsl(dsl_text: str) -> list[str]:
    """Extract distinct node type names from generated workflow DSL text.

    DSL lines look like ``<var> = <NodeType>(id=..., ...)``; the line anchor
    excludes comment lines and parameter values. Order of first appearance is
    preserved and duplicates are dropped.
    """
    node_names: list[str] = []
    seen: set[str] = set()
    for _, node_name in _NODE_DEFINITION_PATTERN.findall(dsl_text):
        if node_name not in seen:
            seen.add(node_name)
            node_names.append(node_name)
    return node_names


def _extract_dsl_node_types(dsl_text: str) -> dict[str, str]:
    """Map DSL node ids to their type names in order of first appearance.

    ``<var> = <NodeType>(id=15, ...)`` → ``{"15": "NodeType"}``. Nodes without
    an explicit ``id=`` keyword fall back to their DSL line number, mirroring
    the middleware's ``node_code`` derivation. Duplicate ids keep the first
    type.
    """
    node_types: dict[str, str] = {}
    for match in _DSL_NODE_LINE_PATTERN.finditer(dsl_text):
        node_name, params = match.group(2), match.group(3)
        id_match = re.search(r"\bid\s*=\s*([^,\s)]+)", params)
        if id_match:
            node_id = id_match.group(1).strip("\"'")
        else:
            node_id = str(dsl_text[: match.start()].count("\n") + 1)
        node_types.setdefault(node_id, node_name)
    return node_types


def _failed_node_names_from_error_log(
    error_log: str | None,
    dsl_text: str,
) -> list[str] | None:
    """Type names of only the failed nodes, when the error log identifies them.

    The decoded error log groups per-node logs under ``--- <node_code> ---``
    headers whose values match DSL ``id=`` keywords. Returns the matched node
    types (DSL order, deduplicated). Returns None when the error log has no
    usable headers or none match, so the caller can fall back to all nodes.
    """
    if not error_log:
        return None
    failed_ids = [
        node_id.strip("\"'")
        for node_id in _ERROR_LOG_NODE_HEADER_PATTERN.findall(error_log)
    ]
    if not failed_ids:
        return None
    node_types = _extract_dsl_node_types(dsl_text)
    matched: list[str] = []
    seen: set[str] = set()
    for node_id in failed_ids:
        node_type = node_types.get(node_id)
        if node_type is None or node_type in seen:
            continue
        seen.add(node_type)
        matched.append(node_type)
    return matched or None


def _env_from_x_cluster(x_cluster: str | None) -> str | None:
    """Derive the platform env (pre/pre2/prod) from an ``x-cluster`` header.

    Mirrors ``dependencies.load_base_env``: ``us-west-1#pre`` → ``pre``. Returns
    None when the header is missing or has no env suffix, letting the signature
    executor fall back to its own ``APP_ENV`` default.
    """
    if not x_cluster or "#" not in x_cluster:
        return None
    return x_cluster.rsplit("#", 1)[-1].strip().lower() or None


def _fetch_node_signatures_text(
    conversation: LocalConversation,
    node_names: list[str],
) -> str | None:
    """Fetch raw signature text for all requested nodes via the middleware API.

    Returns None when nothing usable was fetched. Headers and auth token come
    from the conversation state, mirroring the workflow tools' wiring.
    """
    from openhands.tools.node_signature.definition import NodeSignatureAction
    from openhands.tools.node_signature.impl import NodeSignatureExecutor
    from openhands.tools.workflow.validate_workflow_dsl import (
        PYROMIND_VALIDATE_HEADERS_STATE_KEY,
    )

    state = cast("ConversationState", conversation.state)
    raw_headers = state.agent_state.get(PYROMIND_VALIDATE_HEADERS_STATE_KEY)
    headers = (
        {str(k): str(v) for k, v in raw_headers.items() if v is not None}
        if isinstance(raw_headers, dict)
        else {}
    )
    observation: NodeSignatureObservation = NodeSignatureExecutor(
        env=_env_from_x_cluster(headers.get("x-cluster")),
        headers=headers,
    )(
        NodeSignatureAction(node_names=node_names, include_source=True),
        conversation=conversation,
    )
    if observation.status == "error" or not observation.results:
        return None
    if not any(result.get("success") for result in observation.results):
        return None
    return "\n".join(
        content.text
        for content in observation.to_llm_content
        if isinstance(content, TextContent)
    )


async def build_node_signature_guidance(
    *,
    event_service: EventService,
    error_log: str | None = None,
    llm: LLM | None = None,
) -> str | None:
    """Summarize the failing run's node signatures for the debug callback.

    Reads ``public_data/workflow_canvas/workflow.py`` from the conversation
    workspace, resolves the failed nodes from the error log's
    ``--- <node_code> ---`` group headers, fetches only their signatures via
    the middleware API, and asks the conversation's LLM to condense them into
    concise fix guidance. When the error log does not identify any node, all
    nodes are used instead.

    Degradation policy: when LLM summarization fails, the raw signature text is
    returned instead; when no workflow file exists or signatures cannot be
    fetched, None is returned and the callback keeps today's error-log-only
    behavior. Never raises — failures must not break terminal delivery.
    """
    try:
        conversation = event_service.get_conversation()
    except ValueError:
        return None

    from openhands.tools.workflow.definition import WORKFLOW_RELATIVE_PATH

    try:
        working_dir = Path(conversation.workspace.working_dir)
        workflow_path = working_dir / WORKFLOW_RELATIVE_PATH
        if not workflow_path.is_file():
            return None
        dsl_text = workflow_path.read_text(encoding="utf-8")
        node_names = _failed_node_names_from_error_log(error_log, dsl_text)
        if node_names is None:
            node_names = _extract_node_names_from_dsl(dsl_text)
        if not node_names:
            return None
        logger.info(
            "Resolved nodes for failed workflow callback: %s",
            ", ".join(node_names),
        )
        raw_text = _fetch_node_signatures_text(conversation, node_names)
    except Exception:
        logger.warning(
            "Failed to fetch node signatures for debug callback; skipping "
            "node signature guidance",
            exc_info=True,
        )
        return None
    if raw_text is None:
        return None

    effective_llm = llm or getattr(
        getattr(conversation.state, "agent", None), "llm", None
    )
    if effective_llm is None:
        return raw_text
    try:
        response = await effective_llm.acompletion(
            messages=[
                Message(
                    role="system",
                    content=[TextContent(text=_NODE_SIGNATURE_SUMMARY_PROMPT)],
                ),
                Message(role="user", content=[TextContent(text=raw_text)]),
            ]
        )
        summary = "\n".join(
            content.text
            for content in response.message.content
            if isinstance(content, TextContent)
        ).strip()
        if summary:
            return summary
    except Exception:
        logger.warning(
            "LLM node signature summarization failed; falling back to raw "
            "node signatures",
            exc_info=True,
        )
    return raw_text


async def resume_conversation_after_workflow(
    *,
    event_service: EventService,
    task_id: str,
    status: RunWorkflowStatus,
    error_log: str | None = None,
    auto_run: bool = True,
    from_workflow_debug: bool = False,
) -> None:
    """Send a visible notification with hidden LLM context on workflow completion.

    A single ``MessageEvent`` (``source="agent"``) carries both:
    - ``llm_message.content`` — user-friendly status text visible to the frontend
    - ``extended_content`` — the ``<system_reminder>`` merged into the LLM prompt
      via ``to_llm_message()``, invisible to the frontend

    For failed debug runs, node signature guidance is appended to the reminder
    (condensed by the LLM, or raw signatures when summarization fails).

    一次调用发送一条可见事件，同时携带隐藏的 ``<system_reminder>``
    （通过 ``extended_content`` 注入 LLM 上下文，前端不可见）。
    避免两次发事件导致 agent 产生空响应。
    """
    guidance: str | None = None
    if from_workflow_debug and status in ("Failed", "Error"):
        guidance = await build_node_signature_guidance(
            event_service=event_service,
            error_log=error_log,
        )
    if status in ("Failed", "Error", "Terminated"):
        logger.info(
            "Workflow terminal failure task_id=%s status=%s\n"
            "--- decoded error_log ---\n%s\n"
            "--- node signature guidance ---\n%s",
            task_id,
            status,
            error_log or "(none)",
            guidance or "(none)",
        )
    visible_text = _build_workflow_completion_text(
        task_id=task_id,
        status=status,
        error_log=error_log,
        from_workflow_debug=from_workflow_debug,
    )
    reminder = build_run_workflow_terminal_reminder(
        task_id=task_id,
        status=status,
        error_log=error_log,
        from_workflow_debug=from_workflow_debug,
        node_signature_guidance=guidance,
    )
    await event_service.send_internal_context(
        [TextContent(text=visible_text)],
        run=auto_run,
        visible=True,
        extended_content=[TextContent(text=reminder)],
    )


def _build_workflow_completion_text(
    *,
    task_id: str,
    status: RunWorkflowStatus,
    error_log: str | None = None,
    from_workflow_debug: bool = False,
) -> str:
    """Build a user-friendly visible summary of a completed workflow."""
    if from_workflow_debug:
        if status == "Succeeded":
            return f"工作流调试运行成功\n\n- task_id: {task_id}\n- status: {status}"
        if status in ("Failed", "Error"):
            lines = [
                "工作流调试运行失败",
                "",
                f"- task_id: {task_id}",
                f"- status: {status}",
            ]
            if error_log:
                lines.append(f"- error_log: {error_log}")
            return "\n".join(lines)
        return f"工作流调试运行已终止\n\n- task_id: {task_id}\n- status: {status}"

    # Production path
    if status == "Succeeded":
        return f"工作流已完成\n\n- task_id: {task_id}\n- status: {status}"
    if status in ("Failed", "Error"):
        lines = [
            "工作流运行失败",
            "",
            f"- task_id: {task_id}",
            f"- status: {status}",
        ]
        if error_log:
            lines.append(f"- error_log: {error_log}")
        return "\n".join(lines)
    return f"工作流运行已终止\n\n- task_id: {task_id}\n- status: {status}"


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
    from_workflow_debug: bool = False,
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
        from_workflow_debug: True when the task was submitted by ``workflow_debug``
            (``out_id`` contains ``agent1#debug#``). Selects debug-only reminder text.
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

    # Step 7: A successful debug round ends here — reset the round counter so
    # the next debug session starts with a fresh attempt budget.
    if from_workflow_debug and normalized_status == "Succeeded":
        _reset_workflow_attempt_counter(event_service)

    # Step 8: Drop the task from the conversation's in-flight list. A task the
    # user already stopped via interrupt is marked ``Stopped`` and must NOT
    # re-wake the conversation (its terminal callback is just the stop echo).
    removed_task = await event_service.remove_active_long_task(task_id)
    if removed_task is not None and removed_task.status == "Stopped":
        auto_run = False

    # Step 9: Deliver to conversation and auto_run.
    try:
        resume_t0 = time.monotonic()
        await resume_conversation_after_workflow(
            event_service=event_service,
            task_id=task_id,
            status=normalized_status,
            error_log=error_log,
            auto_run=auto_run,
            from_workflow_debug=from_workflow_debug,
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
