from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pyromind_runtime.domain.content import ContentBlock, JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_chat import (
    CHAT_PREFIX,
    CHAT_PROFILE,
    MANAGER_PROFILE,
    PUBLIC_UPDATE_PREFIX,
)
from pyromind_runtime.domain.events import ProductEvent
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    ExecutionNoticeLink,
    TimelineMessage,
    TimelineOperation,
)
from pyromind_runtime.ports.harness import SessionHandle


if TYPE_CHECKING:
    from pyromind_runtime.application.conversation_runtime import ConversationRuntime
    from pyromind_runtime.infrastructure.file_product_store import FileProductStore


logger = logging.getLogger(__name__)
_NOTICE_PREFIX = "execution-notice:"
_NOTICE_EVENTS = frozenset(
    {
        "status.changed",
        "preparation.changed",
        "input.requested",
        "permission.requested",
        "experiment_execution.changed",
        "external_task.completed",
        "artifact.registered",
    }
)
_EXECUTION_STATES = frozenset({"waiting_analysis", "succeeded"})
_PLATFORM_STATES = frozenset({"succeeded", "failed", "terminated", "stopped"})
_NO_RESULT = "本轮执行已结束，执行记录中没有可展示的结果正文。"
_NO_RESULT_EN = "This round ended without a recorded user-facing result."


@runtime_checkable
class NoticeComposer(Protocol):
    async def compose_execution_notice(
        self, handle: SessionHandle, facts: JsonObject
    ) -> str: ...


@dataclass(frozen=True)
class Candidate:
    key: str
    facts: JsonObject
    link: ExecutionNoticeLink
    chinese_text: str | None = None
    english_text: str | None = None

    def current(self, snapshot: ConversationSnapshot) -> bool:
        if self.link.action == "plan":
            return (
                plan_ready(snapshot)
                and snapshot.preparation is not None
                and str(snapshot.preparation.revision) == self.link.target_id
            )
        if self.link.action == "confirmation":
            return any(
                request.request_id == self.link.target_id
                for request in snapshot.pending_inputs
            ) or any(
                request.permission_id == self.link.target_id
                for request in snapshot.pending_permissions
            )
        return True


def plan_ready(snapshot: ConversationSnapshot) -> bool:
    plan = snapshot.preparation
    return bool(
        plan is not None
        and plan.phase == "ready"
        and not plan.review_pending
        and not snapshot.project
        and not snapshot.automatic_turns_suspended
        and snapshot.status in {"idle", "finished"}
        and not snapshot.experiment_queue
        and not snapshot.pending_inputs
        and not snapshot.pending_permissions
    )


def latest_origin(snapshot: ConversationSnapshot) -> str | None:
    latest = max(
        (
            run
            for run in (snapshot.task_runs or {}).values()
            if run.task_id in snapshot.task_origins
        ),
        key=lambda run: run.seq,
        default=None,
    )
    return snapshot.task_origins[latest.task_id] if latest else None


def origin_for(
    snapshot: ConversationSnapshot, task_id: str | None, run_id: str | None
) -> str | None:
    runs = snapshot.task_runs or {}
    external = {task.task_id: task.run_id for task in snapshot.external_tasks}
    if not task_id and run_id is not None and run_id in runs:
        task_id = runs[run_id].task_id
    seen: set[str] = set()
    while task_id and task_id not in seen:
        if turn := snapshot.task_origins.get(task_id):
            return turn
        seen.add(task_id)
        parent = external.get(task_id) or run_id
        run_id = None
        task_id = runs[parent].task_id if parent and parent in runs else None
    project = snapshot.project
    return (
        snapshot.task_origins.get(f"project-start:{project.project_id}")
        if project
        else None
    )


def report_candidate(artifact_id: str, origin: str | None) -> Candidate:
    return Candidate(
        f"report:{artifact_id}",
        {"milestone": "report_ready", "report_available": True},
        ExecutionNoticeLink(
            origin_turn_id=origin, action="report", target_id=artifact_id
        ),
        "报告已生成，点击「查看报告」阅读。",
        "The report is ready. Select View report to read it.",
    )


def report_save_operation(
    snapshot: ConversationSnapshot, artifact_id: str
) -> TimelineOperation | None:
    return next(
        (
            item
            for item in reversed(snapshot.timeline)
            if isinstance(item, TimelineOperation)
            and item.name == "product_versions"
            and item.status == "completed"
            and isinstance(item.arguments, dict)
            and item.arguments.get("action") == "save"
            and isinstance(item.details, dict)
            and item.details.get("id") == artifact_id
        ),
        None,
    )


def _authorization_failure(event: ProductEvent) -> bool:
    return (
        event.type == "operation.failed"
        and event.payload.get("error_code") == "sandbox_authorization_required"
    )


def candidate_for(
    event: ProductEvent, snapshot: ConversationSnapshot, readable_reports: set[str]
) -> Candidate | None:
    run = (snapshot.task_runs or {}).get(event.run_id) if event.run_id else None
    task_id = event.task_id or (run.task_id if run else None)
    origin = origin_for(snapshot, task_id, event.run_id)
    if origin and _authorization_failure(event):
        return Candidate(
            f"sandbox-auth:{event.run_id or event.event_id}",
            {"milestone": "sandbox_authorization_required"},
            ExecutionNoticeLink(origin_turn_id=origin),
            "执行暂时受阻，沙箱需要登录。请在本会话的凭据设置中应用有效的登录后继续。",
            "Execution is blocked waiting for sandbox access. Apply your current "
            "login in this conversation’s credential settings to continue.",
        )
    if event.type == "artifact.registered":
        artifact_id = event.payload.get("artifact_id")
        if (
            event.payload.get("kind") != "report"
            or not isinstance(artifact_id, str)
            or artifact_id not in readable_reports
        ):
            return None
        if not event.run_id:
            save = report_save_operation(snapshot, artifact_id)
            if save:
                origin = origin_for(snapshot, save.task_id, save.run_id)
        return report_candidate(artifact_id, origin)

    if event.type in {"input.requested", "permission.requested"}:
        asking = event.type == "input.requested"
        field = "request_id" if asking else "permission_id"
        identity = event.payload.get(field)
        if not isinstance(identity, str):
            return None
        facts: JsonObject = {
            "milestone": "information_required" if asking else "permission_required"
        }
        if asking:
            request = next(
                (
                    item
                    for item in snapshot.pending_inputs
                    if item.request_id == identity
                ),
                None,
            )
            if request:
                facts["questions"] = [
                    prompt
                    for question in request.questions
                    if (prompt := safe_question(question.prompt))
                ]
        return Candidate(
            f"{field}:{identity}",
            facts,
            ExecutionNoticeLink(
                origin_turn_id=origin, action="confirmation", target_id=identity
            ),
            "执行需要你回复，请查看待确认事项。",
            "I need a response to continue. Please review the pending request.",
        )

    plan_event = event.type == "preparation.changed" or (
        event.type == "status.changed"
        and event.payload.get("status") in ("idle", "finished")
    )
    if plan_event and plan_ready(snapshot):
        plan = snapshot.preparation
        assert plan is not None
        return Candidate(
            f"plan:{plan.revision}",
            {"milestone": "plan_ready", "requires_user_confirmation": True},
            ExecutionNoticeLink(
                origin_turn_id=origin or latest_origin(snapshot),
                action="plan",
                target_id=str(plan.revision),
            ),
            "工作流计划已就绪，请确认后启动。",
            "The workflow plan is ready. Please review and confirm it to start.",
        )

    if event.type == "experiment_execution.changed":
        execution = event.payload.get("execution")
        if not isinstance(execution, dict):
            return None
        state, run = execution.get("state"), execution.get("run_id")
        if not isinstance(state, str) or not isinstance(run, str):
            return None
        platform = execution.get("platform_task_id")
        origin = (
            origin_for(
                snapshot, platform if isinstance(platform, str) else None, event.run_id
            )
            or origin
        )
        report_state = execution.get("report_state")
        report_ready = state == "reported" or (
            state == "succeeded" and report_state == "ready"
        )
        stage = execution.get("stage_id")
        project = snapshot.project
        stage_name = (
            next((step.name for step in project.plan.stages if step.id == stage), None)
            if project
            else None
        )
        stage_facts: JsonObject = (
            {"stage": label}
            if stage_name and (label := safe_question(stage_name))
            else {}
        )
        if state == "failed" and not platform and report_state != "failed":
            # Submission validation can fail inside a still-running turn which
            # can correct the parameters itself. User decisions have separate
            # input/permission events; the final turn still delivers its result.
            if snapshot.status == "running" or any(
                r.run_id == run and r.state != "failed"
                for r in snapshot.experiment_executions or ()
            ):
                return None
        if report_ready:
            verified = next(
                (
                    item
                    for item in snapshot.experiment_executions or ()
                    if item.run_id == run
                ),
                None,
            )
            if verified and not (
                verified.state == "reported"
                or (verified.state == "succeeded" and verified.report_state == "ready")
            ):
                return None
            artifacts = execution.get("artifacts")
            if isinstance(artifacts, list):
                matching = [
                    item["artifact_id"]
                    for item in artifacts
                    if isinstance(item, dict)
                    and isinstance(item.get("artifact_id"), str)
                    and item["artifact_id"] in readable_reports
                    and (
                        not execution.get("report_uri")
                        or item.get("uri") == execution["report_uri"]
                    )
                ]
                if len(matching) == 1:
                    return report_candidate(matching[0], origin)
            return Candidate(
                f"run:{run}:report",
                {
                    "milestone": "report_ready",
                    "report_available": bool(execution.get("report_uri")),
                    **stage_facts,
                },
                ExecutionNoticeLink(origin_turn_id=origin),
                "执行报告已就绪，可在执行详情中查看。",
                "The execution report is ready in the execution details.",
            )
        if (
            state in {"failed", "needs_input", "waiting_authorization"}
            or report_state == "failed"
        ):
            milestone = "report_failed" if report_state == "failed" else state
            failure = (
                f"platform:{platform}:failed"
                if state == "failed" and isinstance(platform, str)
                else f"run:{run}:{milestone}"
            )
            return Candidate(
                failure,
                {"milestone": milestone, **stage_facts},
                ExecutionNoticeLink(origin_turn_id=origin),
                "执行遇到阻塞，请查看执行详情。",
                "Execution is blocked. See the execution details.",
            )
        if state in _EXECUTION_STATES:
            return Candidate(
                f"platform:{platform}:succeeded"
                if isinstance(platform, str)
                else f"run:{run}:{state}",
                {"milestone": state, **stage_facts},
                ExecutionNoticeLink(origin_turn_id=origin),
            )
        return None

    if event.type == "external_task.completed":
        task, status = event.payload.get("task_id"), event.payload.get("status")
        if (
            not isinstance(task, str)
            or not isinstance(status, str)
            or status not in _PLATFORM_STATES
        ):
            return None
        return Candidate(
            f"platform:{task}:{status}",
            {
                "milestone": "platform_result",
                "status": status,
                "report_available": False,
            },
            ExecutionNoticeLink(
                origin_turn_id=origin_for(snapshot, task, event.run_id)
            ),
            "平台任务执行失败，请查看执行详情。" if status == "failed" else None,
            "The platform task failed. See the execution details."
            if status == "failed"
            else None,
        )

    if (
        event.type == "status.changed"
        and event.payload.get("status") in ("error", "stuck")
        and origin
    ):
        return Candidate(
            f"attempt:{event.run_id or event.event_id}:blocked",
            {"milestone": "execution_blocked"},
            ExecutionNoticeLink(origin_turn_id=origin),
            "本轮执行遇到问题，请查看执行详情。",
            "This run could not finish. See the execution details.",
        )
    return round_completion_candidate(event, snapshot)


def resolved_submission_notices(
    manager: ConversationSnapshot, chat: ConversationSnapshot
) -> list[ProductEvent]:
    # Only the pre-submission failure uses run:<attempt>:failed. Once this
    # exact attempt has a bound receipt, its original blocker is historical.
    notices = {
        item.item_id: item
        for item in chat.timeline
        if isinstance(item, TimelineMessage) and item.execution_notice
    }
    updates = []
    for step in manager.experiment_executions or ():
        if step.role_id != "main" or not step.platform_task_id:
            continue
        key = f"run:{step.run_id}:failed"
        message_id = _NOTICE_PREFIX + sha256(key.encode()).hexdigest()
        item = notices.get(message_id)
        if not item or not item.execution_notice:
            continue
        chinese = any(
            isinstance(b, TextContent) and re.search(r"[\u3400-\u9fff]", b.text)
            for b in item.content
        )
        text = (
            "此前提交遇到问题，之后已成功提交到平台。"
            if chinese
            else (
                "Submission initially failed. The platform later accepted this attempt."
            )
        )
        if item.content == (TextContent(text=text),):
            continue
        updates.append(
            ProductEvent(
                conversation_id=chat.conversation_id,
                event_id=message_id + ":submitted",
                type="message.completed",
                run_id=item.run_id,
                payload={
                    "role": "assistant",
                    "message_id": message_id,
                    "content": [{"type": "text", "text": text}],
                    "execution_notice": item.execution_notice.model_dump(
                        mode="json", exclude_none=True
                    ),
                },
            )
        )
    return updates


def direct_project_receipts(
    snapshot: ConversationSnapshot,
) -> dict[str, tuple[str, int]]:
    """Link a direct project resume to chat only through a bound main submission."""
    origins = {
        step.platform_task_id: origin
        for step in snapshot.experiment_executions or ()
        if step.role_id == "main"
        and step.platform_task_id
        and (origin := snapshot.task_origins.get(f"project-start:{step.project_id}"))
    }
    runs: dict[str, tuple[str, int]] = {}
    for op in snapshot.timeline:
        if (
            isinstance(op, TimelineOperation)
            and op.run_id
            and not op.task_id
            and op.run_id not in (snapshot.task_runs or {})
            and op.status == "completed"
            and op.completed_seq
            and op.name in {"df_submit_evaluation", "submit_workflow"}
            and isinstance(op.details, dict)
            and isinstance(receipt := op.details.get("task_id"), str)
            and (origin := origins.get(receipt))
        ):
            runs.setdefault(op.run_id, (origin, op.completed_seq))
    return runs


def round_completion_candidate(
    event: ProductEvent, snapshot: ConversationSnapshot
) -> Candidate | None:
    run = (snapshot.task_runs or {}).get(event.run_id) if event.run_id else None
    task_id = event.task_id or (run.task_id if run else None)
    if (
        event.type != "status.changed"
        or event.payload.get("status") not in ("idle", "finished")
        or not event.run_id
    ):
        return None
    if task_id:
        origin = origin_for(snapshot, task_id, event.run_id)
        linked = task_id in snapshot.task_origins or any(
            task.task_id == task_id and task.run_id for task in snapshot.external_tasks
        )
    else:
        direct = direct_project_receipts(snapshot).get(event.run_id)
        origin = direct[0] if direct and direct[1] < event.seq else None
        linked = bool(origin)
    if linked and origin:
        followup = any(
            item.get("kind") == "project_continue"
            and item.get("origin_turn_id") == origin
            for item in snapshot.experiment_queue or ()
        )
        facts: JsonObject = {
            "milestone": "execution_round_finished",
            "continuation_queued": followup,
            "waiting_for_platform": any(
                task.run_id == event.run_id
                and task.status in {"pending", "running", "waiting"}
                for task in snapshot.external_tasks
            ),
        }
        return Candidate(
            f"round:{event.run_id}",
            facts,
            ExecutionNoticeLink(origin_turn_id=origin, has_result=False),
            "本轮执行已结束，已排队继续下一步。" if followup else _NO_RESULT,
            "This round has ended and the next step is queued."
            if followup
            else _NO_RESULT_EN,
        )
    return None


# Backticks quote literal tag names in public format documentation. A bare
# closing tag can end a provider reasoning prefix whose opening tag was lost.
_REASONING_TAG = re.compile(
    r"(?P<literal>^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n"
    r".*?^[ \t]{0,3}(?P=fence)[ \t]*$"
    r"|(?<!`)(?P<ticks>`+)(?!`)[^\n]*?(?<!`)(?P=ticks)(?!`))"
    r"|<(?P<close>/)?(?:think|thinking|analysis|reasoning)\b[^>]*>",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def public_blocks(
    blocks: tuple[ContentBlock, ...], *, publish: bool
) -> tuple[ContentBlock, ...]:
    if not publish:
        return ()
    visible: list[ContentBlock] = []
    hidden = False
    for block in blocks:
        if not isinstance(block, TextContent):
            if not hidden:
                visible.append(block)
            continue
        start = 0
        for tag in _REASONING_TAG.finditer(block.text):
            if tag["literal"]:
                continue
            if tag["close"]:
                # Even an apparent opener can be an example inside an unmarked
                # reasoning prefix. Only the suffix after a close is public.
                visible.clear()
                hidden = False
            else:
                if not hidden and block.text[start : tag.start()].strip():
                    visible.append(TextContent(text=block.text[start : tag.start()]))
                hidden = True
            start = tag.end()
        if not hidden and block.text[start:]:
            visible.append(TextContent(text=block.text[start:]))
    if (
        hidden
        or not visible
        or not any(
            not isinstance(block, TextContent) or block.text.strip()
            for block in visible
        )
    ):
        return ()
    return tuple(visible)


def visible_chat_snapshot(snapshot: ConversationSnapshot) -> ConversationSnapshot:
    return snapshot.model_copy(
        update={
            "timeline": tuple(
                item
                for item in snapshot.timeline
                if not (
                    isinstance(item, TimelineMessage)
                    and item.execution_notice
                    and item.execution_notice.source_message_id
                    and (not item.execution_notice.public_copy or not item.content)
                )
            )
        }
    )


def visible_chat_event(event: ProductEvent) -> ProductEvent:
    link = event.payload.get("execution_notice")
    if (
        isinstance(link, dict)
        and link.get("source_message_id")
        and not link.get("public_copy")
    ):
        return event.model_copy(update={"payload": {**event.payload, "content": []}})
    return event


def manager_messages(
    manager: ConversationSnapshot,
    chat: ConversationSnapshot,
    through_seq: int,
    previous_cursor: int,
    ended_runs: frozenset[str] = frozenset(),
) -> tuple[dict[int, ProductEvent], set[str]]:
    """Mirror completed assistant messages and reuse legacy round bubbles."""
    messages = sorted(
        (
            item
            for item in manager.timeline
            if isinstance(item, TimelineMessage)
            and item.role == "assistant"
            and item.run_id
            and item.status == "completed"
            and item.completed_seq is not None
            and item.completed_seq <= through_seq
            and any(
                not isinstance(block, TextContent) or block.text.strip()
                for block in item.content
            )
        ),
        key=lambda item: item.completed_seq or 0,
    )
    latest = {item.run_id: item.item_id for item in messages}
    existing = {
        item.item_id: item
        for item in chat.timeline
        if isinstance(item, TimelineMessage) and item.execution_notice
    }
    copies = {
        item.execution_notice.source_message_id: item
        for item in existing.values()
        if item.execution_notice and item.execution_notice.source_message_id
    }
    last_event: dict[str, int] = {}
    for entry in manager.timeline:
        if isinstance(entry, (TimelineMessage, TimelineOperation)) and entry.run_id:
            last_event[entry.run_id] = max(
                last_event.get(entry.run_id, 0),
                entry.completed_seq or entry.started_seq,
            )
    direct = direct_project_receipts(manager)
    spoken = set()
    updates = {}
    for item in messages:
        assert item.run_id and item.completed_seq
        run = (manager.task_runs or {}).get(item.run_id)
        legacy = _NOTICE_PREFIX + sha256(f"round:{item.run_id}".encode()).hexdigest()
        terminal = (
            (item.run_id in ended_runs)
            or (run and run.seq <= through_seq and run.status in {"idle", "finished"})
            or (legacy in existing)
            or (item.run_id in direct and manager.status in {"idle", "finished"})
        )
        final = bool(terminal and item.completed_seq == last_event[item.run_id])
        prior = copies.get(item.item_id)
        if prior and prior.execution_notice and prior.execution_notice.public_copy:
            if prior.content:
                spoken.add(item.run_id)
                continue
            if not final:
                continue
        content = public_blocks(
            item.content, publish=final or item.item_id.startswith(PUBLIC_UPDATE_PREFIX)
        )
        if content:
            spoken.add(item.run_id)
        elif prior and prior.execution_notice and prior.execution_notice.public_copy:
            continue
        if not content and prior is None:
            continue
        key = ("public-message:" if content else "withdraw-message:") + item.item_id
        event_id = _NOTICE_PREFIX + sha256(key.encode()).hexdigest()
        message_id = prior.item_id if prior else event_id
        origin = origin_for(manager, item.task_id, item.run_id)
        if not prior and latest[item.run_id] == item.item_id:
            prior_round = existing.get(legacy)
            if (
                prior_round
                and prior_round.execution_notice
                and not prior_round.execution_notice.source_message_id
            ):
                message_id = legacy
                origin = prior_round.execution_notice.origin_turn_id or origin
        link = ExecutionNoticeLink(
            origin_turn_id=origin,
            public_copy=True,
            has_result=bool(content),
            source_message_id=item.item_id,
            history_timestamp=item.started_at
            if (item.completed_seq or 0) <= previous_cursor
            else None,
        )
        updates[item.completed_seq or 0] = ProductEvent(
            conversation_id=chat.conversation_id,
            event_id=event_id,
            type="message.completed",
            run_id=item.run_id,
            payload={
                "role": "assistant",
                "message_id": message_id,
                "content": [block.model_dump(mode="json") for block in content],
                "execution_notice": link.model_dump(mode="json", exclude_none=True),
            },
        )
    return updates, spoken


def safe_question(prompt: str) -> str | None:
    if len(prompt) > 240 or re.search(
        r"://|[/\\<>]|[A-Za-z0-9_-]{24,}|api.?key|token|cookie|password|密钥|密码",
        prompt,
        re.IGNORECASE,
    ):
        return None
    return prompt


def public_message(raw: str, *, max_length: int = 280) -> str | None:
    content = raw.strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[len("```json") : -3].strip()
    try:
        data = json.loads(content)
    except ValueError:
        return None
    message = data.get("message") if isinstance(data, dict) else None
    if not isinstance(message, str):
        return None
    message = message.strip()
    if (
        not message
        or len(message) > max_length
        or re.search(
            r"[<>]|://|\]\(|\b(?:main|chat)\s+agent\b|(?:api.?key|password|token|cookie|authorization|secret)\s*[:=]",
            message,
            re.IGNORECASE,
        )
    ):
        return None
    return message


class EchoMindNotices:
    def __init__(self, runtime: ConversationRuntime) -> None:
        self.runtime = runtime
        self.workers: dict[str, asyncio.Task[None]] = {}
        self.observed: dict[str, dict[int, ProductEvent]] = {}
        self.closed = False

    def observe(self, event: ProductEvent) -> None:
        manager_id = event.conversation_id
        if (
            not self.closed
            and (
                event.type in _NOTICE_EVENTS
                or _authorization_failure(event)
                or (
                    event.type == "message.completed"
                    and event.run_id
                    and event.payload.get("role", "assistant") == "assistant"
                    and event.payload.get("content", True)
                )
            )
            and not manager_id.startswith(CHAT_PREFIX)
            and self.runtime._store(CHAT_PREFIX + manager_id).metadata_path.is_file()
        ):
            self.observed.setdefault(manager_id, {})[event.seq] = event
            self.schedule(manager_id)

    def schedule(self, manager_id: str) -> None:
        if not self.closed and manager_id not in self.workers:
            self.workers[manager_id] = asyncio.create_task(self._work(manager_id))

    async def close(self) -> None:
        self.closed = True
        workers = tuple(self.workers.values())
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    async def _work(self, manager_id: str) -> None:
        try:
            manager = self.runtime._store(manager_id)
            chat_id = CHAT_PREFIX + manager_id
            chat = self.runtime._store(chat_id)
            owner = await asyncio.to_thread(chat.owner_id)
            await asyncio.to_thread(chat.authorize, owner, CHAT_PROFILE)
            await asyncio.to_thread(manager.authorize, owner, MANAGER_PROFILE)
            context = RequestContext(user_id=owner, profile_id=MANAGER_PROFILE)
            snapshot = await asyncio.to_thread(manager.load_snapshot)
            first_observed = min(
                self.observed.get(manager_id, {}), default=snapshot.through_seq + 1
            )
            baseline = max(0, first_observed - 1)
            cursor = await asyncio.to_thread(chat.execution_notice_cursor, baseline)
            if manager_id not in self.observed and snapshot.task_runs:
                chat_snapshot = await asyncio.to_thread(chat.load_snapshot)
                for update in resolved_submission_notices(snapshot, chat_snapshot):
                    saved, _ = await asyncio.to_thread(chat.append, update)
                    self.runtime._publish(saved)
            while True:
                events = await asyncio.to_thread(manager.replay, cursor)
                fresh = self.observed.pop(manager_id, {})
                fresh.update(
                    (event.seq, event)
                    for event in events
                    if event.type in _NOTICE_EVENTS or _authorization_failure(event)
                )
                through_seq = events[-1].seq if events else cursor
                candidates = [
                    event for seq, event in fresh.items() if seq <= through_seq
                ]
                for seq, event in fresh.items():
                    if seq > through_seq:
                        self.observed.setdefault(manager_id, {})[seq] = event
                await self._deliver(
                    manager,
                    chat,
                    candidates,
                    manager_id,
                    chat_id,
                    context,
                    through_seq,
                    cursor,
                )
                if events:
                    cursor = events[-1].seq
                    await asyncio.to_thread(chat.save_execution_notice_cursor, cursor)
                if manager_id not in self.observed:
                    return
        except Exception as exc:
            logger.warning("Could not process EchoMind notices: %s", type(exc).__name__)
        finally:
            self.observed.pop(manager_id, None)
            self.workers.pop(manager_id, None)

    async def _deliver(
        self,
        manager: FileProductStore,
        chat: FileProductStore,
        events: list[ProductEvent],
        manager_id: str,
        chat_id: str,
        context: RequestContext,
        through_seq: int,
        previous_cursor: int,
    ) -> None:
        snapshot = await self.runtime._load_saved_snapshot(
            manager_id, manager, context, recover_inputs=False
        )
        snapshot = await self.runtime._project_execution_status(snapshot, manager)
        readable_reports = (
            {
                version.id
                for version in await asyncio.to_thread(
                    self.runtime._artifact_versions, manager_id, context
                )
                if version.kind == "report" and version.state == "ready"
            }
            if any(
                event.type in {"artifact.registered", "experiment_execution.changed"}
                for event in events
            )
            else set()
        )
        chat_snapshot = await asyncio.to_thread(chat.load_snapshot)
        messages, spoken_runs = manager_messages(
            snapshot,
            chat_snapshot,
            through_seq,
            previous_cursor,
            frozenset(
                e.run_id
                for e in events
                if e.type == "status.changed"
                and e.payload.get("status") in ("idle", "finished")
                and e.run_id
            ),
        )
        milestones = {
            event.seq: event for event in events if event.type != "message.completed"
        }
        if not milestones and not messages:
            return
        chat_events = await asyncio.to_thread(chat.replay) if milestones else ()
        considered = {
            event.event_id: event
            for event in chat_events
            if event.event_id.startswith(_NOTICE_PREFIX)
        }
        delivered_runs = {
            event.run_id
            for event in chat_events
            if event.event_id.startswith(_NOTICE_PREFIX)
            and event.run_id
            and isinstance(link := event.payload.get("execution_notice"), dict)
            and link.get("action") == "report"
        }
        for seq in sorted(messages.keys() | milestones.keys()):
            if original := messages.get(seq):
                persisted, _ = await asyncio.to_thread(chat.append, original)
                self.runtime._publish(persisted)
                continue
            event = milestones[seq]
            candidate = candidate_for(event, snapshot, readable_reports)
            if candidate is None or not candidate.current(snapshot):
                continue
            if candidate.key.startswith("round:") and (
                event.run_id in delivered_runs or event.run_id in spoken_runs
            ):
                continue
            identity = _NOTICE_PREFIX + sha256(candidate.key.encode()).hexdigest()
            if identity in considered:
                continue
            chat_snapshot = await asyncio.to_thread(chat.load_snapshot)
            users = [
                item
                for item in reversed(chat_snapshot.timeline)
                if isinstance(item, TimelineMessage) and item.role == "user"
            ]
            user = next(
                (
                    item
                    for item in users
                    if item.turn_id == candidate.link.origin_turn_id
                ),
                users[0] if users else None,
            )
            english = (
                user is not None
                and any(
                    isinstance(block, TextContent)
                    and re.search(r"[A-Za-z]", block.text)
                    for block in user.content
                )
                and not any(
                    isinstance(block, TextContent)
                    and re.search(r"[\u3400-\u9fff]", block.text)
                    for block in user.content
                )
            )
            fallback = candidate.english_text if english else candidate.chinese_text
            facts: JsonObject = {
                **candidate.facts,
                "language": "English" if english else "Chinese",
                "notification_required": fallback is not None,
            }
            if candidate.key.startswith(("round:", "sandbox-auth:")):
                text = fallback
            else:
                text = await self._compose(chat_id, context, facts) or fallback
            if candidate.link.action in {"plan", "confirmation"} and text:
                latest = await self.runtime._load_saved_snapshot(
                    manager_id, manager, context, recover_inputs=False
                )
                if not candidate.current(latest):
                    continue
            notice = ProductEvent(
                conversation_id=chat_id,
                event_id=identity,
                run_id=event.run_id,
                type="message.completed" if text else "chat.notification.skipped",
                payload={
                    "role": "assistant",
                    "message_id": identity,
                    "content": [{"type": "text", "text": text}],
                    "execution_notice": candidate.link.model_dump(
                        mode="json", exclude_none=True
                    ),
                }
                if text
                else {},
            )
            persisted, _ = await asyncio.to_thread(chat.append, notice)
            self.runtime._publish(persisted)
            considered[identity] = persisted
            if text and candidate.link.action == "report" and event.run_id:
                delivered_runs.add(event.run_id)

    async def _compose(
        self, chat_id: str, context: RequestContext, facts: JsonObject
    ) -> str | None:
        try:
            async with asyncio.timeout(12):
                active = self.runtime._active.get(chat_id)
                if active is None or not active.ready.is_set() or active.stopping:
                    await self.runtime._ensure_active(
                        chat_id,
                        active.context
                        if active
                        else replace(context, profile_id=CHAT_PROFILE),
                    )
                async with self.runtime._use(chat_id) as active:
                    if isinstance(active.adapter, NoticeComposer):
                        return public_message(
                            await active.adapter.compose_execution_notice(
                                active.handle, facts
                            ),
                            max_length=900 if facts.get("result_evidence") else 280,
                        )
        except Exception as exc:
            logger.warning(
                "EchoMind notice composition unavailable: %s", type(exc).__name__
            )
        finally:
            self.runtime._schedule_release(chat_id)
        return None
