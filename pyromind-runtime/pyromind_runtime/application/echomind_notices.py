from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_chat import (
    CHAT_PREFIX,
    CHAT_PROFILE,
    MANAGER_PROFILE,
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
_ROUND_END = "本轮执行已结束。可查看原任务中的执行详情。"
_ROUND_END_EN = "This execution round has ended. Open the original task for details."
_NO_RESULT = "本轮执行已结束，执行记录中没有可展示的结果正文。"
_NO_RESULT_EN = "This round ended without a recorded user-facing result."
_QUOTED_REASONING_TAG = re.compile(
    r"(?<!`)`<(/?(?:think|thinking|reasoning|analysis))>`(?!`)", re.IGNORECASE
)


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
    report_excerpt: str | None = None

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


def completion_summary(
    snapshot: ConversationSnapshot, event: ProductEvent
) -> str | None:
    final = max(
        (
            item
            for item in snapshot.timeline
            if isinstance(item, (TimelineMessage, TimelineOperation))
            and item.run_id == event.run_id
            and (item.completed_seq or item.started_seq) < event.seq
        ),
        key=lambda item: item.completed_seq or item.started_seq,
        default=None,
    )
    if (
        not isinstance(final, TimelineMessage)
        or final.role != "assistant"
        or final.status != "completed"
    ):
        return None
    text = "\n".join(
        block.text.strip() for block in final.content if isinstance(block, TextContent)
    ).strip()
    # A standalone tag quoted as inline code names a data format in the report.
    text = _QUOTED_REASONING_TAG.sub(r"`\1`", text)
    text = re.sub(
        r"<(think|thinking|analysis|reasoning)\b[^>]*>.*?</\1\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if not text or re.search(
        r"<\s*/?\s*(?:think|thinking|analysis|reasoning)\b", text, re.IGNORECASE
    ):
        return None
    text = re.sub(r"https?://\S+", "[link omitted]", text)
    text = re.sub(r"\b[A-Za-z0-9_+=-]{24,}\b", "[redacted]", text)
    text = re.sub(
        r"(?i)((?:api.?key|password|token|cookie|authorization|secret|密钥|密码)\s*[:=]\s*)[^\n]+",
        "",
        text,
    )
    return text


def public_summary_excerpt(evidence: str) -> str | None:
    # Fenced examples and tool transcripts can contain apparent Markdown headings.
    prose = re.sub(
        r"^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?(?:^[ \t]*\1[ \t]*$|\Z)",
        "",
        evidence,
        flags=re.MULTILINE | re.DOTALL,
    )
    private_section = re.compile(
        r"working notes|chain of thought|reasoning|internal notes|raw output"
        r"|工作笔记|推理过程|内部|原始工具输出",
        re.IGNORECASE,
    )
    unsafe_line = re.compile(
        r"[<>]|\]\(|\[redacted\]|\[link omitted\]|\b(?:main|chat)\s+agent\b"
        r"|内部指令|密钥|密码",
        re.IGNORECASE,
    )
    hidden_depth = None
    visible = []
    for line in prose.splitlines():
        heading = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+)$", line)
        if heading:
            depth = len(heading[1])
            if hidden_depth is not None and depth > hidden_depth:
                continue
            hidden_depth = depth if private_section.search(heading[2]) else None
        elif private_section.match(line.strip("* :：\t")):
            hidden_depth = hidden_depth or 6
        if hidden_depth is None and not unsafe_line.search(line):
            visible.append(line)
    if not any(line.strip() and not line.lstrip().startswith("#") for line in visible):
        return None
    return "\n".join(visible).strip() or None


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
        evidence = completion_summary(snapshot, event)
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
        excerpt = public_summary_excerpt(evidence) if evidence else None
        return Candidate(
            f"round:{event.run_id}",
            facts,
            ExecutionNoticeLink(origin_turn_id=origin, has_result=bool(excerpt)),
            "本轮执行已结束，已排队继续下一步。" if followup else _NO_RESULT,
            "This round has ended and the next step is queued."
            if followup
            else _NO_RESULT_EN,
            report_excerpt=excerpt,
        )
    return None


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
            and (event.type in _NOTICE_EVENTS or _authorization_failure(event))
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
            historical: dict[int, ProductEvent] = {}
            if manager_id not in self.observed and snapshot.task_runs:
                chat_snapshot = await asyncio.to_thread(chat.load_snapshot)
                for update in resolved_submission_notices(snapshot, chat_snapshot):
                    saved, _ = await asyncio.to_thread(chat.append, update)
                    self.runtime._publish(saved)
                missing: set[str] = set()
                no_result: set[str] = set()
                for item in chat_snapshot.timeline:
                    if (
                        isinstance(item, TimelineMessage)
                        and item.execution_notice
                        and item.run_id
                        and item.run_id in snapshot.task_runs
                        and len(item.content) == 1
                        and isinstance(block := item.content[0], TextContent)
                    ):
                        if block.text in (_ROUND_END, _ROUND_END_EN):
                            missing.add(item.run_id)
                        elif block.text in (_NO_RESULT, _NO_RESULT_EN):
                            no_result.add(item.run_id)
                # A genuine no-result notice needs no replay. Check only runs
                # whose previous result may have contained a quoted tag.
                if no_result:
                    missing.update(
                        item.run_id
                        for item in snapshot.timeline
                        if isinstance(item, TimelineMessage)
                        and item.run_id
                        and item.run_id in no_result
                        and item.role == "assistant"
                        and any(
                            isinstance(block, TextContent)
                            and _QUOTED_REASONING_TAG.search(block.text)
                            for block in item.content
                        )
                    )
                starts = {run: snapshot.task_runs[run].seq for run in missing}
                delivered = {
                    item.run_id
                    for item in chat_snapshot.timeline
                    if isinstance(item, TimelineMessage)
                    and item.execution_notice
                    and (
                        item.execution_notice.has_result
                        or item.execution_notice.action == "report"
                    )
                }
                for run, (_, seq) in direct_project_receipts(snapshot).items():
                    if run not in delivered and run != snapshot.active_run_id:
                        starts[run] = seq
                if starts:
                    missing.update(starts)
                    first = min(starts.values())
                    historical = {
                        e.seq: e
                        for e in await asyncio.to_thread(manager.replay, first - 1)
                        if e.run_id in missing
                        and e.seq <= cursor
                        and e.type == "status.changed"
                        and e.payload.get("status") in {"idle", "finished"}
                    }
            while True:
                events = await asyncio.to_thread(manager.replay, cursor)
                fresh = self.observed.pop(manager_id, {})
                replayed_rounds = set(historical)
                fresh.update(historical)
                historical.clear()
                fresh.update(
                    (event.seq, event)
                    for event in events
                    if event.type in _NOTICE_EVENTS or _authorization_failure(event)
                )
                candidates = [fresh[seq] for seq in sorted(fresh)]
                if candidates:
                    await self._deliver(
                        manager,
                        chat,
                        candidates,
                        manager_id,
                        chat_id,
                        context,
                        replayed_rounds,
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
        replayed_rounds: set[int],
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
        chat_events = await asyncio.to_thread(chat.replay)
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
        for event in events:
            candidate = (
                round_completion_candidate(event, snapshot)
                if event.seq in replayed_rounds
                else candidate_for(event, snapshot, readable_reports)
            )
            if candidate is None or not candidate.current(snapshot):
                continue
            if candidate.key.startswith("round:") and event.run_id in delivered_runs:
                continue
            identity = _NOTICE_PREFIX + sha256(candidate.key.encode()).hexdigest()
            message_id = identity
            if prior := considered.get(identity):
                if not candidate.report_excerpt or prior.payload.get("content") not in (
                    [{"type": "text", "text": _ROUND_END}],
                    [{"type": "text", "text": _ROUND_END_EN}],
                    [{"type": "text", "text": _NO_RESULT}],
                    [{"type": "text", "text": _NO_RESULT_EN}],
                ):
                    continue
                identity += ":readable-report"
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
            fallback = candidate.report_excerpt or (
                candidate.english_text if english else candidate.chinese_text
            )
            facts: JsonObject = {
                **candidate.facts,
                "language": "English" if english else "Chinese",
                "notification_required": fallback is not None,
            }
            excerpt = candidate.report_excerpt
            if excerpt:
                text = excerpt
                if facts.get("continuation_queued"):
                    text += "\n\n" + (
                        "The next step is queued." if english else "下一步已排队。"
                    )
                elif facts.get("waiting_for_platform"):
                    text += "\n\n" + (
                        "Waiting for the platform result."
                        if english
                        else "正在等待平台结果。"
                    )
            elif candidate.key.startswith(("round:", "sandbox-auth:")):
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
                    "message_id": message_id,
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
