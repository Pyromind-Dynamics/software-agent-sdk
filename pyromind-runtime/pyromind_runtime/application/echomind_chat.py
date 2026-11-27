from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from pyromind_runtime.domain.commands import EnqueueTaskCommand
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.echomind_chat import (
    CHAT_INSTRUCTIONS,
    CHAT_PREFIX,
    CHAT_PROFILE,
    CHAT_REQUIREMENT_COMMAND_PREFIX,
    CHAT_TOOLS,
    ENQUEUE,
    GET_STATUS,
    MANAGER_PROFILE,
    READ_REPORT,
    EnqueueRequirement,
    GetStatus,
    ReadReport,
)
from pyromind_runtime.domain.errors import ProductRuntimeError
from pyromind_runtime.domain.snapshot import (
    ConversationSnapshot,
    TimelineMessage,
    TimelineOperation,
)
from pyromind_runtime.domain.storage_paths import to_storage_api_path
from pyromind_runtime.domain.versions import VersionToolRequest
from pyromind_runtime.infrastructure.file_product_store import ProductStoreError
from pyromind_runtime.ports.harness import SessionSpec


if TYPE_CHECKING:
    from pyromind_runtime.application.conversation_runtime import ConversationRuntime
    from pyromind_runtime.infrastructure.file_product_store import FileProductStore


def completed_chat_turns(
    snapshot: ConversationSnapshot,
    current: str,
    *,
    preparation: bool = False,
    budget: int = 32000,
) -> list[dict[str, str]]:
    questions: dict[str, str] = {}
    replies: dict[str, str] = {}
    latest: dict[str, int] = {}
    for item in snapshot.timeline:
        if not isinstance(item, (TimelineMessage, TimelineOperation)):
            continue
        turn = item.turn_id or item.run_id
        if isinstance(item, TimelineOperation):
            if turn:
                replies.pop(turn, None)
            continue
        if item.status != "completed" or (preparation and item.task_id):
            continue
        text = "\n".join(
            b.text for b in item.content if isinstance(b, TextContent)
        ).strip()
        if preparation and item.role == "assistant":
            text = item.preparation_reply or ""
        if not text:
            continue
        if item.execution_notice:
            continue
        if item.role == "user":
            key = item.command_id or turn or item.item_id
            if key != current:
                questions[key] = text
        elif item.role == "assistant" and turn and turn != current:
            replies[turn] = text
            latest[turn] = max(
                latest.get(turn, 0), item.completed_seq or item.started_seq
            )

    # Select recently answered turns (including queued requirements), then restore
    # original question order for the model. Never split a pair across the budget.
    selected: dict[str, dict[str, str]] = {}
    remaining = budget
    for key in sorted(questions, key=lambda k: latest.get(k, 0), reverse=True):
        if key not in replies:
            continue
        question = questions[key][:4000]
        answer = replies[key]
        if len(answer) > 8000:
            answer = answer[:4000] + "\n…\n" + answer[-3997:]
        size = len(question) + len(answer)
        if size <= remaining:
            selected[key] = {"question": question, "answer": answer}
            remaining -= size
    return [selected[key] for key in questions if key in selected]


class EchoMindChat:
    def __init__(self, runtime: ConversationRuntime) -> None:
        self._runtime = runtime

    async def _manager(
        self, manager_id: str, context: RequestContext
    ) -> tuple[FileProductStore, RequestContext]:
        context = replace(context, profile_id=MANAGER_PROFILE)
        store = await self._runtime._authorized_store(manager_id, context)
        project = await self._runtime.get_project(manager_id, context)
        if project is not None and project.cluster:
            context = replace(context, x_cluster=project.cluster)
        return store, context

    async def route(self, manager_id: str, context: RequestContext) -> RequestContext:
        _, context = await self._manager(manager_id, context)
        return replace(context, profile_id=CHAT_PROFILE)

    async def snapshot(
        self, manager_id: str, context: RequestContext
    ) -> ConversationSnapshot | None:
        context = await self.route(manager_id, context)
        chat_id = CHAT_PREFIX + manager_id
        if not self._runtime._store(chat_id).metadata_path.is_file():
            return None
        return await self._runtime.get_snapshot(chat_id, context)

    async def saved_updated_at(self, manager_id: str, user_id: str) -> datetime | None:
        store = self._runtime._store(CHAT_PREFIX + manager_id)
        if not store.metadata_path.is_file():
            return None

        def load_updated_at() -> datetime | None:
            try:
                store.authorize(user_id, CHAT_PROFILE)
                return store.load_snapshot().updated_at
            except (OSError, ValueError, ProductStoreError):
                return None

        return await asyncio.to_thread(load_updated_at)

    async def open(
        self, manager_id: str, model: JsonObject, context: RequestContext
    ) -> ConversationSnapshot:
        manager, context = await self._manager(manager_id, context)
        if manager.harness_id() not in self._runtime.adapters:
            raise ProductRuntimeError(
                "legacy_harness_retired", "The EchoMind conversation is archived."
            )
        chat_id = CHAT_PREFIX + manager_id
        context = replace(context, profile_id=CHAT_PROFILE)
        async with manager.lock_chat_creation():
            manager.authorize(context.user_id, MANAGER_PROFILE)
            if self._runtime._store(chat_id).metadata_path.is_file():
                return await self._runtime.get_snapshot(chat_id, context)
            await self._runtime.create_conversation(
                SessionSpec(
                    conversation_id=chat_id,
                    user_id=context.user_id,
                    workspace_root=str(self._runtime.conversation_root / chat_id),
                    model_configuration=model,
                    product_instructions=CHAT_INSTRUCTIONS,
                    profile_id=CHAT_PROFILE,
                    tool_allowlist=tuple(CHAT_TOOLS),
                    skill_allowlist=(),
                    command_allowlist=("user_message", "cancel"),
                ),
                context,
            )
            return await self._runtime.get_snapshot(chat_id, context)

    async def model_history(
        self, chat_id: str, current: str, context: RequestContext
    ) -> list[JsonObject]:
        if not chat_id.startswith(CHAT_PREFIX) or context.profile_id != CHAT_PROFILE:
            raise PermissionError("EchoMind chat session required")
        store = await self._runtime._authorized_store(chat_id, context)
        snapshot = await asyncio.to_thread(store.load_snapshot)
        turns = completed_chat_turns(snapshot, current)
        manager = await self._runtime._authorized_store(
            chat_id.removeprefix(CHAT_PREFIX),
            replace(context, profile_id=MANAGER_PROFILE),
        )
        history = await asyncio.to_thread(manager.load_snapshot)
        remaining = 32000 - sum(
            len(row["question"]) + len(row["answer"]) for row in turns
        )
        preparation = completed_chat_turns(
            history, current, preparation=True, budget=remaining
        )
        return [dict(row) for row in (*preparation, *turns)]

    async def call(
        self,
        chat_id: str,
        name: str,
        arguments: JsonObject,
        tool_call_id: str,
        context: RequestContext,
        origin_command_id: str | None,
    ) -> JsonObject:
        if not chat_id.startswith(CHAT_PREFIX):
            raise PermissionError("EchoMind Chat Agent session required")
        chat_context = replace(context, profile_id=CHAT_PROFILE)
        chat = await self._runtime._authorized_store(chat_id, chat_context)
        if not chat.permits_tool(name):
            raise PermissionError("EchoMind Chat Agent tool is not enabled")
        manager_id = chat_id.removeprefix(CHAT_PREFIX)
        store, context = await self._manager(manager_id, context)
        if name == ENQUEUE:
            work = EnqueueRequirement.model_validate(arguments)
            if not origin_command_id or not tool_call_id:
                raise ValueError("The user command and tool call IDs are required")
            identity = sha256(
                json.dumps([chat_id, origin_command_id, tool_call_id]).encode()
            ).hexdigest()
            return await self._runtime.enqueue_chat_requirement(
                manager_id,
                EnqueueTaskCommand(
                    command_id=CHAT_REQUIREMENT_COMMAND_PREFIX + identity,
                    instruction=work.instruction,
                ),
                context,
                origin_turn_id=origin_command_id,
            )

        if name == GET_STATUS:
            GetStatus.model_validate(arguments)
            snapshot = await self._runtime._load_saved_snapshot(
                manager_id, store, context, recover_inputs=False
            )
            snapshot = await self._runtime._project_execution_status(snapshot, store)
            versions = await asyncio.to_thread(
                self._runtime._artifact_versions, manager_id, context
            )
            reports = [v for v in versions if v.kind == "report" and v.state == "ready"]
            return {
                "status": snapshot.status,
                "preparation_phase": snapshot.preparation.phase
                if snapshot.preparation
                else None,
                "project_id": snapshot.project.project_id if snapshot.project else None,
                "pending_questions": len(snapshot.pending_inputs),
                "pending_permissions": len(snapshot.pending_permissions),
                "queued_requirements": len(snapshot.experiment_queue or ()),
                "automatic_turns_suspended": bool(snapshot.automatic_turns_suspended),
                "runs": [
                    run.model_dump(
                        mode="json",
                        include={
                            "run_id",
                            "task_name",
                            "stage_id",
                            "state",
                            "report_state",
                            "waiting_reason",
                            "platform_task_id",
                        },
                    )
                    for run in (snapshot.experiment_executions or ())[-20:]
                ],
                "platform_tasks": [
                    task.model_dump(
                        mode="json", include={"task_id", "status", "updated_at"}
                    )
                    for task in snapshot.external_tasks[-20:]
                ],
                "reports": [
                    {"artifact_id": report.id, "name": report.logical_key}
                    for report in reports[-20:]
                ],
            }

        if name != READ_REPORT:
            raise ValueError("Unknown EchoMind Chat Agent tool")
        request = ReadReport.model_validate(arguments)
        text = await self._read_report(manager_id, request.artifact_id, context)
        return {
            "artifact_id": request.artifact_id,
            "text": text[:16000],
            "truncated": len(text) > 16000,
        }

    async def report_attachment(
        self, manager_id: str, report_id: str, file_id: str, context: RequestContext
    ) -> tuple[str, RequestContext]:
        _, context = await self._manager(manager_id, context)
        versions = {
            v.id: v
            for v in await asyncio.to_thread(
                self._runtime._artifact_versions, manager_id, context
            )
        }
        report = versions.get(report_id)
        item = versions.get(file_id)
        if (
            report is None
            or report.kind != "report"
            or report.state != "ready"
            or item is None
            or item.state not in {"ready", "reference_only"}
            or not any(p.version_id == item.id for p in report.manifest.parents)
        ):
            raise ValueError("No registered report attachment")
        path = item.manifest.source.storage_ref
        if not path or PurePosixPath(path).suffix.lower() not in {
            ".json",
            ".md",
            ".html",
            ".htm",
        }:
            raise ValueError("Report attachment has no preview")
        return to_storage_api_path(path), context

    async def read_report(
        self, manager_id: str, artifact_id: str, context: RequestContext
    ) -> str:
        _, context = await self._manager(manager_id, context)
        return await self._read_report(manager_id, artifact_id, context)

    async def _read_report(
        self, manager_id: str, artifact_id: str, context: RequestContext
    ) -> str:
        versions = self._runtime.versions
        reports = await asyncio.to_thread(
            self._runtime._artifact_versions, manager_id, context
        )
        if versions is None or not any(
            v.id == artifact_id and v.kind == "report" and v.state == "ready"
            for v in reports
        ):
            raise ValueError("No ready report with this ID in the current project")
        result = await asyncio.to_thread(
            versions.call,
            manager_id,
            VersionToolRequest(action="read", arguments={"artifact_id": artifact_id}),
            f"read-report:{artifact_id}",
            context,
        )
        return result["text"]
