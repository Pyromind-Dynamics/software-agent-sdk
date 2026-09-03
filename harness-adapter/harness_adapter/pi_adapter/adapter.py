from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.commands import (
    CancelCommand,
    PermissionResponseCommand,
    ProductCommand,
    RollbackWorkflowCommand,
    UserMessageCommand,
)
from pyromind_runtime.domain.content import JsonObject, TextContent
from pyromind_runtime.domain.context import RequestContext
from pyromind_runtime.domain.events import HarnessEvent
from pyromind_runtime.domain.snapshot import ConversationSnapshot
from pyromind_runtime.ports.harness import (
    ExternalTaskNotification,
    ForkSpec,
    ProductCheckpoint,
    RestoreWorkflowResult,
    RestoreWorkflowSpec,
    SessionHandle,
    SessionSpec,
)

from harness_adapter.pi_adapter.business_tool_host import (
    PyromindBusinessToolHost,
    ToolExecutionContext,
)
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from harness_adapter.pi_adapter.permissions import TerminalPermissionPolicy
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.protocol import PROTOCOL_VERSION
from harness_adapter.pi_adapter.runner import PiRunnerExit, PiRunnerProcess
from harness_adapter.pi_adapter.terminal_backend import validate_pi_terminal_backend
from openhands.agent_server.workflow_canvas_models import (
    SaveWorkflowCanvasEventSnapshotRequest,
)
from openhands.agent_server.workflow_canvas_store import FileWorkflowCanvasStore
from openhands.tools.workflow.dsl_to_xyflow import (
    convert_dsl_to_xyflow,
    convert_xyflow_to_dsl,
)


logger = logging.getLogger(__name__)


PI_CAPABILITIES = HarnessCapabilities(
    resume=True,
    cancel=True,
    permission_reply=True,
    partial_message=True,
    fork=True,
    workflow_rollback=True,
    external_task_resume=True,
    native_workspace_tools=frozenset({"read", "write", "edit", "terminal"}),
)
_WORKFLOW_PATH = Path("public_data/workflow_canvas/workflow.py")
_SYSTEM_PROMPT = """You are a coding agent inside one conversation workspace.
Use read, write, edit, and terminal for workspace operations. Keep generated files
under public_data/. Every terminal call starts at the workspace root (`.`). You may
use cd within one command, but never rely on a directory change from an earlier call.
Workspace files use public_data/... paths; authorized absolute paths are also accepted.
Pi advertises skills in <available_skills>; their absolute locations are read-only
resource addresses, not workspace locations. Read the exact advertised skill path
and resolve its references against that skill directory, but never derive a workspace
path from it. Shared Pyromind knowledge is read-only at logical paths under knowledge/.
Use read rather than terminal or repository search for skill and knowledge resources.
For Pyromind workflow requests, read the matching
skill before editing exactly
public_data/workflow_canvas/workflow.py, then call validate_workflow_dsl without
dsl_path; pass it only when validating another workspace file. Do not inspect
credentials or work around failed validation authentication.

Route dataset work before acting. Use the data-processing skill for any
dataset request; its SKILL.md routing table selects the paradigm:
format-conversion for deterministic field, format, structure, regex, keyword,
and length transformations; llm-pipeline for content assessment, DataFlow
operators, LLM processing, images, and multimodal work. If the intent is
ambiguous, ask the user first.
Never start both full-run paths for one request. Read only the matching SKILL.md
and its explicitly referenced files before using the business tools."""


@dataclass(slots=True)
class _PendingPermission:
    permission_id: str
    operation_id: str
    future: asyncio.Future[tuple[bool, str | None]]


@dataclass(slots=True)
class _PiSession:
    session_id: str
    workspace_root: Path
    files: PiSessionFiles
    config: dict[str, Any]
    context: RequestContext
    model_configuration: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    queue: asyncio.Queue[HarnessEvent | None] = field(default_factory=asyncio.Queue)
    runner: PiRunnerProcess | None = None
    running: bool = False
    pending_permission: _PendingPermission | None = None
    finished_runs: set[str] = field(default_factory=set)
    active_external_tasks: dict[str, str] = field(default_factory=dict)
    pending_checkpoint_events: dict[str, list[str]] = field(default_factory=dict)


class PiAdapter:
    def __init__(
        self,
        conversation_root: Path | str,
        *,
        terminal_backend: str,
        skill_root: Path | None = None,
        skill_roots: list[Path] | None = None,
        knowledge_root: Path | None = None,
    ) -> None:
        if getattr(sys, "frozen", False):
            # PyInstaller: __file__ lives under the _MEIPASS extraction dir, so
            # parents[3] would escape the bundle; bundled skill datas sit at
            # _MEIPASS/.agents/skills (see agent-server.spec).
            repository = Path(getattr(sys, "_MEIPASS", ""))
        else:
            repository = Path(__file__).parents[3]
        self._terminal_backend = validate_pi_terminal_backend(terminal_backend)
        self._conversation_root = Path(conversation_root).resolve()
        skills_directory = Path(
            os.getenv("PYROMIND_SKILLS_PATH") or repository / ".agents" / "skills"
        )
        default_skill_roots = {
            name: skills_directory / name
            for name in (
                "generate-workflow-dsl",
                "data-processing",
                "debug-workflow",
                "training-analysis",
            )
        }
        configured_roots = list(skill_roots or ())
        if skill_root is not None:
            configured_roots.insert(0, skill_root)
        configured_names = {Path(path).name for path in configured_roots}
        configured_roots.extend(
            path
            for name, path in default_skill_roots.items()
            if name not in configured_names
        )
        self._skill_roots = [Path(path).resolve() for path in configured_roots]
        missing = [str(path) for path in self._skill_roots if not path.is_dir()]
        if missing:
            raise ValueError(f"Pi skill roots do not exist: {', '.join(missing)}")
        self._business_tools = PyromindBusinessToolHost(self._skill_roots)
        configured_knowledge = knowledge_root or os.getenv(
            "PYROMIND_KNOWLEDGE_BASE_PATH"
        )
        knowledge_candidate = Path(
            configured_knowledge or repository / "knowledge"
        ).resolve()
        self._knowledge_root = (
            knowledge_candidate if knowledge_candidate.is_dir() else None
        )
        self._sessions: dict[str, _PiSession] = {}
        self._permissions = TerminalPermissionPolicy()
        self._lock = asyncio.Lock()

    async def describe(self) -> tuple[str, HarnessCapabilities]:
        return "pi", PI_CAPABILITIES

    async def create_session(
        self, spec: SessionSpec, context: RequestContext
    ) -> SessionHandle:
        root = Path(os.path.abspath(spec.workspace_root))
        expected = self._conversation_root / spec.conversation_id
        if root != expected:
            raise ValueError(
                "Pi SessionSpec.workspace_root must be the conversation directory"
            )
        session: _PiSession | None = None
        try:
            _prepare_workspace(root, create=True)
            _prepare_pi_runtime_directories(root)
            files = PiSessionFiles(root)
            config = _session_config(spec)
            files.initialize(config)
            session = _PiSession(
                spec.conversation_id,
                root,
                files,
                config,
                context,
                model_configuration=dict(spec.model_configuration),
                extra=dict(spec.extra),
            )
            await self._register(session)
            workflow_event_id = None
            if spec.workflow_xyflow is not None:
                workflow_event_id = await self._stage_xyflow(
                    session, dict(spec.workflow_xyflow)
                )
            await self._start_runner(session, _api_key(spec.model_configuration))
            if workflow_event_id is not None:
                await self._append_workflow_context(session, workflow_event_id)
            session.queue.put_nowait(_history_synced(session.session_id))
            if spec.initial_message:
                await self._prompt(
                    session, uuid4().hex, _runner_content(spec.initial_message)
                )
            return self._handle(session.session_id)
        except Exception:
            if session is not None:
                await self._remove(session.session_id)
                if session.runner is not None:
                    try:
                        await session.runner.close()
                    finally:
                        _remove_created_workspace(root, self._conversation_root)
                else:
                    _remove_created_workspace(root, self._conversation_root)
            else:
                _remove_created_workspace(root, self._conversation_root)
            raise

    async def attach_session(
        self, conversation_id: str, context: RequestContext
    ) -> SessionHandle:
        existing = self._sessions.get(conversation_id)
        if existing is not None:
            existing.context = context
            return self._handle(conversation_id)
        root = self._safe_conversation_dir(conversation_id)
        _prepare_workspace(root, create=False)
        _prepare_pi_runtime_directories(root)
        files = PiSessionFiles(root)
        config = files.load_session()
        files.ensure_session_log()
        session = _PiSession(
            conversation_id,
            root,
            files,
            config,
            context,
            model_configuration=_restored_model_configuration(config),
            extra=dict(config.get("extra") or {}),
            active_external_tasks=_restore_active_external_tasks(root),
        )
        await self._register(session)
        try:
            self._recover_inflight(session)
            await self._start_runner(session, _api_key({}))
            session.queue.put_nowait(_history_synced(conversation_id))
            return self._handle(conversation_id)
        except Exception:
            await self._remove(conversation_id)
            if session.runner is not None:
                await session.runner.close()
            raise

    async def send(
        self, handle: SessionHandle, command: ProductCommand, context: RequestContext
    ) -> JsonObject:
        session = self._session(handle.session_id)
        session.context = context
        if isinstance(command, UserMessageCommand):
            if command.workflow_xyflow is not None:
                await self._sync_xyflow(session, dict(command.workflow_xyflow))
            await self._ensure_runner(session)
            return await self._prompt(
                session, command.command_id, _runner_content(command.content)
            )
        if isinstance(command, CancelCommand):
            self._business_tools.cancel(session.session_id)
            pending = session.pending_permission
            if pending is not None and not pending.future.done():
                pending.future.set_result((False, "Cancelled by user"))
            runner_cancel_error = None
            if session.runner is not None and session.runner.running:
                try:
                    await session.runner.request("cancel", {})
                except Exception as exc:
                    runner_cancel_error = type(exc).__name__
            stop_results = []
            for task_id in tuple(session.active_external_tasks):
                stop_result = await self._business_tools.stop_platform_task(
                    task_id, self._tool_context(session)
                )
                stop_results.append(stop_result)
                if not stop_result.get("is_error"):
                    self._emit_external_task_stopped(session, task_id)
                session.active_external_tasks.pop(task_id, None)
            return {
                "cancelled": runner_cancel_error is None,
                "runner_error": runner_cancel_error,
                "external_tasks": stop_results,
            }
        if isinstance(command, PermissionResponseCommand):
            pending = session.pending_permission
            if pending is None or pending.permission_id != command.permission_id:
                raise ValueError(f"permission is not pending: {command.permission_id}")
            if not pending.future.done():
                pending.future.set_result(
                    (command.decision == "allow_once", command.reason)
                )
            return {"resolved": True}
        if isinstance(command, RollbackWorkflowCommand):
            raise TypeError("workflow rollback is orchestrated by ConversationRuntime")
        raise TypeError(f"unsupported command: {type(command).__name__}")

    async def fork(
        self,
        handle: SessionHandle,
        spec: ForkSpec,
        checkpoint: ProductCheckpoint,
        context: RequestContext,
    ) -> SessionHandle:
        source = self._session(handle.session_id)
        if source.running or source.pending_permission is not None:
            raise ValueError("Pi source conversation is busy")
        leaf_id = source.files.load_checkpoint_index().get(checkpoint.event_id)
        if not leaf_id:
            raise ValueError("Pi checkpoint entry is unavailable")
        target_root = (self._conversation_root / spec.target_conversation_id).resolve()
        if target_root.parent != self._conversation_root:
            raise ValueError("unsafe Pi fork target")
        _copy_public_data_for_fork(source.workspace_root, target_root)
        target: _PiSession | None = None
        try:
            target_workflow = target_root / _WORKFLOW_PATH
            checkpoint_dsl = checkpoint.workflow.dsl
            if checkpoint_dsl.strip():
                _atomic_text(target_workflow, checkpoint_dsl)
            else:
                target_workflow.unlink(missing_ok=True)
            _prepare_pi_runtime_directories(target_root)
            target_files = PiSessionFiles(target_root)
            target_config = {
                **source.config,
                "session_id": spec.target_conversation_id,
            }
            target_files.initialize(target_config)
            await self._ensure_runner(source)
            assert source.runner is not None
            branch = await source.runner.request(
                "fork",
                {
                    "leaf_id": leaf_id,
                    "target_session_dir": str(target_files.directory),
                    "target_cwd": str(target_root),
                },
            )
            if not isinstance(branch, dict) or not isinstance(
                branch.get("session_path"), str
            ):
                raise RuntimeError("Pi runner did not return a branched session")
            generated = Path(branch["session_path"]).resolve()
            if generated.parent != target_files.directory.resolve():
                raise ValueError("Pi runner returned an unsafe session path")
            os.replace(generated, target_files.session_log_path)
            target = _PiSession(
                spec.target_conversation_id,
                target_root,
                target_files,
                target_config,
                context,
                model_configuration=dict(source.model_configuration),
                extra=dict(source.extra),
            )
            await self._register(target)
            await self._start_runner(target, _api_key(source.model_configuration))
            target.queue.put_nowait(_history_synced(target.session_id))
            return self._handle(target.session_id)
        except Exception:
            if target is not None:
                await self._remove(target.session_id)
                if target.runner is not None:
                    try:
                        await target.runner.close()
                    finally:
                        _remove_created_workspace(target_root, self._conversation_root)
                else:
                    _remove_created_workspace(target_root, self._conversation_root)
            else:
                _remove_created_workspace(target_root, self._conversation_root)
            raise

    async def restore_workflow(
        self,
        handle: SessionHandle,
        spec: RestoreWorkflowSpec,
        context: RequestContext,
    ) -> RestoreWorkflowResult:
        session = self._session(handle.session_id)
        session.context = context
        _prepare_workspace(session.workspace_root, create=False)
        path = session.workspace_root / _WORKFLOW_PATH
        dsl = spec.checkpoint.workflow.dsl
        if dsl.strip():
            _atomic_text(path, dsl)
            action = "updated"
        else:
            path.unlink(missing_ok=True)
            action = "removed"
        await self._ensure_runner(session)
        assert session.runner is not None
        append_result = await session.runner.request(
            "context.append",
            {
                "content": (
                    "<system_reminder>The workflow was restored to Product "
                    f"checkpoint {spec.checkpoint.event_id}. Treat the workspace "
                    "workflow file as authoritative.</system_reminder>"
                ),
                "details": {
                    "checkpoint_event_id": spec.checkpoint.event_id,
                    "workflow_file_action": action,
                },
                "trigger_turn": spec.trigger_turn,
            },
        )
        if isinstance(append_result, dict) and isinstance(
            append_result.get("checkpoint_entry_id"), str
        ):
            index = session.files.load_checkpoint_index()
            index[f"rollback:{spec.command_id}:workflow"] = append_result[
                "checkpoint_entry_id"
            ]
            session.files.save_checkpoint_index(index)
        return RestoreWorkflowResult(workflow_file_action=action)

    async def notify_external_task(
        self,
        handle: SessionHandle,
        notification: ExternalTaskNotification,
        context: RequestContext,
    ) -> JsonObject:
        session = self._session(handle.session_id)
        session.context = context
        status = _external_task_status(notification.status)
        if status in {"succeeded", "failed", "terminated", "stopped"}:
            session.active_external_tasks.pop(notification.task_id, None)
        if status == "stopped":
            return {"accepted": False, "reason": "user_stopped"}
        await self._ensure_runner(session)
        assert session.runner is not None
        task_id = notification.task_id
        if notification.reset_attempt_budget:
            self._business_tools.reset_attempt_budget(session.workspace_root)
        result = await session.runner.request(
            "notify" if notification.trigger_turn else "context.append",
            {
                "run_id": f"callback:{task_id}:{status}",
                "content": notification.hidden_text,
                "details": notification.model_dump(mode="json"),
                "trigger_turn": notification.trigger_turn,
            },
        )
        return result if isinstance(result, dict) else {"accepted": True}

    def subscribe(self, handle: SessionHandle) -> AsyncIterator[HarnessEvent]:
        queue = self._session(handle.session_id).queue

        async def stream() -> AsyncIterator[HarnessEvent]:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event

        return stream()

    async def close(self, handle: SessionHandle) -> None:
        session = await self._remove(handle.session_id)
        if session is None:
            return
        if (
            session.pending_permission is not None
            and not session.pending_permission.future.done()
        ):
            session.pending_permission.future.set_result((False, "Session closed"))
        if session.runner is not None:
            await session.runner.close()
        session.queue.put_nowait(None)

    async def _register(self, session: _PiSession) -> None:
        async with self._lock:
            if session.session_id in self._sessions:
                raise ValueError(f"Pi session is already active: {session.session_id}")
            self._sessions[session.session_id] = session

    async def _remove(self, session_id: str) -> _PiSession | None:
        async with self._lock:
            return self._sessions.pop(session_id, None)

    def _session(self, session_id: str) -> _PiSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise ValueError(f"Pi session is not active: {session_id}") from exc

    def _safe_conversation_dir(self, conversation_id: str) -> Path:
        if not conversation_id or "/" in conversation_id or "\\" in conversation_id:
            raise ValueError("unsafe conversation id")
        root = self._conversation_root / conversation_id
        if root.is_symlink():
            raise RuntimeError(
                "PI_WORKSPACE_INVALID: conversation root must not be a symbolic link"
            )
        if not root.is_dir():
            raise FileNotFoundError(f"Pi conversation not found: {conversation_id}")
        return root

    async def _start_runner(self, session: _PiSession, api_key: str) -> None:
        runner: PiRunnerProcess

        async def on_request(method: str, params: dict[str, Any]) -> Any:
            return await self._runner_request(session, method, params)

        async def on_event(frame: dict[str, Any]) -> None:
            await self._runner_event(session, frame)

        async def on_exit(exit_result: PiRunnerExit) -> None:
            inflight = session.files.load_inflight()
            stale_callback = session.runner is not runner
            logger.info(
                "pi.runner_exit conversation_id=%s pid=%s returncode=%s "
                "exit_reason=%s has_inflight=%s stale_callback=%s",
                session.session_id,
                runner.pid,
                exit_result.returncode,
                exit_result.reason,
                inflight is not None,
                stale_callback,
            )
            if stale_callback:
                return
            session.runner = None
            if exit_result.reason != "unexpected" or inflight is None:
                return
            run_id_value = inflight.get("run_id")
            if not isinstance(run_id_value, str) or not run_id_value:
                logger.error(
                    "Ignoring unexpected Pi runner exit with invalid inflight state "
                    "conversation_id=%s pid=%s",
                    session.session_id,
                    runner.pid,
                )
                return
            await self._runner_event(
                session,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "type": "pi.event",
                    "eventId": uuid4().hex,
                    "sessionId": session.session_id,
                    "runId": run_id_value,
                    "occurredAt": datetime.now().astimezone().isoformat(),
                    "kind": "run.finished",
                    "payload": {
                        "outcome": {
                            "status": "failed",
                            "error_code": "pi_runner_exited",
                            "message": (
                                "Pi runner exited unexpectedly "
                                f"(code {exit_result.returncode})"
                            ),
                        }
                    },
                },
            )

        runner = PiRunnerProcess(
            request_handler=on_request, event_handler=on_event, exit_handler=on_exit
        )
        session.runner = runner
        started_at = time.perf_counter()
        await runner.start(
            {
                "session_id": session.session_id,
                "workspace_root": str(session.workspace_root),
                "terminal_backend": self._terminal_backend,
                "session_path": str(session.files.session_log_path),
                "skill_roots": [
                    {"name": path.name, "path": str(path)} for path in self._skill_roots
                ],
                **(
                    {"knowledge_root": str(self._knowledge_root)}
                    if self._knowledge_root is not None
                    else {}
                ),
                "system_prompt": _SYSTEM_PROMPT,
                "model": {**session.config["model"], "api_key": api_key},
                "tools": self._business_tools.specs(),
            }
        )
        logger.info(
            "pi.runner_start_ms=%.3f conversation_id=%s",
            (time.perf_counter() - started_at) * 1000,
            session.session_id,
        )

    async def _ensure_runner(self, session: _PiSession) -> None:
        if session.runner is not None and session.runner.running:
            return
        await self._start_runner(session, _api_key(session.model_configuration))

    async def _prompt(
        self, session: _PiSession, run_id: str, content: list[dict[str, Any]]
    ) -> JsonObject:
        assert session.runner is not None
        method = "steer" if session.running else "prompt"
        result = await session.runner.request(
            method, {"run_id": run_id, "content": content}
        )
        return result if isinstance(result, dict) else {"accepted": True}

    async def _runner_request(
        self, session: _PiSession, method: str, params: dict[str, Any]
    ) -> Any:
        if method == "tool.execute":
            tool_name = params.get("tool_name")
            if not isinstance(tool_name, str):
                raise ValueError("tool_name must be a string")
            arguments = params.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            result = await self._business_tools.execute(
                tool_name,
                arguments,
                self._tool_context(session),
                tool_call_id=(
                    params.get("tool_call_id")
                    if isinstance(params.get("tool_call_id"), str)
                    else None
                ),
            )
            self._emit_tool_signals(session, tool_name, result)
            return result
        if method == "permission.check":
            arguments = params.get("arguments")
            tool_call_id = params.get("tool_call_id")
            if not isinstance(arguments, dict) or not isinstance(tool_call_id, str):
                raise ValueError("invalid terminal permission request")
            return await self._check_permission(session, tool_call_id, arguments)
        raise ValueError(f"unsupported runner request: {method}")

    def _emit_tool_signals(
        self, session: _PiSession, tool_name: str, result: dict[str, Any]
    ) -> None:
        signals = result.get("signals")
        details = result.get("details")
        if not isinstance(signals, list) or not isinstance(details, dict):
            return
        if (
            tool_name == "df_stop_task"
            and not result.get("is_error")
            and details.get("stopped") is True
            and isinstance(details.get("task_id"), str)
        ):
            task_id = details["task_id"]
            self._emit_external_task_stopped(session, task_id)
            session.active_external_tasks.pop(task_id, None)
        for signal in signals:
            if not isinstance(signal, dict):
                continue
            if signal.get("type") == "external_task.submitted":
                task = signal.get("task")
                if not isinstance(task, dict):
                    continue
                task_id = task.get("task_id") or details.get("task_id")
                run_id = details.get("run_id") or task_id
                output_dir = details.get("output_dir")
                kind = task.get("kind")
                if not isinstance(task_id, str) or not task_id:
                    continue
                if not isinstance(kind, str) or not kind:
                    continue
                now = datetime.now().astimezone().isoformat()
                session.active_external_tasks[task_id] = kind
                session.queue.put_nowait(
                    HarnessEvent(
                        event_id=f"external-task:{task_id}:submitted",
                        session_id=session.session_id,
                        type="external_task.submitted",
                        payload={
                            "task_id": task_id,
                            "kind": kind,
                            "run_id": run_id if isinstance(run_id, str) else task_id,
                            "status": _external_task_status(
                                task.get("status") or details.get("status")
                            ),
                            "output_dir": (
                                output_dir if isinstance(output_dir, str) else None
                            ),
                            "attempt": details.get("attempt"),
                            "max_attempts": details.get("max_attempts"),
                            "keep_ui_lock": bool(details.get("keep_ui_lock", False)),
                            "submitted_at": now,
                            "updated_at": now,
                            "resume_pending": False,
                        },
                    )
                )
            elif signal.get("type") == "agent.message":
                content = signal.get("content")
                if isinstance(content, str) and content:
                    session.queue.put_nowait(
                        HarnessEvent(
                            session_id=session.session_id,
                            type="notice.raised",
                            payload={
                                "severity": "info",
                                "code": "business_tool_message",
                                "message": content,
                            },
                        )
                    )

    @staticmethod
    def _emit_external_task_stopped(session: _PiSession, task_id: str) -> None:
        session.queue.put_nowait(
            HarnessEvent(
                event_id=f"external-task:{task_id}:stopped",
                session_id=session.session_id,
                type="external_task.completed",
                payload={
                    "task_id": task_id,
                    "status": "stopped",
                    "updated_at": datetime.now().astimezone().isoformat(),
                    "resume_pending": False,
                },
            )
        )

    @staticmethod
    def _tool_context(session: _PiSession) -> ToolExecutionContext:
        return ToolExecutionContext(
            conversation_id=session.session_id,
            workspace_root=session.workspace_root,
            request_context=session.context,
            model_configuration=session.model_configuration,
            extra=session.extra,
        )

    async def _check_permission(
        self, session: _PiSession, tool_call_id: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if not self._permissions.requires_confirmation(tool_call_id, arguments):
            return {"allow": True}
        permission_id = uuid4().hex
        future: asyncio.Future[tuple[bool, str | None]] = (
            asyncio.get_running_loop().create_future()
        )
        session.pending_permission = _PendingPermission(
            permission_id, tool_call_id, future
        )
        inflight = session.files.load_inflight() or {}
        inflight.update({"permission_id": permission_id, "operation_id": tool_call_id})
        session.files.save_inflight(inflight)
        session.queue.put_nowait(
            HarnessEvent(
                session_id=session.session_id,
                type="permission.requested",
                run_id=str(inflight.get("run_id") or ""),
                payload={
                    "permission_id": permission_id,
                    "operation_ids": [tool_call_id],
                    "description": (
                        f"Allow risky terminal command: {arguments.get('command')}"
                    ),
                    "choices": ["allow_once", "deny"],
                },
            )
        )
        allow, reason = await future
        session.pending_permission = None
        inflight.pop("permission_id", None)
        session.files.save_inflight(inflight)
        session.queue.put_nowait(
            HarnessEvent(
                session_id=session.session_id,
                type="permission.resolved",
                run_id=str(inflight.get("run_id") or ""),
                payload={
                    "permission_id": permission_id,
                    "decision": "allow_once" if allow else "deny",
                },
            )
        )
        return {"allow": allow, "reason": reason or "User denied terminal command"}

    async def _runner_event(self, session: _PiSession, frame: dict[str, Any]) -> None:
        kind = frame.get("kind")
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        run_id = str(frame.get("runId") or "")
        if kind == "agent.started":
            session.running = True
            session.files.save_inflight({"run_id": run_id})
        elif kind == "run.finished":
            if run_id in session.finished_runs:
                return
            session.finished_runs.add(run_id)
            session.running = False
            session.files.clear_inflight()
            checkpoint_entry_id = payload.get("checkpoint_entry_id")
            pending_events = getattr(session, "pending_checkpoint_events", {}).pop(
                run_id, []
            )
            if isinstance(checkpoint_entry_id, str) and checkpoint_entry_id:
                checkpoint_index = session.files.load_checkpoint_index()
                checkpoint_index.update(
                    {event_id: checkpoint_entry_id for event_id in pending_events}
                )
                session.files.save_checkpoint_index(checkpoint_index)
            logger.info(
                "pi.run_finished conversation_id=%s run_id=%s outcome=%s",
                session.session_id,
                run_id,
                payload.get("outcome"),
            )
        else:
            self._record_inflight(session, kind, payload, run_id)
        for event in translate_runner_event(frame):
            session.queue.put_nowait(event)
        if kind == "tool.completed" and _is_workflow_mutation(
            session.workspace_root, payload
        ):
            source_event_id = str(frame.get("eventId") or uuid4().hex)
            await self._emit_workflow(
                session,
                f"{source_event_id}:workflow",
                source_event_id=source_event_id,
                run_id=run_id,
            )

    def _record_inflight(
        self, session: _PiSession, kind: Any, payload: dict[str, Any], run_id: str
    ) -> None:
        inflight = session.files.load_inflight() or {"run_id": run_id}
        if kind == "message.started" and payload.get("role") == "assistant":
            inflight["message_id"] = payload.get("message_id")
        elif kind == "message.completed":
            inflight.pop("message_id", None)
        elif kind == "tool.started":
            inflight["operation_id"] = payload.get("tool_call_id")
        elif kind in {"tool.completed", "tool.failed"}:
            inflight.pop("operation_id", None)
        session.files.save_inflight(inflight)

    def _recover_inflight(self, session: _PiSession) -> None:
        inflight = session.files.load_inflight()
        if inflight is None:
            return
        run_id = str(inflight.get("run_id") or "")
        message_id = inflight.get("message_id")
        if isinstance(message_id, str):
            session.queue.put_nowait(
                HarnessEvent(
                    session_id=session.session_id,
                    type="message.completed",
                    run_id=run_id,
                    payload={"message_id": message_id, "role": "assistant"},
                )
            )
        operation_id = inflight.get("operation_id")
        if isinstance(operation_id, str):
            session.queue.put_nowait(
                HarnessEvent(
                    session_id=session.session_id,
                    type="operation.failed",
                    run_id=run_id,
                    payload={
                        "operation_id": operation_id,
                        "name": "operation",
                        "details": None,
                        "error_code": "runner_restarted",
                    },
                )
            )
        permission_id = inflight.get("permission_id")
        if isinstance(permission_id, str):
            session.queue.put_nowait(
                HarnessEvent(
                    session_id=session.session_id,
                    type="permission.resolved",
                    run_id=run_id,
                    payload={"permission_id": permission_id, "decision": "deny"},
                )
            )
        session.queue.put_nowait(
            HarnessEvent(
                session_id=session.session_id,
                type="status.changed",
                run_id=run_id,
                payload={"status": "paused"},
            )
        )
        session.files.clear_inflight()

    async def _sync_xyflow(self, session: _PiSession, xyflow: dict[str, Any]) -> None:
        event_id = await self._stage_xyflow(session, xyflow)
        await self._ensure_runner(session)
        await self._append_workflow_context(session, event_id)

    async def _stage_xyflow(self, session: _PiSession, xyflow: dict[str, Any]) -> str:
        _prepare_workspace(session.workspace_root, create=False)
        dsl = await asyncio.to_thread(convert_xyflow_to_dsl, xyflow)
        _atomic_text(session.workspace_root / _WORKFLOW_PATH, dsl)
        event_id = uuid4().hex
        await self._emit_workflow(session, event_id, canvas=xyflow)
        return event_id

    async def _append_workflow_context(
        self, session: _PiSession, event_id: str
    ) -> None:
        assert session.runner is not None
        checkpoint = await session.runner.request(
            "context.append",
            {
                "content": (
                    "<system_reminder>The workflow canvas supplied by the Product "
                    "request was synchronized to the workspace.</system_reminder>"
                ),
                "details": {"workflow_event_id": event_id},
                "trigger_turn": False,
            },
        )
        if isinstance(checkpoint, dict) and isinstance(
            checkpoint.get("checkpoint_entry_id"), str
        ):
            index = session.files.load_checkpoint_index()
            index[event_id] = checkpoint["checkpoint_entry_id"]
            session.files.save_checkpoint_index(index)

    async def _emit_workflow(
        self,
        session: _PiSession,
        event_id: str,
        *,
        source_event_id: str | None = None,
        run_id: str | None = None,
        canvas: dict[str, Any] | None = None,
    ) -> None:
        path = session.workspace_root / _WORKFLOW_PATH
        dsl = path.read_text(encoding="utf-8")
        if canvas is None:
            try:
                canvas = await asyncio.to_thread(convert_dsl_to_xyflow, dsl)
            except Exception:
                canvas = None
        request = SaveWorkflowCanvasEventSnapshotRequest(
            sessionId=session.session_id,
            eventId=event_id,
            snapshotRole="out",
            workflowDslData=dsl,
            workflowXyflowData=canvas,
            eventType="pi.workflow.updated",
        )
        snapshot = await asyncio.to_thread(
            FileWorkflowCanvasStore(
                session.workspace_root, session.session_id
            ).save_event_snapshot,
            request,
        )
        session.queue.put_nowait(
            HarnessEvent(
                event_id=event_id,
                source_event_id=source_event_id,
                session_id=session.session_id,
                type="workflow.updated",
                payload={
                    "resource_id": "pyromind_workflow",
                    "version": snapshot.version_id,
                    "dsl": dsl,
                    "canvas": canvas,
                },
            )
        )
        if run_id:
            session.pending_checkpoint_events.setdefault(run_id, []).append(event_id)

    @staticmethod
    def _handle(session_id: str) -> SessionHandle:
        return SessionHandle(
            session_id=session_id,
            adapter_session_ref=session_id,
            harness_id="pi",
            capabilities=PI_CAPABILITIES,
        )


def _session_config(spec: SessionSpec) -> dict[str, Any]:
    model = str(
        spec.model_configuration.get("model") or os.getenv("LLM_MODEL") or "gpt-4o"
    )
    base_url = spec.model_configuration.get("base_url") or os.getenv("LLM_BASE_URL")
    provider, model_id = _resolve_model(
        model, base_url if isinstance(base_url, str) else None
    )
    api = _resolve_api(
        spec.model_configuration.get("api"),
        base_url if isinstance(base_url, str) else None,
    )
    context_window = _resolve_context_window(
        spec.model_configuration.get("context_window"),
        model_id,
        base_url if isinstance(base_url, str) else None,
    )
    return {
        "session_id": spec.conversation_id,
        "protocol_version": PROTOCOL_VERSION,
        "model": {
            "provider": provider,
            "id": model_id,
            **(
                {"base_url": base_url} if isinstance(base_url, str) and base_url else {}
            ),
            **({"api": api} if api is not None else {}),
            **({"context_window": context_window} if context_window else {}),
        },
        "extra": _safe_session_extra(spec.extra),
    }


def _safe_session_extra(extra: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "env",
        "storage_base_url",
        "storage_api_base_url",
        "dataset_cleaning_output_root",
        "dataset_extraction_output_root",
        "preview_dataset_timeout_seconds",
        "training_analysis_api_base",
        "training_analysis_timeout_seconds",
    }
    return {
        key: value
        for key, value in extra.items()
        if key in allowed and isinstance(value, (str, int, float, bool))
    }


def _restored_model_configuration(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    if not isinstance(model, dict):
        return {}
    return {
        "model": model.get("id"),
        "base_url": model.get("base_url"),
        "api": model.get("api"),
        "context_window": model.get("context_window"),
        "api_key": _api_key({}),
    }


def _resolve_model(model: str, base_url: str | None) -> tuple[str, str]:
    if base_url and "openrouter.ai" in base_url.lower():
        return "openrouter", model.removeprefix("openrouter/")
    if base_url:
        return "openai", model.removeprefix("openai/")
    if "/" in model:
        provider, model_id = model.split("/", 1)
        return provider, model_id
    return "openai", model


def _resolve_api(value: Any, base_url: str | None) -> str | None:
    if value is not None:
        if value not in {"openai-completions", "openai-responses"}:
            raise ValueError(f"unsupported Pi model API: {value}")
        return str(value)
    if not base_url:
        return None
    hostname = (urlparse(base_url).hostname or "").lower()
    if hostname == "api.openai.com":
        return None
    return "openai-completions"


def _resolve_context_window(
    value: Any, model_id: str, base_url: str | None
) -> int | None:
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("context_window must be a positive integer")
        return value
    if "deepseek-v4" in model_id.lower():
        return 200_000
    if base_url and (urlparse(base_url).hostname or "").lower() != "api.openai.com":
        return 128_000
    return None


def _is_workflow_mutation(workspace_root: Path, payload: dict[str, Any]) -> bool:
    if payload.get("tool_name") not in {"write", "edit"}:
        return False
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return False
    value = arguments.get("path")
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    target = path.resolve() if path.is_absolute() else (workspace_root / path).resolve()
    return target == (workspace_root / _WORKFLOW_PATH).resolve()


def _api_key(configuration: dict[str, Any]) -> str:
    value = (
        configuration.get("api_key")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not isinstance(value, str) or not value:
        raise ValueError("Pi requires LLM_API_KEY or OPENAI_API_KEY")
    return value


def _runner_content(content: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, TextContent):
            raise ValueError("Pi version two currently accepts text content only")
        blocks.append({"type": "text", "text": block.text})
    return blocks


def _history_synced(session_id: str) -> HarnessEvent:
    return HarnessEvent(
        event_id=f"{session_id}:history-synced",
        session_id=session_id,
        type="history.synced",
    )


def _external_task_status(value: Any) -> str:
    normalized = str(value or "pending").strip().lower()
    return {
        "success": "succeeded",
        "succeeded": "succeeded",
        "error": "failed",
        "failed": "failed",
        "terminated": "terminated",
        "stopped": "stopped",
        "running": "running",
        "pending": "pending",
    }.get(normalized, "pending")


def _restore_active_external_tasks(root: Path) -> dict[str, str]:
    try:
        snapshot = ConversationSnapshot.model_validate_json(
            (root / "product" / "snapshot.json").read_bytes()
        )
    except (OSError, ValueError):
        return {}
    return {
        task.task_id: task.kind
        for task in snapshot.external_tasks
        if task.status in {"pending", "running"}
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _prepare_workspace(root: Path, *, create: bool) -> Path:
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(
            f"PI_WORKSPACE_INVALID: conversation root must be a real directory: {root}"
        )
    canonical_root = root.resolve(strict=True)
    if canonical_root != root:
        raise RuntimeError(
            f"PI_WORKSPACE_INVALID: conversation root resolves outside itself: {root}"
        )
    root.chmod(0o700)

    public_data = root / "public_data"
    if public_data.is_symlink():
        raise RuntimeError(
            "PI_WORKSPACE_INVALID: public_data must not be a symbolic link"
        )
    try:
        public_data.mkdir(mode=0o700, exist_ok=True)
    except FileExistsError as exc:
        raise RuntimeError(
            "PI_WORKSPACE_INVALID: public_data must be a directory"
        ) from exc
    if not public_data.is_dir() or public_data.resolve(strict=True).parent != root:
        raise RuntimeError(
            "PI_WORKSPACE_INVALID: public_data must stay inside the conversation root"
        )
    public_data.chmod(0o700)
    return public_data


def _prepare_pi_runtime_directories(root: Path) -> None:
    pi_directory = root / "pi"
    if pi_directory.is_symlink():
        raise RuntimeError("PI_WORKSPACE_INVALID: pi must not be a symbolic link")
    try:
        pi_directory.mkdir(mode=0o700, exist_ok=True)
    except FileExistsError as exc:
        raise RuntimeError("PI_WORKSPACE_INVALID: pi must be a directory") from exc
    if not pi_directory.is_dir() or pi_directory.resolve(strict=True).parent != root:
        raise RuntimeError(
            "PI_WORKSPACE_INVALID: pi must stay inside the conversation root"
        )
    pi_directory.chmod(0o700)

    terminal_output = pi_directory / "terminal-output"
    if terminal_output.is_symlink():
        raise RuntimeError(
            "PI_WORKSPACE_INVALID: terminal-output must not be a symbolic link"
        )
    try:
        terminal_output.mkdir(mode=0o700, exist_ok=True)
    except FileExistsError as exc:
        raise RuntimeError(
            "PI_WORKSPACE_INVALID: terminal-output must be a directory"
        ) from exc
    if (
        not terminal_output.is_dir()
        or terminal_output.resolve(strict=True).parent != pi_directory
    ):
        raise RuntimeError("PI_WORKSPACE_INVALID: terminal-output must stay inside pi")
    terminal_output.chmod(0o700)


def _copy_public_data_for_fork(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Pi fork target already exists: {target.name}")
    source_public = _prepare_workspace(source, create=False)
    try:
        _prepare_workspace(target, create=True)
        shutil.copytree(
            source_public,
            target / "public_data",
            dirs_exist_ok=True,
            symlinks=False,
            ignore=lambda directory, names: [
                name for name in names if (Path(directory) / name).is_symlink()
            ],
        )
    except Exception:
        if target.exists() and not target.is_symlink():
            shutil.rmtree(target)
        raise


def _remove_created_workspace(root: Path, conversation_root: Path) -> None:
    if root.parent != conversation_root or root.is_symlink() or not root.exists():
        return
    shutil.rmtree(root)
