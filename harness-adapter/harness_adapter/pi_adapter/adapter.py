from __future__ import annotations

import asyncio
import logging
import os
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
from pyromind_runtime.ports.harness import SessionHandle, SessionSpec

from harness_adapter.pi_adapter.business_tools import (
    execute_validation_tool,
    validation_tool_spec,
)
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from harness_adapter.pi_adapter.permissions import TerminalPermissionPolicy
from harness_adapter.pi_adapter.persistence import PiSessionFiles
from harness_adapter.pi_adapter.protocol import PROTOCOL_VERSION
from harness_adapter.pi_adapter.runner import PiRunnerProcess
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
    fork=False,
    workflow_rollback=False,
    native_workspace_tools=frozenset({"read", "write", "edit", "terminal"}),
)
_PRODUCTION_ENVS = {"prod", "production", "online"}
_WORKFLOW_PATH = Path("public_data/workflow_canvas/workflow.py")
_SYSTEM_PROMPT = """You are a coding agent inside one conversation workspace.
Use read, write, edit, and terminal for workspace operations. Keep generated files
in this workspace. Workspace files use workspace-relative paths. Pi advertises
skills in <available_skills>; read the advertised skill location and resolve its
references against the skill directory. Shared Pyromind knowledge is read-only at
logical paths under knowledge/. Use read rather than terminal or repository search
for skill and knowledge resources. For Pyromind workflow requests, read the matching
skill before editing exactly
public_data/workflow_canvas/workflow.py, then call validate_workflow_dsl with
that relative dsl_path. Do not inspect credentials or work around failed
validation authentication."""


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
    queue: asyncio.Queue[HarnessEvent | None] = field(default_factory=asyncio.Queue)
    runner: PiRunnerProcess | None = None
    running: bool = False
    pending_permission: _PendingPermission | None = None
    finished_runs: set[str] = field(default_factory=set)


class PiAdapter:
    def __init__(
        self,
        conversation_root: Path | str,
        *,
        skill_root: Path | None = None,
        knowledge_root: Path | None = None,
    ) -> None:
        if os.getenv("APP_ENV", "dev").strip().lower() in _PRODUCTION_ENVS:
            raise RuntimeError(
                "Pi local execution is disabled in production until sk-sandbox "
                "is integrated"
            )
        repository = Path(__file__).parents[3]
        self._conversation_root = Path(conversation_root).resolve()
        self._skill_root = (
            skill_root or repository / ".agents" / "skills" / "generate-workflow-dsl"
        )
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
        root = Path(spec.workspace_root).resolve()
        expected = (self._conversation_root / spec.conversation_id).resolve()
        if root != expected:
            raise ValueError(
                "Pi SessionSpec.workspace_root must be the conversation directory"
            )
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        files = PiSessionFiles(root)
        config = _session_config(spec)
        files.initialize(config)
        session = _PiSession(spec.conversation_id, root, files, config, context)
        await self._register(session)
        try:
            await self._start_runner(session, _api_key(spec.model_configuration))
            if spec.workflow_xyflow is not None:
                await self._sync_xyflow(session, dict(spec.workflow_xyflow))
            session.queue.put_nowait(_history_synced(session.session_id))
            if spec.initial_message:
                await self._prompt(
                    session, uuid4().hex, _runner_content(spec.initial_message)
                )
            return self._handle(session.session_id)
        except Exception:
            await self._remove(session.session_id)
            raise

    async def attach_session(
        self, conversation_id: str, context: RequestContext
    ) -> SessionHandle:
        existing = self._sessions.get(conversation_id)
        if existing is not None:
            existing.context = context
            return self._handle(conversation_id)
        root = self._safe_conversation_dir(conversation_id)
        files = PiSessionFiles(root)
        config = files.load_session()
        session = _PiSession(conversation_id, root, files, config, context)
        await self._register(session)
        try:
            self._recover_inflight(session)
            await self._start_runner(session, _api_key({}))
            session.queue.put_nowait(_history_synced(conversation_id))
            return self._handle(conversation_id)
        except Exception:
            await self._remove(conversation_id)
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
            pending = session.pending_permission
            if pending is not None and not pending.future.done():
                pending.future.set_result((False, "Cancelled by user"))
            if session.runner is not None and session.runner.running:
                await session.runner.request("cancel", {})
            return {"cancelled": True}
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
            raise ValueError("Pi does not support workflow rollback")
        raise TypeError(f"unsupported command: {type(command).__name__}")

    async def fork(
        self,
        handle: SessionHandle,
        snapshot: ConversationSnapshot,
        context: RequestContext,
    ) -> SessionHandle:
        raise NotImplementedError("Pi does not support fork")

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
        if not root.is_dir():
            raise FileNotFoundError(f"Pi conversation not found: {conversation_id}")
        return root

    async def _start_runner(self, session: _PiSession, api_key: str) -> None:
        async def on_request(method: str, params: dict[str, Any]) -> Any:
            return await self._runner_request(session, method, params)

        async def on_event(frame: dict[str, Any]) -> None:
            await self._runner_event(session, frame)

        async def on_exit(code: int | None) -> None:
            session.runner = None
            inflight = session.files.load_inflight() or {}
            run_id = str(inflight.get("run_id") or f"runner-exit-{uuid4().hex}")
            await self._runner_event(
                session,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "type": "pi.event",
                    "eventId": uuid4().hex,
                    "sessionId": session.session_id,
                    "runId": run_id,
                    "occurredAt": datetime.now().astimezone().isoformat(),
                    "kind": "run.finished",
                    "payload": {
                        "outcome": {
                            "status": "failed",
                            "error_code": "pi_runner_exited",
                            "message": (f"Pi runner exited unexpectedly (code {code})"),
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
                "session_path": str(session.files.session_log_path),
                "skill_root": str(self._skill_root),
                **(
                    {"knowledge_root": str(self._knowledge_root)}
                    if self._knowledge_root is not None
                    else {}
                ),
                "system_prompt": _SYSTEM_PROMPT,
                "model": {**session.config["model"], "api_key": api_key},
                "tools": [validation_tool_spec()],
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
        await self._start_runner(session, _api_key({}))

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
            if params.get("tool_name") != "validate_workflow_dsl":
                raise ValueError(
                    f"unsupported business tool: {params.get('tool_name')}"
                )
            arguments = params.get("arguments")
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            return await execute_validation_tool(
                session.workspace_root, arguments, session.context
            )
        if method == "permission.check":
            arguments = params.get("arguments")
            tool_call_id = params.get("tool_call_id")
            if not isinstance(arguments, dict) or not isinstance(tool_call_id, str):
                raise ValueError("invalid terminal permission request")
            return await self._check_permission(session, tool_call_id, arguments)
        raise ValueError(f"unsupported runner request: {method}")

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
        dsl = await asyncio.to_thread(convert_xyflow_to_dsl, xyflow)
        _atomic_text(session.workspace_root / _WORKFLOW_PATH, dsl)
        await self._emit_workflow(session, uuid4().hex, canvas=xyflow)

    async def _emit_workflow(
        self,
        session: _PiSession,
        event_id: str,
        *,
        source_event_id: str | None = None,
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
