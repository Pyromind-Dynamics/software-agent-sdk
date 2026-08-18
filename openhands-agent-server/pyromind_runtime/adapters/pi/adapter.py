from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from pydantic import JsonValue

from pyromind_runtime.adapters.pi.protocol import (
    PROTOCOL_VERSION,
    PiRunnerError,
    PiRunnerEvent,
    PiRunnerRequest,
    PiRunnerResponse,
)
from pyromind_runtime.contracts.content import JsonObject
from pyromind_runtime.contracts.events import HarnessEvent, HarnessEventType
from pyromind_runtime.contracts.harness import (
    HarnessCapabilities,
    HarnessCommand,
    HarnessDescriptor,
    PermissionResponse,
    SessionHandle,
    SessionSpec,
)


logger = logging.getLogger(__name__)

_SAFE_ENVIRONMENT_NAMES = frozenset(
    {"PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "NODE_EXTRA_CA_CERTS"}
)
_CAPABILITIES = HarnessCapabilities(
    resume=False,
    steer=False,
    cancel=True,
    permission_reply=False,
    partial_message=True,
    custom_tools=True,
    native_workspace_tools=frozenset({"read", "write", "edit", "bash"}),
)
_RUNNER_EVENT_TYPES: dict[str, HarnessEventType] = {
    "agent.started": "run.started",
    "agent.completed": "run.completed",
    "agent.failed": "run.failed",
    "agent.cancelled": "run.completed",
    "message.started": "message.started",
    "message.delta": "message.delta",
    "message.completed": "message.completed",
    "tool.started": "tool.started",
    "tool.progress": "tool.progress",
    "tool.completed": "tool.completed",
    "tool.failed": "tool.failed",
    "usage.updated": "usage.updated",
    "resource.updated": "resource.updated",
}


class PiAdapterError(RuntimeError):
    pass


class PiRunnerExitedError(PiAdapterError):
    pass


class PiRunnerProtocolError(PiAdapterError):
    pass


type PiRunnerCleanup = Callable[[], Awaitable[None]]
type PiRunnerRequestHandler = Callable[[str, JsonObject], Awaitable[JsonValue]]


@dataclass(frozen=True, slots=True)
class PiRunnerLaunch:
    command: tuple[str, ...]
    cwd: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    runtime_config: JsonObject = field(default_factory=dict)
    request_handler: PiRunnerRequestHandler | None = None
    cleanup: PiRunnerCleanup | None = None

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("Pi runner command must not be empty")


class PiRunnerLauncher(Protocol):
    async def prepare(self, spec: SessionSpec) -> PiRunnerLaunch: ...


class StaticPiRunnerLauncher:
    def __init__(
        self,
        command: tuple[str, ...],
        *,
        cwd: str | None = None,
        environment: Mapping[str, str] | None = None,
        runtime_config: JsonObject | None = None,
    ) -> None:
        self._launch = PiRunnerLaunch(
            command=command,
            cwd=cwd,
            environment=dict(environment or {}),
            runtime_config=dict(runtime_config or {}),
        )

    async def prepare(self, spec: SessionSpec) -> PiRunnerLaunch:
        del spec
        return self._launch


def safe_runner_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    candidate = source if source is not None else os.environ
    return {
        name: candidate[name] for name in _SAFE_ENVIRONMENT_NAMES if name in candidate
    }


@dataclass(slots=True)
class _PiSession:
    spec: SessionSpec
    adapter_session_ref: str
    queue: asyncio.Queue[HarnessEvent | None]
    process: _PiRunnerProcess | None = None
    lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closed: bool = False


class _PiRunnerProcess:
    def __init__(
        self,
        *,
        session_id: str,
        process: asyncio.subprocess.Process,
        launch: PiRunnerLaunch,
        queue: asyncio.Queue[HarnessEvent | None],
        request_handler: PiRunnerRequestHandler,
        request_timeout_seconds: float,
        close_timeout_seconds: float,
    ) -> None:
        self.session_id = session_id
        self.process = process
        self.launch = launch
        self.queue = queue
        self.request_handler = request_handler
        self.request_timeout_seconds = request_timeout_seconds
        self.close_timeout_seconds = close_timeout_seconds
        self.pending: dict[str, asyncio.Future[JsonValue]] = {}
        self.callback_tasks: dict[str, asyncio.Task[None]] = {}
        self.write_lock = asyncio.Lock()
        self.closing = False
        self.active_run_id: str | None = None
        self._cleanup_complete = False
        self.reader_task = asyncio.create_task(
            self._read_stdout(),
            name=f"pi-stdout-{session_id}",
        )
        self.stderr_task = asyncio.create_task(
            self._drain_stderr(),
            name=f"pi-stderr-{session_id}",
        )
        self.wait_task = asyncio.create_task(
            self._wait_for_exit(),
            name=f"pi-exit-{session_id}",
        )

    @property
    def is_alive(self) -> bool:
        return not self.closing and self.process.returncode is None

    async def request(self, method: str, params: JsonObject) -> JsonValue:
        if not self.is_alive:
            raise PiRunnerExitedError(f"Pi runner is not active: {self.session_id}")
        request_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self._write(
                PiRunnerRequest(
                    requestId=request_id,
                    method=method,
                    params=params,
                ).model_dump(mode="json")
            )
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.request_timeout_seconds,
            )
        except TimeoutError as exc:
            raise PiAdapterError(f"Pi runner request timed out: {method}") from exc
        finally:
            self.pending.pop(request_id, None)

    async def close(self) -> None:
        if self.closing:
            await self._wait_tasks()
            return
        self.closing = True
        if self.process.returncode is None:
            try:
                await self._request_while_closing("session.close", {})
            except (PiAdapterError, BrokenPipeError):
                logger.debug("Pi runner did not acknowledge session.close")
            try:
                await asyncio.wait_for(
                    self.process.wait(),
                    timeout=self.close_timeout_seconds,
                )
            except TimeoutError:
                self.process.terminate()
                try:
                    await asyncio.wait_for(
                        self.process.wait(),
                        timeout=self.close_timeout_seconds,
                    )
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
        await self._wait_tasks()
        await self._cleanup()

    async def _request_while_closing(
        self,
        method: str,
        params: JsonObject,
    ) -> JsonValue:
        self.closing = False
        try:
            return await self.request(method, params)
        finally:
            self.closing = True

    async def _read_stdout(self) -> None:
        stdout = self.process.stdout
        if stdout is None:
            raise PiAdapterError("Pi runner stdout is unavailable")
        try:
            while line := await stdout.readline():
                await self._handle_line(line)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Invalid Pi runner output for session %s", self.session_id)
            self._fail_pending(PiRunnerProtocolError(str(exc)))
            if self.process.returncode is None:
                self.process.terminate()

    async def _handle_line(self, line: bytes) -> None:
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PiRunnerProtocolError("invalid JSONL from Pi runner") from exc
        if (
            not isinstance(value, dict)
            or value.get("protocolVersion") != PROTOCOL_VERSION
        ):
            raise PiRunnerProtocolError("invalid Pi runner protocol envelope")
        message_type = value.get("type")
        if message_type == "response":
            self._handle_response(PiRunnerResponse.model_validate(value))
            return
        if message_type == "pi.event":
            event = PiRunnerEvent.model_validate(value)
            self._handle_event(event)
            return
        if message_type == "request":
            request = PiRunnerRequest.model_validate(value)
            self._schedule_request(request)
            return
        raise PiRunnerProtocolError("unknown Pi runner message type")

    def _handle_response(self, response: PiRunnerResponse) -> None:
        future = self.pending.get(response.requestId)
        if future is None or future.done():
            return
        if response.error is not None:
            future.set_exception(
                PiAdapterError(
                    f"Pi runner request failed ({response.error.code}): "
                    f"{response.error.message}"
                )
            )
        else:
            future.set_result(response.result)

    def _handle_event(self, event: PiRunnerEvent) -> None:
        translated = _translate_runner_event(event)
        if event.kind == "agent.started":
            self.active_run_id = event.runId
        elif event.kind in {
            "agent.completed",
            "agent.failed",
            "agent.cancelled",
        }:
            self.active_run_id = None
        self.queue.put_nowait(translated)

    def _schedule_request(self, request: PiRunnerRequest) -> None:
        if request.method == "rpc.cancel":
            task = asyncio.create_task(
                self._handle_cancel_request(request),
                name=f"pi-cancel-{self.session_id}-{request.requestId}",
            )
        else:
            task = asyncio.create_task(
                self._handle_request(request),
                name=f"pi-callback-{self.session_id}-{request.requestId}",
            )
            self.callback_tasks[request.requestId] = task
            task.add_done_callback(
                lambda _completed, request_id=request.requestId: (
                    self.callback_tasks.pop(
                        request_id,
                        None,
                    )
                )
            )

    async def _handle_request(self, request: PiRunnerRequest) -> None:
        try:
            result = await self.request_handler(request.method, request.params)
            response = PiRunnerResponse(
                requestId=request.requestId,
                result=result,
            )
        except asyncio.CancelledError:
            response = PiRunnerResponse(
                requestId=request.requestId,
                error=PiRunnerError(
                    code="callback_cancelled",
                    message="Runner callback was cancelled",
                ),
            )
            try:
                await asyncio.shield(
                    self._write(response.model_dump(mode="json", exclude_unset=True))
                )
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        except Exception:
            logger.exception(
                "Pi runner callback failed for session %s and method %s",
                self.session_id,
                request.method,
            )
            response = PiRunnerResponse(
                requestId=request.requestId,
                error=PiRunnerError(
                    code="callback_failed",
                    message="Runner callback failed",
                ),
            )
        await self._write(response.model_dump(mode="json", exclude_unset=True))

    async def _handle_cancel_request(self, request: PiRunnerRequest) -> None:
        target_id = request.params.get("request_id")
        cancelled = False
        if isinstance(target_id, str):
            target = self.callback_tasks.get(target_id)
            if target is not None and not target.done():
                target.cancel()
                cancelled = True
        response = PiRunnerResponse(
            requestId=request.requestId,
            result={"cancelled": cancelled},
        )
        await self._write(response.model_dump(mode="json", exclude_unset=True))

    async def _write(self, message: dict[str, object]) -> None:
        stdin = self.process.stdin
        if stdin is None:
            raise PiRunnerExitedError("Pi runner stdin is unavailable")
        encoded = json.dumps(message, separators=(",", ":")) + "\n"
        async with self.write_lock:
            stdin.write(encoded.encode())
            await stdin.drain()

    async def _drain_stderr(self) -> None:
        stderr = self.process.stderr
        if stderr is None:
            return
        byte_count = 0
        while chunk := await stderr.read(4096):
            byte_count += len(chunk)
        if byte_count:
            logger.debug(
                "Suppressed %d bytes of Pi runner stderr for session %s",
                byte_count,
                self.session_id,
            )

    async def _wait_for_exit(self) -> None:
        return_code = await self.process.wait()
        self._fail_pending(
            PiRunnerExitedError(
                f"Pi runner exited with status {return_code}: {self.session_id}"
            )
        )
        await self._cleanup()
        if not self.closing:
            self.queue.put_nowait(
                HarnessEvent(
                    session_id=self.session_id,
                    type="run.failed",
                    run_id=self.active_run_id,
                    payload={
                        "error_code": "runner_exited",
                        "message": "Pi runner exited unexpectedly",
                    },
                )
            )
            self.active_run_id = None

    def _fail_pending(self, error: Exception) -> None:
        for future in tuple(self.pending.values()):
            if not future.done():
                future.set_exception(error)

    async def _wait_tasks(self) -> None:
        current = asyncio.current_task()
        for callback in tuple(self.callback_tasks.values()):
            callback.cancel()
        tasks = tuple(
            task
            for task in (
                self.reader_task,
                self.stderr_task,
                self.wait_task,
                *self.callback_tasks.values(),
            )
            if task is not current
        )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _cleanup(self) -> None:
        if self._cleanup_complete:
            return
        self._cleanup_complete = True
        if self.launch.cleanup is not None:
            try:
                await self.launch.cleanup()
            except Exception:
                logger.exception("Failed to clean up Pi runner resources")


class PiAdapter:
    def __init__(
        self,
        launcher: PiRunnerLauncher,
        *,
        request_handler: PiRunnerRequestHandler | None = None,
        request_timeout_seconds: float = 10.0,
        close_timeout_seconds: float = 3.0,
    ) -> None:
        self._launcher = launcher
        self._request_handler = request_handler or self._unsupported_request
        self._request_timeout_seconds = request_timeout_seconds
        self._close_timeout_seconds = close_timeout_seconds
        self._sessions: dict[str, _PiSession] = {}
        self._lock = asyncio.Lock()

    async def describe(self) -> HarnessDescriptor:
        return HarnessDescriptor(
            harness_id="pi",
            display_name="Pi",
            capabilities=_CAPABILITIES,
        )

    async def create_session(self, spec: SessionSpec) -> SessionHandle:
        missing = _CAPABILITIES.missing(spec.required_capabilities)
        if missing:
            raise ValueError(
                "Pi does not support required capabilities: "
                + ", ".join(sorted(missing))
            )
        session = _PiSession(
            spec=spec,
            adapter_session_ref=f"pi-{uuid4().hex}",
            queue=asyncio.Queue(),
        )
        async with self._lock:
            if spec.product_session_id in self._sessions:
                raise PiAdapterError(
                    f"Pi session is already active: {spec.product_session_id}"
                )
            self._sessions[spec.product_session_id] = session
        try:
            await self._ensure_process(session)
        except Exception:
            async with self._lock:
                self._sessions.pop(spec.product_session_id, None)
            raise
        return SessionHandle(
            session_id=spec.product_session_id,
            harness_id="pi",
            adapter_session_ref=session.adapter_session_ref,
            capabilities=_CAPABILITIES,
        )

    async def send(self, session_id: str, command: HarnessCommand) -> None:
        if command.delivery != "auto":
            raise PiAdapterError("Pi does not support steering or queued delivery")
        session = self._session(session_id)
        process = await self._ensure_process(session)
        await process.request(
            "run.prompt",
            {
                "command_id": command.command_id,
                "content": [block.model_dump(mode="json") for block in command.content],
            },
        )

    async def cancel(self, session_id: str) -> None:
        session = self._session(session_id)
        process = await self._ensure_process(session)
        await process.request("run.cancel", {})

    async def respond_permission(
        self,
        session_id: str,
        response: PermissionResponse,
    ) -> None:
        raise PiAdapterError(
            "Pi does not support permission response "
            f"{response.permission_id} for session {session_id}"
        )

    def subscribe(self, session_id: str) -> AsyncIterator[HarnessEvent]:
        queue = self._session(session_id).queue

        async def stream() -> AsyncIterator[HarnessEvent]:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event

        return stream()

    async def close(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.closed = True
        async with session.lifecycle_lock:
            if session.process is not None:
                await session.process.close()
                session.process = None
        session.queue.put_nowait(None)

    async def _ensure_process(self, session: _PiSession) -> _PiRunnerProcess:
        async with session.lifecycle_lock:
            if session.closed:
                raise PiAdapterError(
                    f"Pi session is already closed: {session.spec.product_session_id}"
                )
            if session.process is not None and session.process.is_alive:
                return session.process
            if session.process is not None:
                await session.process.close()
            launch = await self._launcher.prepare(session.spec)
            process = await asyncio.create_subprocess_exec(
                *launch.command,
                cwd=launch.cwd,
                env=dict(launch.environment),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            runner = _PiRunnerProcess(
                session_id=session.spec.product_session_id,
                process=process,
                launch=launch,
                queue=session.queue,
                request_handler=launch.request_handler or self._request_handler,
                request_timeout_seconds=self._request_timeout_seconds,
                close_timeout_seconds=self._close_timeout_seconds,
            )
            session.process = runner
            try:
                await runner.request(
                    "session.start",
                    _session_start_params(session.spec, launch.runtime_config),
                )
            except Exception:
                await runner.close()
                session.process = None
                raise
            return runner

    def _session(self, session_id: str) -> _PiSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise PiAdapterError(f"Pi session is not active: {session_id}")
        return session

    @staticmethod
    async def _unsupported_request(method: str, params: JsonObject) -> JsonValue:
        del params
        raise PiAdapterError(f"Pi runner callback is not configured: {method}")


def _session_start_params(
    spec: SessionSpec,
    runtime_config: JsonObject,
) -> JsonObject:
    return {
        "session_id": spec.product_session_id,
        "workspace": spec.workspace.model_dump(mode="json"),
        "sandbox": spec.sandbox.model_dump(mode="json"),
        "model_profile": spec.model_profile.model_dump(mode="json"),
        "tools": [tool.model_dump(mode="json") for tool in spec.tools],
        "runtime_config": runtime_config,
    }


def _translate_runner_event(event: PiRunnerEvent) -> HarnessEvent:
    event_type = _RUNNER_EVENT_TYPES[event.kind]
    payload = dict(event.payload)
    if event.kind == "agent.cancelled":
        payload.setdefault("outcome", "cancelled")
    return HarnessEvent.model_validate(
        {
            "event_id": event.eventId,
            "session_id": event.sessionId,
            "occurred_at": event.occurredAt,
            "type": event_type,
            "run_id": event.runId,
            "payload": payload,
        }
    )
