from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from harness_adapter.pi_adapter.protocol import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    PiProtocolError,
    decode_frame,
    encode_frame,
)


logger = logging.getLogger(__name__)
_STREAM_LIMIT = MAX_FRAME_BYTES + 2
RequestHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]
EventHandler = Callable[[dict[str, Any]], Awaitable[None]]
ExitHandler = Callable[[int | None], Awaitable[None]]


class PiRunnerError(RuntimeError):
    pass


class PiRunnerProcess:
    """One private Node runner connected by bounded stdin/stdout JSONL."""

    def __init__(
        self,
        *,
        request_handler: RequestHandler,
        event_handler: EventHandler,
        exit_handler: ExitHandler,
        entrypoint: Path | None = None,
    ) -> None:
        runtime = Path(__file__).parents[2] / "pi-runtime" / "dist" / "index.js"
        self._entrypoint = entrypoint or runtime
        self._request_handler = request_handler
        self._event_handler = event_handler
        self._exit_handler = exit_handler
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._write_lock = asyncio.Lock()
        self._closing = False

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self, config: dict[str, Any]) -> None:
        if self.running:
            raise PiRunnerError("Pi runner is already running")
        if not self._entrypoint.is_file():
            raise PiRunnerError(
                f"Pi runner is not built: {self._entrypoint}; run npm run build"
            )
        self._closing = False
        self._process = await asyncio.create_subprocess_exec(
            "node",
            str(self._entrypoint),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_runner_environment(),
            limit=_STREAM_LIMIT,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        try:
            await self.request("start", config)
        except Exception:
            await self.terminate()
            raise

    async def request(self, method: str, params: dict[str, Any]) -> Any:
        process = self._process
        if process is None or process.returncode is not None or process.stdin is None:
            raise PiRunnerError("Pi runner is not running")
        request_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write(
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "type": "request",
                    "requestId": request_id,
                    "method": method,
                    "params": params,
                }
            )
            return await future
        finally:
            self._pending.pop(request_id, None)

    async def close(self) -> None:
        if not self.running:
            await self.terminate()
            return
        self._closing = True
        with contextlib.suppress(Exception):
            await self.request("close", {})
        await self.terminate()

    async def terminate(self) -> None:
        self._closing = True
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), 5)
            if process.returncode is None:
                process.kill()
                await process.wait()
        current = asyncio.current_task()
        tasks = tuple(
            task
            for task in (self._reader_task, self._stderr_task)
            if task is not None and task is not current
        )
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._process = None
        self._reader_task = None
        self._stderr_task = None

    async def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        error: Exception | None = None
        try:
            while line := await self._process.stdout.readline():
                frame = decode_frame(line.rstrip(b"\n"))
                await self._handle_frame(frame)
        except Exception as exc:
            error = exc
            logger.exception("Pi runner protocol failed")
            if self._process.returncode is None:
                self._process.terminate()
        finally:
            code = await self._process.wait()
            failure = error or PiRunnerError(f"Pi runner exited with code {code}")
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(failure)
            if not self._closing:
                await self._exit_handler(code)

    async def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while line := await self._process.stderr.readline():
            logger.warning("Pi runner: %s", line.decode(errors="replace").rstrip())

    async def _handle_frame(self, frame: dict[str, Any]) -> None:
        frame_type = frame.get("type")
        if frame_type == "response":
            request_id = frame.get("requestId")
            if not isinstance(request_id, str):
                raise PiProtocolError("response requestId must be a string")
            future = self._pending.get(request_id)
            if future is None or future.done():
                return
            error = frame.get("error")
            if isinstance(error, dict):
                future.set_exception(
                    PiRunnerError(
                        f"{error.get('code', 'runner_error')}: "
                        f"{error.get('message', 'runner request failed')}"
                    )
                )
            else:
                future.set_result(frame.get("result"))
            return
        if frame_type == "pi.event":
            await self._event_handler(frame)
            return
        if frame_type == "request":
            asyncio.create_task(self._answer_request(frame))
            return
        raise PiProtocolError("unexpected runner frame")

    async def _answer_request(self, frame: dict[str, Any]) -> None:
        request_id = frame.get("requestId")
        try:
            method = frame.get("method")
            params = frame.get("params")
            if not isinstance(request_id, str) or not isinstance(method, str):
                raise PiProtocolError("invalid runner request")
            if not isinstance(params, dict):
                raise PiProtocolError("runner request params must be an object")
            result = await self._request_handler(method, params)
            response = {
                "protocolVersion": PROTOCOL_VERSION,
                "type": "response",
                "requestId": request_id,
                "result": result,
            }
        except Exception as exc:
            response = {
                "protocolVersion": PROTOCOL_VERSION,
                "type": "response",
                "requestId": request_id,
                "error": {
                    "code": "request_failed",
                    "message": str(exc),
                },
            }
        await self._write(response)

    async def _write(self, frame: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise PiRunnerError("Pi runner stdin is closed")
        async with self._write_lock:
            process.stdin.write(encode_frame(frame))
            await process.stdin.drain()


def _runner_environment() -> dict[str, str]:
    allowed = (
        "PATH",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
    )
    return {name: os.environ[name] for name in allowed if name in os.environ}
