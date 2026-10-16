from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import shlex
import threading
import time
from collections.abc import Callable, Coroutine, Sequence
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from agentgenome import GenomeService, RunCancelled
from agentgenome.dispatch import invoke as invoke_genome
from core.ports import PortError, Ports, ShellResult, ToolResult
from pyromind_runtime.domain.content import JsonObject, JsonValue
from pyromind_runtime.ports.workflows import WorkflowEventSink

from harness_adapter.pi_adapter.runner import PiRunnerError


type Request = Callable[[str, JsonObject], Coroutine[Any, Any, Any]]
type Subscribe = Callable[[str, WorkflowEventSink | None], None]


def _is_missing_file(error: Exception) -> bool:
    """Whether the execution plane answered "that file is not there".

    The sandbox file API reports a missing file as a 400 request error, and only the
    portal's own wording says which failure it was.
    """
    message = str(error)
    return "FILE_READ_ERROR" in message and "No such file or directory" in message


class BridgeTool:
    """Run a graph's fixed tool call through the same bridge the agent's tools use.

    The call travels Python → Pi runtime → adapter, so the tool executes with the
    conversation's own context, permissions and signals instead of a second
    implementation of the same business tool.
    """

    def __init__(self, host: ExecutionHost) -> None:
        self.host = host

    def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        response = self.host._call(
            "execution.tool",
            {
                "name": name,
                "arguments": arguments,
                "tool_call_id": f"genome-{uuid4().hex}",
            },
        )
        if not isinstance(response, dict) or not isinstance(
            response.get("is_error"), bool
        ):
            raise PortError(
                "tool bridge returned an invalid response", kind="bad_output"
            )
        details = response.get("details")
        text = "\n".join(
            block["text"]
            for block in response.get("content") or []
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ).strip()
        return ToolResult(
            response["is_error"], details if isinstance(details, dict) else {}, text
        )


class ExecutionHost:
    def __init__(
        self, request: Request, subscribe: Subscribe, local_root: Path | None = None
    ) -> None:
        self.request = request
        self.subscribe = subscribe
        self.local_root = local_root
        self.loop = asyncio.get_running_loop()
        self.cancel = threading.Event()
        self.call_timeout = float(
            os.getenv("PYROMIND_AGENTGENOME_CALL_TIMEOUT_SECONDS", "180")
        )
        self.emit: Callable[[dict[str, Any]], None] = lambda _event: None
        self.root = ""
        self.verification = None
        self.model_judge = False
        self.verification_files: dict[str, str] = {}
        self._output_lock = asyncio.Lock()
        self._reads: dict[str, bytes] = {}
        self._missing: set[str] = set()

    def _call(self, method: str, payload: JsonObject) -> Any:
        """One bounded bridge round trip from the genome worker thread.

        The execution plane can answer a file request by retrying transports for
        minutes, so an unbounded wait turns one wedged read into a distill that
        never returns. Optional probes treat the timeout as "not there" and a
        required read reports it.
        """
        future = asyncio.run_coroutine_threadsafe(
            self.request(method, payload), self.loop
        )
        try:
            return future.result(self.call_timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(
                f"{method} did not answer within {self.call_timeout:.0f}s; "
                "the execution plane is unreachable or still preparing"
            ) from None

    def prepare(
        self,
        run_id: str,
        package: Path,
        params: dict[str, Any],
        path_parameters: list[str],
        cancel: threading.Event,
        emit: Callable[[dict[str, Any]], None],
        *,
        verification_package: Path | None = None,
    ) -> tuple[Ports, dict[str, Any]]:
        self.cancel, self.emit = cancel, emit
        self.root = self._resolve(f"public_data/workflow-runs/{run_id}", True)
        cwd = f"{self.root}/package"
        capabilities = self._call("execution.capabilities", {})
        manifest = json.loads((package / "manifest.json").read_text())
        protected = (
            set(manifest.get("revision", {}).get("verification_resources", []))
            if capabilities.get("protected_verification")
            else set()
        )
        for file in sorted(package.rglob("*")):
            if file.is_symlink():
                raise ValueError("package symlinks are not supported")
            if (
                not file.is_file()
                or "__pycache__" in file.parts
                or file.suffix == ".pyc"
                or file.name == ".DS_Store"
            ):
                continue
            if file.relative_to(package).as_posix() in protected:
                continue
            target = f"{cwd}/{file.relative_to(package).as_posix()}"
            self._write(target, file.read_bytes())
        self._python(
            f"from pathlib import Path; Path({self.root + '/artifacts'!r}).mkdir()"
        )
        resolved = {
            key: self._resolve(value, False) if key in path_parameters else value
            for key, value in params.items()
        }
        verification = None
        if capabilities.get("protected_verification"):
            verification = Path(self._protect(run_id, verification_package or package))
        self.verification = str(verification) if verification else None
        self.model_judge = capabilities.get("model_judge", False)
        return Ports(
            shell=self,
            artifacts=self,
            cwd=Path(cwd),
            verification_cwd=verification,
            tool=BridgeTool(self),
        ), resolved

    def _protect(self, run_id: str, package: Path) -> str:
        protected = ""
        for file in sorted(package.rglob("*")):
            if file.is_symlink():
                raise ValueError("package symlinks are not supported")
            if not file.is_file() or "__pycache__" in file.parts:
                continue
            content = file.read_bytes()
            if len(content) > 256 * 1024:
                raise ValueError("protected package file exceeds 256 KiB")
            protected = self._call(
                "execution.protect",
                {
                    "run_id": run_id,
                    "files": [
                        {
                            "path": file.relative_to(package).as_posix(),
                            "content": base64.b64encode(content).decode("ascii"),
                        }
                    ],
                },
            )["path"]
            self.verification_files[
                f"{protected}/{file.relative_to(package).as_posix()}"
            ] = hashlib.sha256(content).hexdigest()
        return protected

    def prepare_revision(
        self, revision_id: str, package: Path, allowed: list[str]
    ) -> dict[str, Any]:
        capabilities = self._call("execution.capabilities", {})
        if not capabilities.get("revision"):
            raise ValueError("host cannot prepare and collect revision files")
        directory = self._resolve(f"public_data/workflow-revisions/{revision_id}", True)
        self.root = directory
        protected = (
            self._protect(revision_id, package)
            if capabilities.get("protected_verification")
            else None
        )
        # Without OS protection, the full copy provides context for the Agent.
        # Submission still collects only allowlisted business scripts.
        names = (
            allowed
            if protected
            else [
                file.relative_to(package).as_posix()
                for file in sorted(package.rglob("*"))
                if file.is_file()
            ]
        )
        for name in names:
            file = package / name
            if not file.is_file() or file.is_symlink():
                raise ValueError("editable script is not a regular package file")
            self._write(f"{directory}/{name}", file.read_bytes())
        return {
            "editable_dir": directory,
            "verification_dir": protected or directory,
            "verification_protection": "read_only" if protected else "prompt",
        }

    def collect_revision(
        self, draft: dict[str, Any], allowed: list[str]
    ) -> dict[str, bytes]:
        return {
            name: base64.b64decode(
                self._call(
                    "execution.read",
                    {
                        "path": f"{draft['editable_dir']}/{name}",
                        "limit": 262144,
                    },
                )["content"],
                validate=True,
            )
            for name in allowed
        }

    def _resolve(self, path: str, write: bool) -> str:
        return self._call("execution.resolve", {"path": path, "write": write})["path"]

    def path(self, name: str) -> str:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact path escapes run directory")
        return str(PurePosixPath(self.root) / path)

    def describe(self, name: str) -> dict[str, Any]:
        result = self._python(
            "import json,hashlib\nfrom pathlib import Path\n"
            f"p=Path({self.path(name)!r})\n"
            "if not p.is_file():\n print(json.dumps({'missing': True}))\n"
            "else:\n h=hashlib.sha256()\n with p.open('rb') as f:\n"
            "  for block in iter(lambda: f.read(65536), b''): h.update(block)\n"
            f" print(json.dumps({{'path': {name!r}, 'bytes': p.stat().st_size, "
            "'sha256': h.hexdigest()}))"
        )
        value = json.loads(result)
        if value.get("missing"):
            raise FileNotFoundError(name)
        value["execution_path"] = self.path(name)
        return value

    def read_text(self, name: str) -> str:
        result = self._call("execution.read", {"path": self.path(name), "limit": 65536})
        return base64.b64decode(result["content"], validate=True).decode("utf-8")

    def write_text(self, name: str, text: str) -> None:
        self._write(self.path(name), text.encode("utf-8"))

    def read(self, path: str) -> bytes:
        """Read a workspace file, answering host-local exports from disk.

        Pi exports traces with its own file APIs next to the conversation, while the
        execution plane can be a remote sandbox. Those exports are already on this
        host, so read them here; mirroring them into the execution plane first only
        paid a second transport for bytes the reader already holds.
        """
        export = self._local_export(path)
        if export is not None:
            return export.read_bytes()
        if path in self._reads:
            return self._reads[path]
        try:
            result = self._call(
                "execution.read", {"path": self._resolve(path, False), "limit": 262144}
            )
        except PiRunnerError as error:
            # Callers probe for optional files; keep the exception faithful to the
            # filesystem so a probe does not have to know how the portal words it.
            if _is_missing_file(error):
                raise FileNotFoundError(path) from error
            raise
        content = base64.b64decode(result["content"], validate=True)
        self._reads[path] = content
        return content

    def read_many(self, paths: Sequence[str]) -> dict[str, bytes | None]:
        """Answer many workspace reads with as few bridge round trips as possible.

        Freezing a chain reads every script it runs, and a chain is built twice
        (draft and capture), so a read per file turns one slow sandbox into dozens
        of them. Answer from what this host already read, ask for the rest in
        batches, and report a file the plane could not read as ``None``.
        """
        wanted = list(dict.fromkeys(str(path) for path in paths))
        answers: dict[str, bytes | None] = {}
        pending: list[str] = []
        for path in wanted:
            export = self._local_export(path)
            if export is not None:
                answers[path] = export.read_bytes()
            elif path in self._reads:
                answers[path] = self._reads[path]
            elif path in self._missing:
                answers[path] = None
            else:
                pending.append(path)
        while pending:
            pending_paths: list[JsonValue] = [str(path) for path in pending]
            result = self._call(
                "execution.read_many", {"paths": pending_paths, "limit": 262144}
            )
            answered = 0
            for item in result.get("files") or []:
                path = item.get("path")
                if not isinstance(path, str):
                    continue
                answered += 1
                content = item.get("content")
                if isinstance(content, str):
                    decoded = base64.b64decode(content, validate=True)
                    self._reads[path] = decoded
                    answers[path] = decoded
                else:
                    self._missing.add(path)
                    answers[path] = None
            remaining = [str(path) for path in (result.get("remaining") or [])]
            if not remaining or answered == 0:
                break
            pending = remaining
        return answers

    def head(self, path: str, limit: int) -> bytes:
        """Read at most ``limit`` bytes from the start of a workspace file.

        The file read API rejects anything larger than the requested window, so a
        bounded head of a real dataset has to be cut in the execution plane; that
        keeps capture-time sampling from pulling gigabytes across the bridge.
        """
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        export = self._local_export(path)
        if export is not None:
            with export.open("rb") as stream:
                return stream.read(limit)
        resolved = self._resolve(path, False)
        result = self._run(
            f"head -c {max(limit, 0)} {shlex.quote(resolved)} | base64 | tr -d '\\n'",
            None,
            stream=False,
        )
        if result.get("truncated"):
            raise RuntimeError("execution head output was truncated")
        if result.get("rc") != 0:
            raise FileNotFoundError(path)
        content = result.get("stdout") or ""
        return base64.b64decode(content, validate=True)

    def _local_export(self, path: str) -> Path | None:
        if self.local_root is None:
            return None
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        candidate = self.local_root / relative
        return candidate if candidate.is_file() else None

    def _write(self, path: str, content: bytes) -> None:
        if self.cancel.is_set():
            raise RunCancelled()
        # Keep each JSONL request below the bridge's 1 MiB frame limit.
        chunk_size = 256 * 1024
        if len(content) > chunk_size:
            transfer = f"{self.root}/.host/transfers/{uuid4().hex}"
            count = (len(content) + chunk_size - 1) // chunk_size
            for index in range(count):
                self._write(
                    f"{transfer}/{index}",
                    content[index * chunk_size : (index + 1) * chunk_size],
                )
            self._python(
                "from pathlib import Path\n"
                f"target=Path({path!r})\n"
                "target.parent.mkdir(parents=True, exist_ok=True)\n"
                f"parts=Path({transfer!r})\n"
                "with target.open('wb') as out:\n"
                f" for index in range({count}):\n"
                "  part=parts/str(index)\n"
                "  out.write(part.read_bytes())\n"
                "  part.unlink()\n"
                "parts.rmdir()\n"
            )
            return
        self._call(
            "execution.write",
            {"path": path, "content": base64.b64encode(content).decode("ascii")},
        )

    def _python(self, source: str) -> str:
        script = f"{self.root}/.host/{uuid4().hex}.py"
        self._write(script, source.encode("utf-8"))
        command = f"python3 {shlex.quote(script)}"
        result = self._run(command, None, stream=False)
        if result.get("rc") != 0 or result.get("truncated"):
            raise RuntimeError(
                f"Execution environment operation failed: {result.get('stdout')}"
            )
        return result["stdout"]

    def execute(self, command: str, cwd: Path) -> ShellResult:
        started = time.monotonic()
        if str(cwd) == self.verification:
            for path, digest in self.verification_files.items():
                content = base64.b64decode(
                    self._call("execution.read", {"path": path, "limit": 262144})[
                        "content"
                    ],
                    validate=True,
                )
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError("frozen verification resource changed")
        result = self._run(command, str(cwd), stream=True)
        if result.get("rc") is None:
            raise RuntimeError("command exited without a confirmed exit status")
        return ShellResult(
            command,
            result["rc"],
            result["stdout"],
            "",
            round((time.monotonic() - started) * 1000),
        )

    def _run(self, command: str, cwd: str | None, *, stream: bool) -> dict[str, Any]:
        if self.cancel.is_set():
            raise RunCancelled()
        return asyncio.run_coroutine_threadsafe(
            self._run_async(command, cwd, stream=stream), self.loop
        ).result()

    async def _run_async(
        self, command: str, cwd: str | None, *, stream: bool
    ) -> dict[str, Any]:
        execution_id = uuid4().hex
        await asyncio.to_thread(
            self.emit,
            {
                "event": "execution",
                "execution": {
                    "request_id": execution_id,
                    "phase": "executing" if stream else "preparing",
                },
            },
        )

        async def output(event: JsonObject) -> None:
            async with self._output_lock:
                await asyncio.to_thread(
                    self.emit, {"event": "output", "text": event.get("text", "")}
                )

        self.subscribe(execution_id, output)
        task = asyncio.create_task(
            self.request(
                "execution.run",
                {
                    "id": execution_id,
                    "command": command,
                    "cwd": cwd,
                    "stream": stream,
                    **(
                        {"judge_directory": f"{self.root}/.judge/{execution_id}"}
                        if self.model_judge and cwd == self.verification
                        else {}
                    ),
                },
            )
        )
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                if self.cancel.is_set() and not task.done():
                    await asyncio.wait_for(
                        self.request("execution.cancel", {"id": execution_id}), 5
                    )
                    await asyncio.wait_for(asyncio.shield(task), 10)
                    break
            result = await task
            if isinstance(result.get("execution"), dict):
                await asyncio.to_thread(
                    self.emit,
                    {"event": "execution", "execution": result["execution"]},
                )
            if self.cancel.is_set():
                if result.get("stopped") is True or isinstance(result.get("rc"), int):
                    raise RunCancelled()
                raise RuntimeError(
                    "Cancellation requested; command termination is unconfirmed"
                )
            if result.get("error"):
                raise RuntimeError(
                    f"Execution infrastructure interrupted: {result['error']}"
                )
            return result
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.subscribe(execution_id, None)


class AgentGenomeBackend:
    def __init__(self, root: Path, host: Callable[[str], ExecutionHost]) -> None:
        self.service = GenomeService(root)
        self.host = host
        self.service.claim_execution_owner()
        self.service.recover()

    async def close(self) -> None:
        self.service.release_execution_owner()

    async def invoke(
        self, scope: str, action: str, arguments: JsonObject, request_id: str
    ) -> JsonObject:
        host = self.host(scope)
        capabilities = set()
        if action in {"get", "status", "run", "prepare_revision"}:
            available = await host.request("execution.capabilities", {})
            capabilities = {key for key, value in available.items() if value is True}
        return await asyncio.to_thread(
            invoke_genome,
            self.service,
            scope,
            action,
            arguments,
            request_id,
            capabilities=capabilities,
            host=host,
        )

    async def execute(
        self, scope: str, run_id: str, emit: WorkflowEventSink
    ) -> JsonObject:
        loop = asyncio.get_running_loop()
        host = self.host(scope)

        def event(payload: dict[str, Any]) -> Any:
            return asyncio.run_coroutine_threadsafe(emit(payload), loop).result()

        return await asyncio.to_thread(self.service.execute, scope, run_id, host, event)
