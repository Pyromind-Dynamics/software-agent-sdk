from __future__ import annotations

import asyncio
import base64
import json
import shlex
import threading
import time
from collections.abc import Callable, Coroutine
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from agentgenome import GenomeService, RunCancelled
from core.ports import Ports, ShellResult
from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.ports.workflows import WorkflowEventSink


type Request = Callable[[str, JsonObject], Coroutine[Any, Any, Any]]
type Subscribe = Callable[[str, WorkflowEventSink | None], None]


class ExecutionHost:
    def __init__(self, request: Request, subscribe: Subscribe) -> None:
        self.request = request
        self.subscribe = subscribe
        self.loop = asyncio.get_running_loop()
        self.cancel = threading.Event()
        self.emit: Callable[[dict[str, Any]], None] = lambda _event: None
        self.root = ""
        self._output_lock = asyncio.Lock()

    def _call(self, method: str, payload: JsonObject) -> Any:
        return asyncio.run_coroutine_threadsafe(
            self.request(method, payload), self.loop
        ).result()

    def prepare(
        self,
        run_id: str,
        package: Path,
        params: dict[str, Any],
        path_parameters: list[str],
        cancel: threading.Event,
        emit: Callable[[dict[str, Any]], None],
    ) -> tuple[Ports, dict[str, Any]]:
        self.cancel, self.emit = cancel, emit
        self.root = self._resolve(f"public_data/workflow-runs/{run_id}", True)
        cwd = f"{self.root}/package"
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
            target = f"{cwd}/{file.relative_to(package).as_posix()}"
            self._write(target, file.read_bytes())
        self._python(
            f"from pathlib import Path; Path({self.root + '/artifacts'!r}).mkdir()"
        )
        resolved = {
            key: self._resolve(value, False) if key in path_parameters else value
            for key, value in params.items()
        }
        return Ports(shell=self, artifacts=self, cwd=Path(cwd)), resolved

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
        result = self._python(
            "import json\nfrom pathlib import Path\n"
            f"p=Path({self.path(name)!r})\n"
            "assert p.stat().st_size <= 65536, 'verification artifact exceeds 64 KiB'\n"
            "print(json.dumps(p.read_text(encoding='utf-8'), ensure_ascii=False))"
        )
        return json.loads(result)

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
            if self.cancel.is_set():
                if result.get("stopped") is True or isinstance(result.get("rc"), int):
                    raise RunCancelled()
                raise RuntimeError(
                    "Cancellation requested; command termination is unconfirmed"
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
        def invoke() -> JsonObject:
            if action == "list":
                return {"assets": [asset for asset in self.service.list()]}
            if action == "get":
                asset = self.service.get(
                    str(arguments["asset_id"]), str(arguments["version"])
                )
                if not asset["published"]:
                    raise ValueError("asset is not published")
                return asset
            if action == "run":
                params = arguments.get("params")
                if not isinstance(params, dict):
                    raise ValueError("params must be an object")
                return self.service.submit(
                    scope,
                    request_id,
                    str(arguments["asset_id"]),
                    str(arguments["version"]),
                    params,
                )
            if action in {"status", "cancel"}:
                method = (
                    self.service.status if action == "status" else self.service.cancel
                )
                return method(scope, str(arguments["run_id"]))
            raise ValueError("unknown workflow action")

        return await asyncio.to_thread(invoke)

    async def execute(
        self, scope: str, run_id: str, emit: WorkflowEventSink
    ) -> JsonObject:
        loop = asyncio.get_running_loop()
        host = self.host(scope)

        def event(payload: dict[str, Any]) -> None:
            asyncio.run_coroutine_threadsafe(emit(payload), loop).result()

        return await asyncio.to_thread(self.service.execute, scope, run_id, host, event)
