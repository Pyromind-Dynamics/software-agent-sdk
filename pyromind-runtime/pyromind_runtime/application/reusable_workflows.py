from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pyromind_runtime.domain.content import JsonObject
from pyromind_runtime.ports.workflows import WorkflowBackend


type Publish = Callable[[str, str, str, JsonObject], Coroutine[Any, Any, None]]


@dataclass
class _Run:
    run_id: str
    origin_run_id: str | None = None
    cancel_requested: bool = False
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None


class ReusableWorkflows:
    def __init__(
        self,
        backend: WorkflowBackend,
        publish: Publish,
        complete: Callable[[str, JsonObject], Coroutine[Any, Any, None]] | None = None,
        agent_task: Callable[[str, JsonObject], Coroutine[Any, Any, JsonObject]]
        | None = None,
    ) -> None:
        self.agent_task = agent_task
        self.complete = complete
        self.backend = backend
        self.publish = publish
        self.active: dict[str, _Run] = {}
        self.deferred: set[asyncio.Task[None]] = set()

    def busy(self, scope: str) -> bool:
        return scope in self.active

    async def invoke(
        self,
        scope: str,
        action: str,
        arguments: JsonObject,
        request_id: str,
        origin_run_id: str | None = None,
    ) -> JsonObject:
        result = await self.backend.invoke(scope, action, arguments, request_id)
        if action == "run" and result.get("status") == "queued":
            run_id = result.get("id")
            if not isinstance(run_id, str):
                raise ValueError("workflow backend returned no run id")
            if scope not in self.active or self.active[scope].run_id != run_id:
                run = _Run(run_id, origin_run_id)
                self.active[scope] = run
                run.task = asyncio.create_task(self._execute(scope, run))
        if (
            action == "cancel"
            and scope in self.active
            and self.active[scope].run_id == arguments.get("run_id")
        ):
            self.active[scope].cancel_requested = True
            if result.get("status") == "cancelled":
                self.release(scope)
        return result

    def release(self, scope: str, origin_run_id: str | None = None) -> None:
        if run := self.active.get(scope):
            if origin_run_id is not None and run.origin_run_id not in {
                None,
                origin_run_id,
            }:
                return
            run.ready.set()

    async def wait_idle(self, scope: str) -> None:
        while run := self.active.get(scope):
            await run.done.wait()

    def defer(self, work: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(work)
        self.deferred.add(task)
        task.add_done_callback(self.deferred.discard)

    async def cancel(self, scope: str) -> None:
        if run := self.active.get(scope):
            await self.invoke(scope, "cancel", {"run_id": run.run_id}, uuid4().hex)
            run.ready.set()

    async def _execute(self, scope: str, run: _Run) -> None:
        operation_id = f"workflow:{run.run_id}"
        tail = ""
        sequence = 0
        last_output = 0.0
        current_node = None
        current_command = None
        result: JsonObject = {"id": run.run_id, "status": "interrupted"}

        async def emit(event: JsonObject) -> JsonObject | None:
            nonlocal tail, sequence, last_output, current_node, current_command
            if event.get("event") == "regression_started":
                current_node = None
                current_command = None
            if isinstance(event.get("node"), str):
                if current_node != event["node"]:
                    current_command = None
                current_node = event["node"]
            if isinstance(event.get("command"), str):
                current_command = event["command"]
            text = event.get("text")
            if isinstance(text, str):
                tail = (tail + text)[-32768:]
                now = time.monotonic()
                if now - last_output < 0.25:
                    return
                last_output = now
            stage_request = event.get("request")
            sequence += 1
            await self.publish(
                scope,
                f"{operation_id}:{sequence}",
                "operation.progress",
                {
                    "operation_id": operation_id,
                    "output": [{"type": "text", "text": tail}],
                    "details": {
                        "workflow_run_id": run.run_id,
                        "validation_case": event.get("validation_case"),
                        "candidate_version": event.get("candidate_version"),
                        "node": current_node,
                        "command": current_command,
                        "verdict": event.get("verdict"),
                        "event": event.get("event"),
                        "stage_request_id": stage_request.get("request_id")
                        if isinstance(stage_request, dict)
                        else None,
                    },
                },
            )

            if event.get("event") == "waiting_agent":
                if self.agent_task is None:
                    raise ValueError("Agent stages are not supported by this host")
                request = event.get("request")
                if not isinstance(request, dict):
                    raise ValueError("invalid agent stage request")
                task = asyncio.create_task(self.agent_task(scope, request))
                try:
                    while not task.done():
                        await asyncio.wait({task}, timeout=0.1)
                        if run.cancel_requested:
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                            return {"status": "cancelled"}
                    stage_result = await task
                    sequence += 1
                    await self.publish(
                        scope,
                        f"{operation_id}:{sequence}",
                        "operation.progress",
                        {
                            "operation_id": operation_id,
                            "details": {
                                "workflow_run_id": run.run_id,
                                "node": current_node,
                                "event": "agent_stage_finished",
                                "stage_request_id": request["request_id"],
                                "stage_result": stage_result,
                            },
                        },
                    )
                    return stage_result
                finally:
                    if not task.done():
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
            return None

        try:
            await run.ready.wait()
            await self.publish(
                scope,
                f"{operation_id}:start",
                "operation.started",
                {
                    "operation_id": operation_id,
                    "name": "历史经验",
                    "category": "tool",
                    "tool": "genome_run",
                    "arguments": {"workflow_run_id": run.run_id},
                },
            )
            result = await self.backend.execute(scope, run.run_id, emit)
            success = result.get("status") in {"succeeded", "cancelled"}
            await self.publish(
                scope,
                f"{operation_id}:end",
                "operation.completed" if success else "operation.failed",
                {
                    "operation_id": operation_id,
                    "output": [
                        {"type": "text", "text": json.dumps(result, ensure_ascii=False)}
                    ],
                    "details": result,
                },
            )
        except Exception as exc:
            result = {
                "id": run.run_id,
                "status": "interrupted",
                "result": {"error": str(exc)},
            }
            await self.publish(
                scope,
                f"{operation_id}:end",
                "operation.failed",
                {
                    "operation_id": operation_id,
                    "output": [{"type": "text", "text": str(exc)}],
                    "details": {"workflow_run_id": run.run_id, "status": "interrupted"},
                },
            )
        finally:
            if self.active.get(scope) is run:
                self.active.pop(scope, None)
        try:
            if run.cancel_requested:
                result = {**result, "cancel_requested": True}
            if self.complete is not None:
                await self.complete(scope, result)
        finally:
            run.done.set()

    async def close(self) -> None:
        for scope in tuple(self.active):
            await self.cancel(scope)
        tasks = [run.task for run in self.active.values() if run.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in self.deferred:
            task.cancel()
        await asyncio.gather(*self.deferred, return_exceptions=True)
        await self.backend.close()
