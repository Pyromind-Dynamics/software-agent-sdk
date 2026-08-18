from __future__ import annotations

import json
import os
import sys
from typing import Any
from uuid import uuid4


session_id = "unknown"
active_run_id: str | None = None


def write(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def respond(request_id: str, result: Any = None) -> None:
    write(
        {
            "protocolVersion": 1,
            "type": "response",
            "requestId": request_id,
            "result": result,
        }
    )


def event(kind: str, run_id: str, payload: dict[str, Any]) -> None:
    write(
        {
            "protocolVersion": 1,
            "type": "pi.event",
            "eventId": uuid4().hex,
            "sessionId": session_id,
            "runId": run_id,
            "occurredAt": "2026-08-17T00:00:00Z",
            "kind": kind,
            "payload": payload,
        }
    )


for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    request_id = request["requestId"]
    params = request["params"]
    if method == "session.start":
        session_id = params["session_id"]
        respond(
            request_id,
            {
                "pid": os.getpid(),
                "credential_inherited": "OPENAI_API_KEY" in os.environ,
            },
        )
    elif method == "run.prompt":
        run_id = str(params["command_id"])
        active_run_id = run_id
        text = params["content"][0].get("text", "")
        respond(request_id, {"accepted": True})
        if text == "crash":
            sys.exit(7)
        event("agent.started", run_id, {})
        event(
            "message.started",
            run_id,
            {
                "message_id": f"{run_id}:user",
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        )
        event(
            "message.completed",
            run_id,
            {
                "message_id": f"{run_id}:user",
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        )
        event(
            "message.started",
            run_id,
            {
                "message_id": f"{run_id}:assistant",
                "role": "assistant",
                "content": [],
            },
        )
        if text == "wait":
            continue
        event(
            "message.delta",
            run_id,
            {"message_id": f"{run_id}:assistant", "text": "done"},
        )
        event(
            "message.completed",
            run_id,
            {
                "message_id": f"{run_id}:assistant",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
            },
        )
        event(
            "usage.updated",
            run_id,
            {
                "input_tokens": 1,
                "output_tokens": 1,
                "cached_tokens": 0,
                "cost_usd": 0,
            },
        )
        event("agent.completed", run_id, {})
        active_run_id = None
    elif method == "run.cancel":
        respond(request_id, {"cancelled": True})
        if active_run_id is not None:
            cancelled_run_id = active_run_id
            event("agent.cancelled", cancelled_run_id, {"outcome": "cancelled"})
            active_run_id = None
    elif method == "session.close":
        respond(request_id, {"closed": True})
        break
    else:
        write(
            {
                "protocolVersion": 1,
                "type": "response",
                "requestId": request_id,
                "error": {"code": "unknown_method", "message": method},
            }
        )
