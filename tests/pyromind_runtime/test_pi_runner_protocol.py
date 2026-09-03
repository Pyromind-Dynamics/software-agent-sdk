from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import harness_adapter.pi_adapter.runner as runner_module
import pytest
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from harness_adapter.pi_adapter.protocol import (
    PROTOCOL_VERSION,
    PiProtocolError,
    decode_frame,
    encode_frame,
)
from harness_adapter.pi_adapter.runner import PiRunnerProcess


def test_protocol_round_trip_and_frame_limit() -> None:
    frame = {
        "protocolVersion": PROTOCOL_VERSION,
        "type": "request",
        "requestId": "r1",
        "method": "start",
        "params": {},
    }
    assert decode_frame(encode_frame(frame).rstrip()) == frame
    with pytest.raises(PiProtocolError, match="size limit"):
        decode_frame(b"x" * (1024 * 1024 + 1))


def test_runner_tool_event_has_product_operation_shape_and_object_details() -> None:
    events = translate_runner_event(
        {
            "protocolVersion": PROTOCOL_VERSION,
            "type": "pi.event",
            "eventId": "e1",
            "sessionId": "s1",
            "runId": "r1",
            "occurredAt": "2026-08-19T00:00:00Z",
            "kind": "tool.completed",
            "payload": {
                "tool_call_id": "t1",
                "tool_name": "read",
                "content": [{"type": "text", "text": "ok"}],
                "details": "not-an-object",
            },
        }
    )
    assert events[0].type == "operation.completed"
    assert events[0].payload["operation_id"] == "t1"
    assert events[0].payload["details"] is None


@pytest.mark.parametrize(
    ("status", "expected_status", "event_types"),
    [
        ("completed", "idle", ["status.changed"]),
        ("failed", "error", ["notice.raised", "status.changed"]),
        ("cancelled", "paused", ["status.changed"]),
        ("suspended", "paused", ["status.changed"]),
        ("future-outcome", "error", ["notice.raised", "status.changed"]),
    ],
)
def test_run_finished_is_the_only_harness_terminal_mapping(
    status: str, expected_status: str, event_types: list[str]
) -> None:
    events = translate_runner_event(
        {
            "protocolVersion": PROTOCOL_VERSION,
            "type": "pi.event",
            "eventId": "e1",
            "sessionId": "s1",
            "runId": "r1",
            "occurredAt": "2026-08-20T00:00:00Z",
            "kind": "run.finished",
            "payload": {
                "outcome": {
                    "status": status,
                    "stop_reason": "length",
                    "error_code": "output_truncated",
                    "message": "truncated",
                }
            },
        }
    )

    assert [event.type for event in events] == event_types
    assert events[-1].payload["status"] == expected_status
    assert all("stop_reason" not in event.payload for event in events)
    if status == "future-outcome":
        assert events[0].payload["code"] == "unknown_pi_outcome"


async def test_oversized_response_still_resolves_the_node_request() -> None:
    sent: list[dict] = []

    class _FakeStdin:
        def write(self, data: bytes) -> None:
            sent.append(json.loads(data))

        async def drain(self) -> None:
            return None

    async def huge_result(_method: str, _params: dict) -> str:
        return "x" * (1024 * 1024 + 1)

    runner = PiRunnerProcess(
        request_handler=huge_result,
        event_handler=_async_noop,
        exit_handler=_async_noop,
    )
    runner._process = cast(Any, SimpleNamespace(stdin=_FakeStdin()))

    await runner._answer_request(
        {
            "protocolVersion": PROTOCOL_VERSION,
            "type": "request",
            "requestId": "r1",
            "method": "tool.execute",
            "params": {},
        }
    )

    assert sent == [
        {
            "protocolVersion": PROTOCOL_VERSION,
            "type": "response",
            "requestId": "r1",
            "error": {
                "code": "response_too_large",
                "message": "runner response exceeded the JSONL frame limit",
            },
        }
    ]


def test_runner_entrypoint_prefers_pi_runtime_entrypoint_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entrypoint = tmp_path / "dist" / "index.js"
    monkeypatch.setenv("PI_RUNTIME_ENTRYPOINT", str(entrypoint))
    runner = PiRunnerProcess(
        request_handler=_async_noop, event_handler=_async_noop, exit_handler=_async_noop
    )
    assert runner._entrypoint == entrypoint


def test_runner_entrypoint_defaults_to_repo_relative_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PI_RUNTIME_ENTRYPOINT", raising=False)
    runner = PiRunnerProcess(
        request_handler=_async_noop, event_handler=_async_noop, exit_handler=_async_noop
    )
    expected = (
        Path(runner_module.__file__).parents[2] / "pi-runtime" / "dist" / "index.js"
    )
    assert runner._entrypoint == expected


async def _async_noop(*_args) -> None:
    return None
