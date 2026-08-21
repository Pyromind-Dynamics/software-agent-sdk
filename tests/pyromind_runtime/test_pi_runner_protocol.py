from __future__ import annotations

import pytest
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from harness_adapter.pi_adapter.protocol import (
    PROTOCOL_VERSION,
    PiProtocolError,
    decode_frame,
    encode_frame,
)


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
