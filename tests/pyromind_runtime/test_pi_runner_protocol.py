from __future__ import annotations

import asyncio
import os
import signal

import pytest
from harness_adapter.pi_adapter.event_translator import translate_runner_event
from harness_adapter.pi_adapter.protocol import (
    PROTOCOL_VERSION,
    PiProtocolError,
    decode_frame,
    encode_frame,
)
from harness_adapter.pi_adapter.runner import PiRunnerExit, PiRunnerProcess
from pyromind_runtime.application import SnapshotProjector
from pyromind_runtime.application.event_projection import ProductEventProjector
from pyromind_runtime.domain.capabilities import HarnessCapabilities
from pyromind_runtime.domain.snapshot import ConversationSnapshot


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


def test_runner_entrypoint_resolves_from_env(monkeypatch, tmp_path) -> None:
    entrypoint = tmp_path / "dist" / "index.js"
    entrypoint.parent.mkdir()
    entrypoint.touch()
    monkeypatch.setenv("PYROMIND_PI_RUNTIME", str(entrypoint))

    async def noop(*_: object) -> None:
        pass

    runner = PiRunnerProcess(
        request_handler=noop,
        event_handler=noop,
        exit_handler=noop,
    )
    assert runner._entrypoint == entrypoint


async def test_runner_start_uses_a_new_process_session_on_posix(
    monkeypatch, tmp_path
) -> None:
    entrypoint = tmp_path / "index.js"
    entrypoint.touch()
    captured: dict[str, object] = {}

    class EmptyStream:
        async def readline(self) -> bytes:
            return b""

    class FinishedProcess:
        pid = 4321
        returncode = 0
        stdin = object()
        stdout = EmptyStream()
        stderr = EmptyStream()

        async def wait(self) -> int:
            return 0

    async def create_process(*_args: object, **kwargs: object) -> FinishedProcess:
        captured.update(kwargs)
        return FinishedProcess()

    exits: list[PiRunnerExit] = []

    async def noop(*_: object) -> None:
        pass

    async def record_exit(result: PiRunnerExit) -> None:
        exits.append(result)

    runner = PiRunnerProcess(
        request_handler=noop,
        event_handler=noop,
        exit_handler=record_exit,
        entrypoint=entrypoint,
    )

    async def start_request(_method: str, _params: dict[str, object]) -> dict:
        return {}

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(runner, "request", start_request)

    await runner.start({})
    await runner.terminate(reason="shutdown")

    if os.name == "posix":
        assert captured["start_new_session"] is True
    else:
        assert "start_new_session" not in captured
    assert exits == [PiRunnerExit(returncode=0, reason="shutdown")]


async def test_runner_reports_exit_once_with_explicit_reason(tmp_path) -> None:
    async def noop(*_: object) -> None:
        pass

    exits: list[PiRunnerExit] = []

    async def record_exit(result: PiRunnerExit) -> None:
        exits.append(result)

    runner = PiRunnerProcess(
        request_handler=noop,
        event_handler=noop,
        exit_handler=record_exit,
        entrypoint=tmp_path / "index.js",
    )
    runner._planned_exit_reason = "restart"

    await runner._notify_exit(0)
    await runner._notify_exit(0)

    assert exits == [PiRunnerExit(returncode=0, reason="restart")]


def test_runner_signals_its_process_group_on_posix(monkeypatch, tmp_path) -> None:
    async def noop(*_: object) -> None:
        pass

    class RunningProcess:
        pid = 4321

        def terminate(self) -> None:
            raise AssertionError("single-process fallback should not be used")

        def kill(self) -> None:
            raise AssertionError("single-process fallback should not be used")

    runner = PiRunnerProcess(
        request_handler=noop,
        event_handler=noop,
        exit_handler=noop,
        entrypoint=tmp_path / "index.js",
    )
    runner._owns_process_group = True
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        os, "killpg", lambda pid, requested: signals.append((pid, requested))
    )

    runner._signal_process(RunningProcess(), signal.SIGTERM)  # type: ignore[arg-type]
    runner._signal_process(RunningProcess(), signal.SIGKILL)  # type: ignore[arg-type]

    assert signals == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]


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


def test_runner_content_normalizes_pi_images_for_product_events() -> None:
    frames = (
        {
            "kind": "tool.completed",
            "payload": {
                "tool_call_id": "t1",
                "tool_name": "read",
                "content": [
                    {"type": "text", "text": "image"},
                    {
                        "type": "image",
                        "data": "aGVsbG8=",
                        "mimeType": "image/jpeg",
                    },
                ],
            },
        },
        {
            "kind": "message.completed",
            "payload": {
                "message_id": "m1",
                "role": "assistant",
                "content": [
                    {
                        "type": "image",
                        "data": "d29ybGQ=",
                        "mime_type": "image/png",
                    }
                ],
            },
        },
    )

    translated = []
    for index, frame in enumerate(frames, start=1):
        translated.extend(
            translate_runner_event(
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "type": "pi.event",
                    "eventId": f"e{index}",
                    "sessionId": "s1",
                    "runId": "r1",
                    "occurredAt": "2026-08-19T00:00:00Z",
                    **frame,
                }
            )
        )

    assert translated[0].payload["output"] == [
        {"type": "text", "text": "image"},
        {
            "type": "image",
            "image_urls": ["data:image/jpeg;base64,aGVsbG8="],
        },
    ]
    assert translated[1].payload["content"] == [
        {
            "type": "image",
            "image_urls": ["data:image/png;base64,d29ybGQ="],
        }
    ]


def test_runner_content_preserves_product_images_and_safely_omits_invalid_blocks(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
                "content": [
                    {
                        "type": "image",
                        "image_urls": ["data:image/png;base64,aGVsbG8="],
                    },
                    {"type": "image_url", "url": "https://example.com/a.png"},
                    {"type": "image", "data": "secret-image-data"},
                    {"type": "future-content", "value": "unsupported"},
                ],
            },
        }
    )

    assert events[0].payload["output"] == [
        {
            "type": "image",
            "image_urls": ["data:image/png;base64,aGVsbG8="],
        },
        {"type": "image_url", "url": "https://example.com/a.png"},
        {
            "type": "text",
            "text": "[Unsupported content omitted: invalid Pi content block.]",
        },
        {
            "type": "text",
            "text": "[Unsupported content omitted: invalid Pi content block.]",
        },
    ]
    assert "secret-image-data" not in caplog.text


def test_three_pi_image_results_are_accepted_by_snapshot_projection() -> None:
    harness_events = []
    for index in range(3):
        harness_events.extend(
            translate_runner_event(
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "type": "pi.event",
                    "eventId": f"image-{index}",
                    "sessionId": "s1",
                    "runId": "r1",
                    "occurredAt": "2026-08-19T00:00:00Z",
                    "kind": "tool.completed",
                    "payload": {
                        "tool_call_id": f"read-{index}",
                        "tool_name": "read",
                        "content": [
                            {
                                "type": "image",
                                "data": "aGVsbG8=",
                                "mimeType": "image/jpeg",
                            }
                        ],
                    },
                }
            )
        )

    event_projector = ProductEventProjector()
    snapshot_projector = SnapshotProjector()
    snapshot = ConversationSnapshot(
        conversation_id="s1", capabilities=HarnessCapabilities()
    )
    for sequence, harness_event in enumerate(harness_events, start=1):
        product_event = event_projector.project("s1", harness_event)
        assert product_event is not None
        snapshot = snapshot_projector.reduce(
            snapshot, product_event.model_copy(update={"seq": sequence})
        )

    assert len(snapshot.timeline) == 3
    assert all(item.status == "completed" for item in snapshot.timeline)


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
