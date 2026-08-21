from __future__ import annotations

import json
from typing import Any


PROTOCOL_VERSION = 2
MAX_FRAME_BYTES = 1024 * 1024


class PiProtocolError(RuntimeError):
    pass


def encode_frame(frame: dict[str, Any]) -> bytes:
    data = json.dumps(frame, ensure_ascii=False, separators=(",", ":")).encode()
    if len(data) > MAX_FRAME_BYTES:
        raise PiProtocolError("JSONL frame exceeds size limit")
    return data + b"\n"


def decode_frame(line: bytes) -> dict[str, Any]:
    if len(line) > MAX_FRAME_BYTES:
        raise PiProtocolError("JSONL frame exceeds size limit")
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PiProtocolError("invalid JSONL frame") from exc
    if not isinstance(value, dict):
        raise PiProtocolError("JSONL frame must be an object")
    if value.get("protocolVersion") != PROTOCOL_VERSION:
        raise PiProtocolError("unsupported protocol version")
    if value.get("type") not in {"request", "response", "pi.event"}:
        raise PiProtocolError("unknown JSONL frame type")
    return value
