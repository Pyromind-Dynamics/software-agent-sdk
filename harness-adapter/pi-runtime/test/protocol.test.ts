import assert from "node:assert/strict";
import test from "node:test";
import { decodeFrame, encodeFrame, PROTOCOL_VERSION } from "../src/protocol.js";

test("protocol round trips a request", () => {
  assert.equal(PROTOCOL_VERSION, 2);
  const frame = { protocolVersion: PROTOCOL_VERSION, type: "request" as const, requestId: "r1", method: "start", params: {} };
  assert.deepEqual(decodeFrame(encodeFrame(frame)), frame);
});

test("protocol round trips the stable run outcome", () => {
  const frame = {
    protocolVersion: PROTOCOL_VERSION,
    type: "pi.event" as const,
    eventId: "e1",
    sessionId: "s1",
    runId: "r1",
    occurredAt: "2026-08-20T00:00:00Z",
    kind: "run.finished" as const,
    payload: { outcome: { status: "failed", stop_reason: "length", error_code: "output_truncated" } },
  };
  assert.deepEqual(decodeFrame(encodeFrame(frame)), frame);
});

test("protocol rejects malformed and oversized frames", () => {
  assert.throws(() => decodeFrame("not-json"), /invalid JSONL/);
  assert.throws(() => decodeFrame("x".repeat(1024 * 1024 + 1)), /size limit/);
});
