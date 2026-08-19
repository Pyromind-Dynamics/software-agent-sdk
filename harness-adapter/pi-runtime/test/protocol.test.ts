import assert from "node:assert/strict";
import test from "node:test";
import { decodeFrame, encodeFrame, PROTOCOL_VERSION } from "../src/protocol.js";

test("protocol round trips a request", () => {
  const frame = { protocolVersion: PROTOCOL_VERSION, type: "request" as const, requestId: "r1", method: "start", params: {} };
  assert.deepEqual(decodeFrame(encodeFrame(frame)), frame);
});

test("protocol rejects malformed and oversized frames", () => {
  assert.throws(() => decodeFrame("not-json"), /invalid JSONL/);
  assert.throws(() => decodeFrame("x".repeat(1024 * 1024 + 1)), /size limit/);
});
