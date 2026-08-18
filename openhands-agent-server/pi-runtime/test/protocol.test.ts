import assert from "node:assert/strict";
import test from "node:test";
import { decodeMessage, encodeMessage, PROTOCOL_VERSION, type PiRunnerRequest } from "../src/protocol.ts";

test("JSONL protocol round-trips a request", () => {
  const message: PiRunnerRequest = {
    protocolVersion: PROTOCOL_VERSION,
    type: "request",
    requestId: "request-1",
    method: "run.prompt",
    params: { text: "hello" },
  };

  assert.deepEqual(decodeMessage(encodeMessage(message)), message);
});

test("JSONL protocol rejects invalid input and ambiguous responses", () => {
  assert.throws(() => decodeMessage("not json"), /invalid JSONL/);
  assert.throws(
    () =>
      decodeMessage(
        JSON.stringify({
          protocolVersion: PROTOCOL_VERSION,
          type: "response",
          requestId: "request-1",
          result: null,
          error: { code: "bad", message: "bad" },
        }),
      ),
    /exactly one/,
  );
});
