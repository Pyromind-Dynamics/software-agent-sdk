import assert from "node:assert/strict";
import test from "node:test";
import { PassThrough } from "node:stream";
import {
  encodeMessage,
  PROTOCOL_VERSION,
  type PiRunnerRequest,
  type PiRunnerResponse,
} from "../src/protocol.ts";
import { JsonlRpcPeer } from "../src/rpc-peer.ts";

function parseRequest(line: string): PiRunnerRequest {
  return JSON.parse(line) as PiRunnerRequest;
}

test("JSONL RPC resolves once the response arrives", async () => {
  const input = new PassThrough();
  const output = new PassThrough();
  const peer = new JsonlRpcPeer(input, output);
  const listen = peer.listen(async () => ({})).then(
    () => undefined,
    () => undefined,
  );

  const pending = peer.request("echo", { value: 7 });
  output.once("data", (chunk: Buffer) => {
    const request = parseRequest(String(chunk).trim());
    const response: PiRunnerResponse = {
      protocolVersion: PROTOCOL_VERSION,
      type: "response",
      requestId: request.requestId,
      result: { echoed: true },
    };
    input.write(`${encodeMessage(response)}\n`);
  });

  assert.deepEqual(await pending, { echoed: true });
  input.end();
  await listen;
});

test("JSONL RPC rejects a request whose response never arrives", async () => {
  const input = new PassThrough();
  const output = new PassThrough();
  const outgoing: string[] = [];
  output.on("data", (chunk: Buffer) => {
    outgoing.push(...String(chunk).split("\n").filter(Boolean));
  });
  const peer = new JsonlRpcPeer(input, output, { requestTimeoutMs: 25 });
  const listen = peer.listen(async () => ({})).then(
    () => undefined,
    () => undefined,
  );

  await assert.rejects(
    peer.request("tool.execute", { tool_name: "x", arguments: {} }),
    /timed out after 25ms/,
  );
  const requests = outgoing.map(parseRequest);
  const cancel = requests.find(
    (request) => request.method === "rpc.cancel",
  );
  if (cancel === undefined) throw new Error("expected an rpc.cancel request");
  const original = requests[0];
  if (original === undefined) throw new Error("expected an rpc request");
  assert.equal(cancel.params.request_id, original.requestId);
  input.end();
  await listen;
});
