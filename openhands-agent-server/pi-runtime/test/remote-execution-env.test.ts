import assert from "node:assert/strict";
import test from "node:test";
import type { JsonObject, JsonValue } from "../src/protocol.ts";
import { RemoteExecutionEnv } from "../src/remote-execution-env.ts";
import type { RunnerRpcClient } from "../src/rpc-peer.ts";

class FakeRpc implements RunnerRpcClient {
  calls: Array<{ method: string; params: JsonObject }> = [];

  async request(method: string, params: JsonObject): Promise<JsonValue> {
    this.calls.push({ method, params });
    if (method === "env.readTextFile") return { ok: true, value: "hello" };
    if (method === "env.exec") {
      return { ok: true, value: { stdout: "out", stderr: "err", exitCode: 0 } };
    }
    if (method === "env.exists") {
      return { ok: false, error: { code: "permission_denied", message: "denied", path: "/outside" } };
    }
    return { ok: true, value: null };
  }
}

test("RemoteExecutionEnv converts file and shell RPC results", async () => {
  const rpc = new FakeRpc();
  const env = new RemoteExecutionEnv(rpc, "/workspace");
  const text = await env.readTextFile("a.txt");
  const stdout: string[] = [];
  const stderr: string[] = [];
  const execution = await env.exec("pwd", {
    timeout: 5,
    onStdout: (chunk) => stdout.push(chunk),
    onStderr: (chunk) => stderr.push(chunk),
  });
  const denied = await env.exists("/outside");

  assert.deepEqual(text, { ok: true, value: "hello" });
  assert.deepEqual(execution, { ok: true, value: { stdout: "out", stderr: "err", exitCode: 0 } });
  assert.deepEqual(stdout, ["out"]);
  assert.deepEqual(stderr, ["err"]);
  assert.equal(denied.ok, false);
  if (!denied.ok) {
    assert.equal(denied.error.code, "permission_denied");
    assert.equal(denied.error.path, "/outside");
  }
  assert.equal(rpc.calls[1]?.method, "env.exec");
});

test("RemoteExecutionEnv encodes binary writes without local filesystem access", async () => {
  const rpc = new FakeRpc();
  const env = new RemoteExecutionEnv(rpc, "/workspace");
  const result = await env.writeFile("a.bin", new Uint8Array([1, 2, 3]));
  assert.deepEqual(result, { ok: true, value: undefined });
  assert.deepEqual(rpc.calls[0], {
    method: "env.writeFile",
    params: { path: "a.bin", content: "AQID", encoding: "base64" },
  });
});
