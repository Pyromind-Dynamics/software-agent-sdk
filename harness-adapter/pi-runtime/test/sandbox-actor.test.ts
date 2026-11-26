import assert from "node:assert/strict";
import test from "node:test";

import { FallbackSandboxFileClient, SandboxEndpointSession, SandboxFileClient } from "../src/sandbox-operations.js";
import { LazySandboxTerminalOperations, SandboxTerminalExecutionError } from "../src/sandbox-terminal.js";

const endpoint = {
  baseUrl: "http://router.example/proxy/",
  actorTarget: "dev/alice", sandboxId: "dev/alice",
  workspacePath: "/workspace/.pyromind-agent/chat", storagePath: "/workspace",
};

function runState(run_id: string, status = "running", stdout = Buffer.alloc(0), offset = stdout.length) {
  return Response.json({
    run_id, status, exit_code: status === "failed" ? 7 : 0,
    stdout: stdout.toString("base64"), stdout_offset: offset,
    stderr: "", stderr_offset: 0, stdout_truncated: false, stderr_truncated: false,
  });
}

function actorUrl(input: RequestInfo | URL, init?: RequestInit): URL {
  const url = new URL(String(input));
  assert.equal(url.origin, "http://router.example");
  assert.equal(new Headers(init?.headers).get("ate-target-actor"), endpoint.actorTarget);
  assert.equal(new Headers(init?.headers).has("authorization"), false);
  assert.equal(init?.redirect, "error");
  return url;
}

test("terminal passes SDK env as data, reads all final pages and preserves nonzero exit", async (t) => {
  const page = Buffer.alloc(64 * 1024, "a");
  let id = "";
  t.mock.method(globalThis, "fetch", async (input: URL, init: RequestInit) => {
    const url = actorUrl(input, init);
    if (url.pathname === "/proxy/exec") {
      assert.equal(init.method, "POST");
      const request = JSON.parse(String(init.body));
      assert.deepEqual(request.command, ["/usr/bin/env", "PYROMIND_API_KEY=key with ' quotes", "/bin/bash", "-c", "run"]);
      assert.equal(request.cwd, endpoint.workspacePath + "/public_data/project");
      assert.equal(request.timeout_seconds, 20);
      id = request.run_id;
      return runState(id, "failed", page);
    }
    assert.equal(url.pathname, "/proxy/exec/" + id);
    assert.equal(url.searchParams.get("stdout_offset"), String(page.length));
    assert.equal(url.searchParams.get("stderr_offset"), "0");
    return runState(id, "failed", Buffer.from("tail"), page.length + 4);
  });
  const output: Buffer[] = [];
  const terminal = new LazySandboxTerminalOperations(new SandboxEndpointSession(async () => endpoint));
  const result = await terminal.exec("run", "public_data/project", {
    env: { PYROMIND_API_KEY: "key with ' quotes" }, timeout: 20, onData: (chunk) => output.push(chunk),
  });
  assert.deepEqual(Buffer.concat(output), Buffer.concat([page, Buffer.from("tail")]));
  assert.equal(result.exitCode, 7);
  assert.equal(result.execution.execution_id, id);
  assert.equal(result.execution.stopped, true);
});

test("aborting an Actor command confirms remote cancellation using the same run ID", async (t) => {
  const controller = new AbortController();
  const routes: string[] = [];
  let id = "";
  t.mock.method(globalThis, "fetch", async (input: URL, init: RequestInit) => {
    const url = actorUrl(input, init);
    routes.push(url.pathname);
    if (url.pathname === "/proxy/exec") {
      id = JSON.parse(String(init.body)).run_id;
      controller.abort();
      return runState(id);
    }
    return runState(id, url.pathname.endsWith("/cancel") ? "running" : "canceled");
  });
  const terminal = new LazySandboxTerminalOperations(new SandboxEndpointSession(async () => endpoint));
  await assert.rejects(terminal.exec("sleep 30", ".", { signal: controller.signal, onData() {} }),
    (error: unknown) => error instanceof SandboxTerminalExecutionError &&
      error.execution.error_code === "aborted" && error.execution.stopped);
  assert.deepEqual(routes, ["/proxy/exec", "/proxy/exec/" + id + "/cancel", "/proxy/exec/" + id]);
});

test("workflow plan files use the Actor file API, creating missing parents and preserving bytes", async (t) => {
  const path = endpoint.workspacePath + "/public_data/lane/public_data/workflowPLAN.json";
  const bytes = Buffer.from('{"plan":"测试"}');
  const requests: string[] = [];
  t.mock.method(globalThis, "fetch", async (input: URL, init: RequestInit) => {
    const url = actorUrl(input, init);
    requests.push((init.method ?? "GET") + " " + url.pathname);
    if (url.pathname === "/proxy/exec") {
      const request = JSON.parse(String(init.body));
      assert.deepEqual(request.command, ["mkdir", "-p", "--", endpoint.workspacePath + "/public_data/lane/public_data"]);
      return runState(request.run_id, "succeeded");
    }
    assert.equal(url.searchParams.get("path"), path);
    if (init.method === "PUT") {
      assert.deepEqual(Buffer.from(init.body as Uint8Array), bytes);
      return new Response(null, { status: requests.length === 1 ? 404 : 204 });
    }
    return new Response(bytes);
  });
  const endpoints = new SandboxEndpointSession(async () => endpoint);
  const unexpected = async (): Promise<never> => { throw new Error("unexpected platform terminal fallback"); };
  const files = new FallbackSandboxFileClient(new SandboxFileClient(endpoints),
    { read: unexpected, access: unexpected, write: unexpected }, { endpoints });
  await files.write(path, bytes);
  assert.deepEqual(await files.read(path), bytes);
  assert.deepEqual(requests, ["PUT /proxy/files", "POST /proxy/exec", "PUT /proxy/files", "GET /proxy/files"]);
});

test("an unavailable fixed Actor fails this call and resolves again on the next one", async () => {
  let requests = 0;
  const terminal = new LazySandboxTerminalOperations(new SandboxEndpointSession(async () => {
    if (++requests === 1) throw Object.assign(new Error("Actor is not ready"), { code: "fixed_actor_unavailable" });
    return endpoint;
  }));
  await assert.rejects(terminal.exec("echo hello", ".", { onData() {} }),
    (error: unknown) => error instanceof SandboxTerminalExecutionError &&
      error.execution.error_code === "fixed_actor_unavailable" && !error.execution.started);
  assert.equal(requests, 1);
  assert.deepEqual(await terminal.endpoint(), endpoint);
  assert.equal(requests, 2);
});

test("a lost Actor launch response is recovered by run ID without a second launch", async (t) => {
  let id = "";
  let launches = 0;
  t.mock.method(globalThis, "fetch", async (input: URL, init: RequestInit) => {
    const url = actorUrl(input, init);
    if (init.method === "POST") {
      launches++;
      id = JSON.parse(String(init.body)).run_id;
      throw new TypeError("fetch failed");
    }
    assert.equal(url.pathname, "/proxy/exec/" + id);
    return runState(id, "succeeded");
  });
  const terminal = new LazySandboxTerminalOperations(new SandboxEndpointSession(async () => endpoint));
  const result = await terminal.exec("echo one", ".", { onData() {} });
  assert.equal(result.exitCode, 0);
  assert.equal(launches, 1);
});
