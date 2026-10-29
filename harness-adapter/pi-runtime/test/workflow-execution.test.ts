import assert from "node:assert/strict";
import test from "node:test";
import { PassThrough } from "node:stream";
import { MAX_READ_BYTES, WorkflowExecution } from "../src/workflow-execution.js";
import { JsonlRpcPeer } from "../src/rpc-peer.js";
import type { JsonObject, JsonValue } from "../src/protocol.js";
import { SandboxTerminalExecutionError } from "../src/sandbox-terminal.js";

function peer() { return new JsonlRpcPeer(new PassThrough(), new PassThrough()); }

test("remote infrastructure errors preserve safe evidence and do not claim cancellation", async () => {
  const evidence = {execution_id: "remote-id", remote_run_dir: "/session/runs/remote-id",
    phase: "starting" as const, started: false, stopped: false, error_code: "startup_timeout"};
  const execution = new WorkflowExecution({
    cwd: "/session", writeFile: async () => {}, resolvePath: async (path) => path,
    operations: {async exec() {throw new SandboxTerminalExecutionError("timeout:300", evidence);}},
  }, peer());
  const result = await execution.handle("execution.run", {id:"one",command:"python3 helper.py"}) as Record<string, unknown>;
  assert.deepEqual(result.execution, evidence);
  assert.equal(result.stopped, false);
  assert.equal(result.rc, null);
  assert.match(String(result.error), /phase=starting/);
});

test("workflow execution uses host operations and quoted cwd, never another shell", async () => {
  const calls: unknown[] = [];
  const execution = new WorkflowExecution({
    writeFile: async () => { assert.fail("must not write files"); },
    cwd: "/session", env: { PATH: "/bin" },
    resolvePath: async (path) => path,
    operations: { async exec(command, cwd, options) {
      calls.push({ command, cwd, env: options.env });
      options.onData(Buffer.from("report"));
      return { exitCode: 0 };
    } },
  }, peer());
  assert.deepEqual(await execution.handle("execution.run", {
    id: "one", command: "python3 clean.py", cwd: "/session/public_data/a'b",
  }), { rc: 0, stdout: "report", truncated: false });
  assert.deepEqual(calls, [{ command: `cd '/session/public_data/a'"'"'b' && python3 clean.py`, cwd: "/session", env: { PATH: "/bin" } }]);
});

test("cancelled commands only report stopped when the host confirms termination", async () => {
  for (const confirmsAbort of [true, false]) {
    let started: () => void = () => {};
    const ready = new Promise<void>((resolve) => { started = resolve; });
    const execution = new WorkflowExecution({
      writeFile: async () => { assert.fail("must not write files"); },
    cwd: "/session", confirmsAbort,
      resolvePath: async (path) => path,
      operations: { async exec(_command, _cwd, options) {
        started();
        await new Promise<void>((_resolve, reject) => options.signal?.addEventListener("abort", () => reject(new Error("aborted"))));
        return { exitCode: 0 };
      } },
    }, peer());
    const running = execution.handle("execution.run", { id: "one", command: "sleep 30" });
    await ready;
    await execution.handle("execution.cancel", { id: "one" });
    if (confirmsAbort) assert.deepEqual(await running, { rc: null, stdout: "", stopped: true, truncated: false });
    else await assert.rejects(running, /aborted/);
  }
});

test("path policy rejection never invokes execution", async () => {
  const execution = new WorkflowExecution({
    writeFile: async () => { assert.fail("must not write files"); },
    cwd: "/session", resolvePath: async () => { throw new Error("PATH_SCOPE_ERROR"); },
    operations: { async exec() { assert.fail("must not execute"); } },
  }, peer());
  await assert.rejects(execution.handle("execution.run", { id: "one", command: "pwd", cwd: "/other" }), /PATH_SCOPE_ERROR/);
});


test("script upload uses the host file API with exact bytes and enforces path policy", async () => {
  const files = new Map<string, Buffer>();
  const execution = new WorkflowExecution({
    cwd: "/session",
    resolvePath: async (path, operation) => {
      assert.equal(operation, "write");
      if (!path.startsWith("public_data/")) throw new Error("PATH_SCOPE_ERROR");
      return `/session/${path}`;
    },
    writeFile: async (path, content) => { files.set(path, content); },
    operations: { async exec() { assert.fail("upload must not use terminal"); } },
  }, peer());
  const content = Buffer.from("print('历史经验')\n".repeat(2000));
  await execution.handle("execution.write", { path: "public_data/run/script.py", content: content.toString("base64") });
  assert.deepEqual(files.get("/session/public_data/run/script.py"), content);
  await assert.rejects(execution.handle("execution.write", { path: "/other/script.py", content: "" }), /PATH_SCOPE_ERROR/);
  assert.equal(files.size, 1);
});

test("verification files use a protected host directory and independent bounded reads", async () => {
  const files = new Map<string, Buffer>();
  const execution = new WorkflowExecution({
    cwd: "/session", protectedRoot: "/session/pi/verification",
    resolvePath: async (path, op) => {
      if (op === "write" && !path.startsWith("/session/public_data/")) throw new Error("PATH_SCOPE_ERROR");
      return path;
    },
    writeFile: async (path, content) => { files.set(path, content); },
    readFile: async (path) => files.get(path)!,
    operations: { async exec() { assert.fail("file transfer must not use terminal"); } },
  }, peer());
  const runId = "a".repeat(32);
  const result = await execution.handle("execution.protect", { run_id: runId, files: [{path:"verify.py",content:Buffer.from("assert True").toString("base64")}] }) as {path:string};
  const path = `${result.path}/verify.py`;
  await assert.rejects(execution.handle("execution.write", { path, content: "" }), /PATH_SCOPE_ERROR/);
  assert.deepEqual(await execution.handle("execution.read", {path,limit:32}), {content:Buffer.from("assert True").toString("base64")});
  await assert.rejects(execution.handle("execution.read", {path,limit:2}), /size limit/);
  await assert.rejects(execution.handle("execution.protect", {run_id:runId,files:[{path:"../escape",content:""}]}), /protected path/);
});

test("batched reads answer every path, report missing files, and page long lists", async () => {
  const files = new Map<string, Buffer>([["/session/scripts/a.py", Buffer.from("assert True")]]);
  const execution = new WorkflowExecution({
    cwd: "/session",
    resolvePath: async (path) => `/session/${path}`,
    readFile: async (path) => {
      const content = files.get(path);
      if (!content) throw new Error("FILE_READ_ERROR");
      return content;
    },
    writeFile: async () => {},
    operations: { async exec() { assert.fail("batched reads must not use terminal"); } },
  }, peer());
  const result = await execution.handle("execution.read_many",
    { paths: ["scripts/a.py", "scripts/ghost.py"], limit: 32 }) as
    { files: Array<{ path: string; content: string | null }>; remaining: string[] };
  assert.deepEqual(result, {
    files: [
      { path: "scripts/a.py", content: Buffer.from("assert True").toString("base64") },
      { path: "scripts/ghost.py", content: null },
    ],
    remaining: [],
  });
  const many = Array.from({ length: 65 }, (_, index) => `scripts/f${index}.py`);
  const paged = await execution.handle("execution.read_many", { paths: many, limit: 32 }) as
    { files: unknown[]; remaining: string[] };
  assert.equal(paged.files.length, 64);
  assert.deepEqual(paged.remaining, ["scripts/f64.py"]);
  await assert.rejects(execution.handle("execution.read_many",
    { paths: ["scripts/a.py"], limit: 0 }), /invalid file size limit/);
});

test("reads serve up to the raised ceiling and page batches past the byte budget", async () => {
  const big = Buffer.alloc(300 * 1024, "x");
  const files = new Map<string, Buffer>([["/session/big.bin", big]]);
  for (let index = 0; index < 5; index += 1) {
    files.set(`/session/scripts/f${index}.py`, Buffer.alloc(200 * 1024, "y"));
  }
  const execution = new WorkflowExecution({
    cwd: "/session",
    resolvePath: async (path) => `/session/${path}`,
    readFile: async (path) => files.get(path)!,
    writeFile: async () => {},
    operations: { async exec() { assert.fail("reads must not use terminal"); } },
  }, peer());
  const single = await execution.handle("execution.read",
    { path: "big.bin", limit: MAX_READ_BYTES }) as { content: string };
  assert.equal(Buffer.from(single.content, "base64").length, big.length);
  await assert.rejects(execution.handle("execution.read",
    { path: "big.bin", limit: MAX_READ_BYTES + 1 }), /invalid file size limit/);
  const batch = await execution.handle("execution.read_many",
    { paths: Array.from({ length: 5 }, (_, index) => `scripts/f${index}.py`),
      limit: MAX_READ_BYTES }) as
    { files: Array<{ path: string; content: string | null }>; remaining: string[] };
  assert.equal(batch.files.length, 2);
  assert.ok(batch.files.every((file) => file.content !== null));
  assert.deepEqual(batch.remaining, ["scripts/f2.py", "scripts/f3.py", "scripts/f4.py"]);
});


test("capability discovery reports stages, revisions and protection independently", async () => {
  for (const readFile of [undefined, async () => Buffer.from("script")]) {
    const execution = new WorkflowExecution({
      cwd: "/session", readFile, writeFile: async () => {},
      resolvePath: async (path) => path,
      operations: { async exec() { assert.fail("capability discovery must not execute"); } },
    }, peer());
    assert.deepEqual(await execution.handle("execution.capabilities", {}), {
      agent_task: true, human: true, revision: !!readFile, file_read: !!readFile,
      protected_verification: false, model_judge: false, tool: true,
    });
  }
});

test("a graph's fixed tool call runs through the adapter tool port, never the terminal", async () => {
  const calls: Array<[string, JsonObject]> = [];
  const answer: JsonObject = {
    is_error: false,
    content: [{ type: "text", text: "submitted" }],
    details: { task_id: "task-1" },
  };
  const peer = {
    request: async (method: string, params: JsonObject): Promise<JsonValue> => {
      calls.push([method, params]);
      return answer;
    },
  } as unknown as JsonlRpcPeer;
  const execution = new WorkflowExecution({
    cwd: "/session", writeFile: async () => {},
    resolvePath: async (path) => path,
    operations: { async exec() { assert.fail("tool calls must not use the terminal"); } },
  }, peer);
  assert.deepEqual(await execution.handle("execution.tool", {
    name: "df_submit_pipeline", arguments: { dataset: "/storage/a.csv" },
    tool_call_id: "genome-1", origin_run_id: "graph-run-1",
  }), answer);
  assert.deepEqual(calls, [["tool.execute", {
    tool_name: "df_submit_pipeline", arguments: { dataset: "/storage/a.csv" },
    tool_call_id: "genome-1", origin_run_id: "graph-run-1",
  }]]);
  // The tool call id only exists so the adapter can correlate signals; a caller
  // without one must not send an empty string.
  await execution.handle("execution.tool", { name: "df_get_task", arguments: {} });
  assert.deepEqual(calls[1], ["tool.execute", { tool_name: "df_get_task", arguments: {} }]);
  await assert.rejects(execution.handle("execution.tool", { name: "df_get_task" }),
    /tool name and arguments are required/);
  await assert.rejects(execution.handle("execution.tool", { name: "", arguments: [] }),
    /tool name and arguments are required/);

  assert.deepEqual(
    await execution.handle("execution.cancel_tool", { tool_call_id: "genome-1" }),
    answer,
  );
  assert.deepEqual(calls.at(-1), ["tool.cancel", { tool_call_id: "genome-1" }]);
  await assert.rejects(
    execution.handle("execution.cancel_tool", { tool_call_id: " " }),
    /tool call id is required/,
  );
});
