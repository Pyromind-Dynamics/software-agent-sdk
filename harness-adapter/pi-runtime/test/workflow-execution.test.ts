import assert from "node:assert/strict";
import test from "node:test";
import { PassThrough } from "node:stream";
import { WorkflowExecution } from "../src/workflow-execution.js";
import { JsonlRpcPeer } from "../src/rpc-peer.js";

function peer() { return new JsonlRpcPeer(new PassThrough(), new PassThrough()); }

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
