import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { setTimeout } from "node:timers/promises";
import test from "node:test";
import { promisify } from "node:util";
import { callSubagentHost, serveSubagentIpc } from "../src/subagent-ipc.js";
import { validateSubagentCall } from "../src/subagent-host.js";

test("IPC forwards results, cancels active work, and fails closed after disconnect", async () => {
  const path = `/tmp/pi-ipc-test-${randomUUID()}.sock`;
  let resolveAbort!: () => void;
  let resolveStarted!: () => void;
  const aborted = new Promise<void>(resolve => { resolveAbort = resolve; });
  const started = new Promise<void>(resolve => { resolveStarted = resolve; });
  const close = await serveSubagentIpc(path, async (request, signal, update) => {
    if (!request.wait) { update("progress"); return "ok"; }
    resolveStarted();
    await new Promise<void>(resolve => signal.addEventListener("abort", () => { resolveAbort(); resolve(); }, { once:true }));
    return null;
  });
  try {
    const updates: unknown[] = [];
    assert.equal(await callSubagentHost(path, {}, undefined, value => updates.push(value)), "ok");
    assert.deepEqual(updates,["progress"]);
    const controller = new AbortController();
    const waiting = callSubagentHost(path,{wait:true},controller.signal);
    await started;
    controller.abort();
    await assert.rejects(waiting,/cancelled/);
    await aborted;
  } finally { await close(); }
  await assert.rejects(callSubagentHost(path,{}),/unavailable|disconnected/);
});

test("a detached child stays alive until its terminal lease is ready", async (t) => {
  const path = `/tmp/pi-ipc-test-${randomUUID()}.sock`;
  const close = await serveSubagentIpc(path, async (request, signal, update) => {
    if (request.method === "tool") return request.lease;
    const disconnected = once(signal, "abort");
    await setTimeout(50);
    update("terminal-lease");
    await disconnected;
    return null;
  });
  t.after(close);
  const ipc = new URL("../src/subagent-ipc.ts", import.meta.url);
  const { stdout } = await promisify(execFile)(process.execPath, [
    "--import", "tsx", "--input-type=module", "--eval", `
      import { callSubagentHost } from ${JSON.stringify(ipc.href)};
      const path = process.argv[1];
      const lease = new Promise((resolve, reject) => {
        callSubagentHost(path, { method: "attach" }, undefined, resolve, true).catch(reject);
      });
      lease.then(async (id) => {
        console.log(await callSubagentHost(path, { method: "tool", lease: id }));
      });
    `, path,
  ], { timeout: 5000 });
  assert.equal(stdout.trim(), "terminal-lease");
});

test("native subagent calls cannot override host paths, tools, agents, or workflows", () => {
  for (const args of [
    {agent:"search",task:"x",cwd:"/"}, {agent:"search",tools:["bash"]},
    {agent:"search",sessionDir:"/tmp/other-conversation"}, {agent:"worker",task:"x"},
    {action:"create",config:{}}, {workflow:"return runs.host('x', {})"},
    {agent:"search",context:"fork"}, {action:"resume",id:"x",options:{tools:["bash"]}},
  ]) assert.ok(validateSubagentCall(args));
  assert.equal(validateSubagentCall({agent:"search",task:"x",async:true}),undefined);
  assert.equal(validateSubagentCall({action:"resume",id:"x",message:"continue"}),undefined);
});
