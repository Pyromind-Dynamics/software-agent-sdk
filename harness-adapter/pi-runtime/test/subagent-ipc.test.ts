import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import test from "node:test";
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
