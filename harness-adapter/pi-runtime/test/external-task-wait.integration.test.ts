import assert from "node:assert/strict";
import { rm } from "node:fs/promises";
import type { ServerResponse } from "node:http";
import test, { type TestContext } from "node:test";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { PiAgentRuntime } from "../src/agent-runtime.js";
import type { JsonObject } from "../src/protocol.js";
import { eventually, fixture, frame, text } from "./fixtures/no-spin.js";

function commands(response: ServerResponse, ...values: string[]): void {
  response.writeHead(200, { "content-type": "text/event-stream" });
  frame(response, { role: "assistant", tool_calls: values.map((command, index) => ({
    index, id: `terminal-${index}`, type: "function",
    function: { name: "terminal", arguments: JSON.stringify({ command }) },
  })) });
  frame(response, {}, "tool_calls");
  response.end("data: [DONE]\n\n");
}

async function setup(t: TestContext, respond: (count: number, response: ServerResponse) => void, enabled = true) {
  t.mock.method(process, "exit", () => undefined);
  let count = 0;
  const f = await fixture(t, (_body, response) => respond(++count, response));
  const state = { pending: false, owner: "main", queries: [] as string[], executed: [] as string[] };
  t.mock.method(f.peer, "request", async (method: string, params: JsonObject) => {
    assert.equal(method, "external_task.waiting");
    const runId = String(params.run_id);
    state.queries.push(runId);
    return { task_ids: state.pending && runId === state.owner ? ["task-1"] : [] };
  });
  const runtime = new PiAgentRuntime(f.peer);
  t.after(async () => {
    await runtime.handle("close", {});
    await new Promise<void>(resolve => setImmediate(resolve));
    await rm(f.root, { recursive: true, force: true });
  });
  await runtime.handle("start", {
    ...f.config, await_external_tasks: enabled, tool_allowlist: ["terminal"],
  });
  const session = (runtime as unknown as { session: AgentSession }).session;
  const terminal = session.agent.state.tools.find(tool => tool.name === "terminal")!;
  t.mock.method(terminal, "execute", async (_id: string, args: { command: string }) => {
    state.executed.push(args.command);
    if (args.command === "submit-sdk") state.pending = true;
    return { content: [{ type: "text" as const, text: "SDK task receipt persisted" }], details: {} };
  });
  const run = async (runId: string) => {
    await runtime.handle("prompt", { run_id: runId, content: [{ type: "text", text: "Continue." }] });
    await eventually(() => f.events.some(e => e.kind === "run.finished" && e.runId === runId));
  };
  return { ...f, runtime, session, state, run };
}

test("native wait ends after SDK submission, blocks a same-batch sleep, and callback resumes", { timeout: 15_000 }, async t => {
  const f = await setup(t, (count, response) => {
    if (count === 1) commands(response, "submit-sdk", "sleep 240");
    else if (count === 2) commands(response, "inspect-result");
    else text(response, "Result verified.");
  });
  assert.equal(f.session.agent.state.tools.find(tool => tool.name === "terminal")!.executionMode, "sequential");
  await f.run("main");
  assert.equal(f.requests.length, 1);
  assert.deepEqual(f.state.executed, ["submit-sdk"]);
  assert.equal((f.events.find(e => e.kind === "run.finished")!.payload.outcome as JsonObject).status, "completed");
  assert.equal(f.events.some(e => e.kind === "execution.policy.updated"), false);
  assert.equal((f.runtime as unknown as { executionSuspended: boolean }).executionSuspended, false);
  assert.equal(f.session.messages.filter(message => message.role === "toolResult").length, 2);
  f.state.pending = false;
  await f.runtime.handle("notify", {
    run_id: "task-1:succeeded", content: "Platform task succeeded.",
    details: { task_id: "task-1", run_id: "main" },
  });
  await eventually(() => f.events.filter(e => e.kind === "run.finished").length === 2);
  assert.equal(f.requests.length, 3);
  assert.deepEqual(f.state.executed, ["submit-sdk", "inspect-result"]);
});

test("a terminal callback queued before the submit tool returns owns the continuation", { timeout: 15_000 }, async t => {
  const f = await setup(t, (count, response) => {
    if (count === 1) commands(response, "submit-sdk");
    else text(response, "Callback handled.");
  });
  const terminal = f.session.agent.state.tools.find(tool => tool.name === "terminal")!;
  t.mock.method(terminal, "execute", async () => {
    await f.runtime.handle("notify", {
      run_id: "early-terminal", content: "Already succeeded.",
      details: { task_id: "task-1", run_id: "main" },
    });
    return { content: [{ type: "text" as const, text: "Task receipt and registration complete" }], details: {} };
  });
  await f.run("main");
  await eventually(() => f.events.filter(e => e.kind === "run.finished").length === 2);
  assert.equal(f.requests.length, 2);
  assert.match(JSON.stringify(f.requests[1]), /Already succeeded/);
  assert.ok(f.events.filter(e => e.kind === "run.finished")
    .every(e => (e.payload.outcome as JsonObject).status === "completed"));
});

test("a project task callback leaves the origin free to deliver another node", { timeout: 15_000 }, async t => {
  const f = await setup(t, (count, response) => {
    if (count === 1) commands(response, "submit-train");
    else if (count === 2) commands(response, "deliver-init");
    else text(response, count === 3 ? "Init report delivered." : "Training callback handled.");
  });
  const terminal = f.session.agent.state.tools.find(tool => tool.name === "terminal")!;
  t.mock.method(terminal, "execute", async (_id: string, args: { command: string }) => {
    f.state.executed.push(args.command);
    if (args.command === "submit-train") {
      await f.runtime.handle("notify", {
        run_id: "train-terminal", content: "Training succeeded.",
        details: { task_id: "training-1", run_id: "main", blocks_origin_turn: false },
      });
    }
    return { content: [{ type: "text" as const, text: "Operation complete" }], details: {} };
  });
  await f.run("main");
  await eventually(() => f.events.filter(e => e.kind === "run.finished").length === 2);
  assert.deepEqual(f.state.executed, ["submit-train", "deliver-init"]);
  assert.match(JSON.stringify(f.requests[3]), /Training succeeded/);
});

test("queued user progress requests bypass the wait without losing the user turn", { timeout: 15_000 }, async t => {
  const f = await setup(t, (count, response) => {
    if (count === 1) commands(response, "submit-sdk");
    else if (count === 2) commands(response, "inspect-progress");
    else text(response, "Still running.");
  });
  const terminal = f.session.agent.state.tools.find(tool => tool.name === "terminal")!;
  t.mock.method(terminal, "execute", async (_id: string, args: { command: string }) => {
    f.state.executed.push(args.command);
    if (args.command === "submit-sdk") {
      f.state.pending = true;
      await f.runtime.handle("steer", { run_id: "user-progress", content: [{ type: "text", text: "Show progress." }] });
    }
    return { content: [{ type: "text" as const, text: "Progress available" }], details: {} };
  });
  await f.run("main");
  assert.deepEqual(f.state.executed, ["submit-sdk", "inspect-progress"]);
  assert.equal(f.requests.length, 3);
  assert.match(JSON.stringify(f.requests[1]), /Show progress/);
});

test("without the opt-in native tool continuations never query external waits", { timeout: 15_000 }, async t => {
  const f = await setup(t, (count, response) => {
    if (count === 1) commands(response, "submit-sdk");
    else text(response, "Submitted.");
  }, false);
  await f.run("main");
  assert.equal(f.requests.length, 2);
  assert.deepEqual(f.state.queries, []);
});

test("the wait boundary preserves native policy blocks and finish-turn decisions", { timeout: 15_000 }, async t => {
  const f = await setup(t, (count, response) => {
    if (count === 1) commands(response, "blocked-by-policy");
    else text(response, "Existing policy preserved.");
  }, false);
  const before = f.session.agent.beforeToolCall;
  const finish = f.session.agent.finishTurn;
  let beforeCalls = 0, finishCalls = 0;
  f.session.agent.beforeToolCall = async (call, signal) => {
    await before?.(call, signal);
    beforeCalls++;
    return { block: true, reason: "Existing policy", terminate: true };
  };
  f.session.agent.finishTurn = async (turn, signal) => {
    await finish?.(turn, signal);
    return ++finishCalls === 1 ? { action: "continue" } : undefined;
  };
  (f.runtime as unknown as { installExternalTaskWaitBoundary(session: AgentSession): void })
    .installExternalTaskWaitBoundary(f.session);
  await f.run("main");
  assert.equal(beforeCalls, 1);
  assert.equal(finishCalls, 2);
  assert.equal(f.state.queries.length, 2);
  assert.deepEqual(f.state.executed, []);
  assert.match(JSON.stringify(f.session.messages), /Existing policy/);
});
