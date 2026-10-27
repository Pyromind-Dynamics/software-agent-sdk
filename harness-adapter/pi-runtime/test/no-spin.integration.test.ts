import assert from "node:assert/strict";
import { readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import test from "node:test";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { PiAgentRuntime } from "../src/agent-runtime.js";
import type { JsonObject } from "../src/protocol.js";
import { eventually, fixture, frame, repeat, text } from "./fixtures/no-spin.js";

for (const field of ["content", "reasoning_content"]) {
  test(`repetition in ${field} fails once, cancels transport, cleans context and resumes`, { timeout: 30_000 }, async (t) => {
    t.mock.method(process, "exit", () => undefined);
    let count = 0, closed = false;
    const f = await fixture(t, (_body, response) => {
      if (++count === 1) {
        response.writeHead(200, { "content-type": "text/event-stream" });
        frame(response, { role: "assistant", tool_calls: [{ index: 0, id: "evidence-1", type: "function",
          function: { name: "evidence", arguments: "{}" } }] });
        frame(response, {}, "tool_calls");
        response.end("data: [DONE]\n\n");
      } else if (count === 2) repeat(response, field, () => { closed = true; });
      else text(response, "Ready to continue.");
    });
    let runtime = new PiAgentRuntime(f.peer);
    const close = async () => {
      await runtime.handle("close", {});
      await new Promise<void>((resolve) => setImmediate(resolve));
    };
    t.after(async () => { await close(); await rm(f.root, { recursive: true, force: true }); });
    await runtime.handle("start", f.config);
    await runtime.handle("prompt", { run_id: "repeating", content: [{ type: "text", text: "Inspect evidence." }] });
    await eventually(() => f.events.some((e) => e.kind === "run.finished"));
    const finished = f.events.filter((e) => e.kind === "run.finished");
    assert.equal(finished.length, 1);
    assert.deepEqual(finished[0]!.payload.outcome, {
      status: "failed", stop_reason: "error", error_code: "model_repetition_detected",
      message: "模型持续重复生成，已停止。请发送新消息后继续。",
    });
    await eventually(() => closed);
    assert.equal(f.events.filter((e) => e.kind === "execution.policy.updated").length, 1);
    await runtime.handle("notify", { run_id: "notification", content: "Pending external result." });
    assert.equal((await runtime.handle("context.append", { content: "Additional evidence", trigger_turn: true }) as JsonObject).trigger_turn, false);
    await new Promise((resolve) => setTimeout(resolve, 100));
    assert.equal(count, 2, "neither plugin follow-up nor external notifications may restart the run");
    const log = await readFile(join(f.root, "pi/session.jsonl"), "utf8");
    const entries = log.trim().split("\n").map((line) => JSON.parse(line));
    assert.ok(entries.some((entry) => entry.type === "message" && JSON.stringify(entry.message).includes("Emit. Emit.")));
    assert.ok(entries.some((entry) => entry.type === "context_edit"));
    const projected = SessionManager.open(join(f.root, "pi/session.jsonl")).buildSessionContext().messages;
    assert.doesNotMatch(JSON.stringify(projected), /Emit\. Emit\./);
    assert.match(JSON.stringify(projected), /PRESERVED_TOOL_EVIDENCE/);
    assert.ok(projected.some((message) => message.role === "assistant" && message.content.some((block) => block.type === "toolCall")));
    await close();
    runtime = new PiAgentRuntime(f.peer);
    await runtime.handle("start", { ...f.config, execution_policy: { automatic_turns_suspended: true } });
    await new Promise((resolve) => setTimeout(resolve, 100));
    assert.equal(count, 2, "restart must honor the persisted product policy");
    await runtime.handle("prompt", { run_id: "manual", content: [{ type: "text", text: "Continue." }] });
    await eventually(() => f.events.some((e) => e.kind === "run.finished" && e.runId === "manual"));
    assert.equal((f.events.find((e) => e.kind === "run.finished" && e.runId === "manual")!.payload.outcome as JsonObject).status, "completed");
    assert.doesNotMatch(JSON.stringify(f.requests[2]), /Emit\. Emit\./);
    assert.match(JSON.stringify(f.requests[2]), /PRESERVED_TOOL_EVIDENCE/);
    assert.equal(f.events.filter((e) => e.kind === "run.finished" && e.runId === "repeating").length, 1);
    await eventually(() => f.events.filter((e) => e.kind === "run.finished").length >= 3);
  });
}

test("healthy output and explicit user cancellation retain their outcomes", { timeout: 30_000 }, async (t) => {
  t.mock.method(process, "exit", () => undefined);
  const healthy = Array.from({ length: 100 }, (_, i) => `记录 ${i}: value=${i * 17}; {"index":${i}}\n`).join("") +
    "```ts\nconst result = rows.map(row => row.id);\n```\n" + "─".repeat(100) + ".".repeat(100);
  let cancelReady = false;
  const f = await fixture(t, (body, response) => {
    if (JSON.stringify(body.messages).includes("CANCEL_ME")) {
      response.writeHead(200, { "content-type": "text/event-stream" });
      frame(response, { role: "assistant", content: "Working on the request." });
      cancelReady = true;
    } else text(response, healthy);
  });
  const runtime = new PiAgentRuntime(f.peer);
  t.after(async () => {
    await runtime.handle("close", {}); await new Promise((resolve) => setImmediate(resolve));
    await rm(f.root, { recursive: true, force: true });
  });
  await runtime.handle("start", f.config);
  await runtime.handle("prompt", { run_id: "healthy", content: [{ type: "text", text: "Report." }] });
  await eventually(() => f.events.some((e) => e.kind === "run.finished"));
  assert.equal((f.events.find((e) => e.kind === "run.finished" && e.runId === "healthy")!.payload.outcome as JsonObject).status, "completed");
  await runtime.handle("prompt", { run_id: "cancel", content: [{ type: "text", text: "CANCEL_ME" }] });
  await eventually(() => cancelReady);
  await runtime.handle("cancel", {});
  await eventually(() => f.events.some((e) => e.kind === "run.finished" && e.runId === "cancel"));
  assert.equal((f.events.find((e) => e.kind === "run.finished" && e.runId === "cancel")!.payload.outcome as JsonObject).status, "cancelled");
  assert.equal(f.events.some((e) => e.kind === "execution.policy.updated"), false);
});

test("a second manual attempt is protected without an upstream reminder cooldown", { timeout: 10_000 }, async (t) => {
  t.mock.method(process, "exit", () => undefined);
  const f = await fixture(t, (_body, response) => repeat(response, "content", () => {}));
  const runtime = new PiAgentRuntime(f.peer);
  t.after(async () => {
    await runtime.handle("close", {}); await new Promise((resolve) => setImmediate(resolve));
    await rm(f.root, { recursive: true, force: true });
  });
  await runtime.handle("start", f.config);
  const started = Date.now();
  for (const runId of ["first", "manual-retry"]) {
    await runtime.handle("prompt", { run_id: runId, content: [{ type: "text", text: "Continue." }] });
    if (runId === "first") {
      await eventually(() => f.requests.length === 1);
      await runtime.handle("steer", { run_id: "queued", content: [{ type: "text", text: "Pending instruction." }] });
    }
    await eventually(() => f.events.some((e) => e.kind === "run.finished" && e.runId === runId));
    assert.equal((f.events.find((e) => e.kind === "run.finished" && e.runId === runId)!.payload.outcome as JsonObject).error_code, "model_repetition_detected");
  }
  assert.ok(Date.now() - started < 3000);
  assert.equal(f.requests.length, 2);
  assert.equal(f.events.find((e) => e.kind === "message.started" && e.runId === "manual-retry" && e.payload.role === "user")!.payload.command_id, "manual-retry");
});

test("compiled runner loads upstream TS and keeps stdout as JSONL", { timeout: 30_000 }, async (t) => {
  const f = await fixture(t, (_body, response) => repeat(response, "content", () => {}));
  const child = spawn(process.execPath, [new URL("../dist/index.js", import.meta.url).pathname], { stdio: ["pipe", "pipe", "pipe"] });
  t.after(() => child.kill());
  const frames: any[] = [];
  let stderr = "", invalid = "";
  child.stderr.on("data", (data) => { stderr += data; });
  const lines = createInterface({ input: child.stdout });
  lines.on("line", (line) => { try { frames.push(JSON.parse(line)); } catch { invalid += line; } });
  const send = (id: string, method: string, params: JsonObject) => child.stdin.write(JSON.stringify({
    protocolVersion: 2, type: "request", requestId: id, method, params,
  }) + "\n");
  send("start", "start", f.config);
  await eventually(() => frames.some((frame) => frame.requestId === "start"));
  assert.equal(frames.find((frame) => frame.requestId === "start").error, undefined, stderr);
  send("prompt", "prompt", { run_id: "compiled", content: [{ type: "text", text: "Answer." }] });
  await eventually(() => frames.some((frame) => frame.kind === "run.finished"));
  assert.equal(frames.find((frame) => frame.kind === "run.finished").payload.outcome.error_code, "model_repetition_detected", stderr);
  assert.equal(invalid, "");
  assert.equal(f.requests.length, 1);
  send("close", "close", {});
  await new Promise<void>((resolve) => child.once("exit", () => resolve()));
  await rm(f.root, { recursive: true, force: true });
});
