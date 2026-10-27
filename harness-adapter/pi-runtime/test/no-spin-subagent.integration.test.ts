import assert from "node:assert/strict";
import { readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { createPiSession } from "../src/pi-session.js";
import { normalizePiOutcome } from "../src/pi-outcome.js";
import { eventually, fixture, repeat } from "./fixtures/no-spin.js";

test("required child extension stops a detached subagent and its host-mediated request", { timeout: 30_000 }, async (t) => {
  let closed = false;
  const f = await fixture(t, (_body, response) => repeat(response, "reasoning_content", () => { closed = true; }));
  const { session, subagents } = await createPiSession(f.config, f.peer);
  await subagents.attach(session);
  t.after(async () => {
    await subagents.stopAll(); await session.abort(); await session.waitForIdle();
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose(); await subagents.close();
    // The detached worker writes its final status just before process exit.
    await rm(f.root, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  });
  const tool = session.getToolDefinition("subagent")!;
  const result: any = await tool.execute("child-loop", { agent: "search", task: "Inspect the evidence.", async: true }, undefined, undefined, session.extensionRunner.createContext());
  assert.ok(result.details.asyncId, JSON.stringify(result));
  const statusPath = join(subagents.paths.stateRoot, "async-subagent-runs", result.details.asyncId, "status.json");
  let status: any;
  await eventually(async () => {
    try { status = JSON.parse(await readFile(statusPath, "utf8")); return ["failed", "complete", "stopped"].includes(status.state); }
    catch { return false; }
  }).catch(async (error) => {
    const log = status?.sessionFile ? await readFile(status.sessionFile, "utf8").catch(() => "") : "";
    throw new Error(`${error}; requests=${f.requests.length}; closed=${closed}; status=${JSON.stringify(status)}; log=${log.slice(-4000)}`);
  });
  assert.equal(status.state, "failed", JSON.stringify(status));
  await eventually(() => closed);
  assert.equal(f.requests.length, 1);
  const manager = SessionManager.open(status.sessionFile);
  assert.ok(manager.getBranch().some((entry) => entry.type === "message" && entry.message.role === "assistant" &&
    normalizePiOutcome([entry.message]).error_code === "model_repetition_detected"));
  assert.doesNotMatch(JSON.stringify(manager.buildSessionContext().messages), /Emit\. Emit\./);
  assert.equal(f.events.some((e) => e.kind === "execution.policy.updated"), false, "child failure does not suspend the parent");
});

