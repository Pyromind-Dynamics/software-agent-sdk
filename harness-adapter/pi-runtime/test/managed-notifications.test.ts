import assert from "node:assert/strict";
import test from "node:test";
import type {
  AgentEndEvent,
  ExtensionAPI,
  ExtensionContext,
  ExtensionHandler,
  SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { nativeSubagentApi } from "../src/pi-session.js";
import { notificationSettled } from "../src/agent-runtime.js";

test("interrupted native notifications are resumable while settled receipts deduplicate", () => {
  const entry = (status?: string) => ({type: "custom", customType: "pyromind.notification.finished", data: {runId: "notice", status}} as unknown as SessionEntry);
  assert.equal(notificationSettled([], "notice"), undefined);
  assert.equal(notificationSettled([entry("cancelled")], "notice"), undefined);
  assert.equal(notificationSettled([entry("completed")], "notice"), "completed");
  assert.equal(notificationSettled([entry("failed")], "notice"), "failed");
  assert.equal(notificationSettled([entry()], "notice"), undefined);
  assert.equal(notificationSettled([entry("completed")], "other"), undefined);
});

test("native completion remains in context while product execution is paused", () => {
  const notices: unknown[] = [];
  const wakes: unknown[] = [];
  let paused = true;
  const api = nativeSubagentApi({
    sendMessage: (...args: unknown[]) => notices.push(args),
    sendUserMessage: (...args: unknown[]) => wakes.push(args),
  } as unknown as ExtensionAPI, { automaticTurnsSuspended: () => paused });
  const notice = {
    customType: "subagent-notify",
    content: "[output](https://storage.invalid/output?X-Amz-Signature=discard)",
    display: false,
  };
  api.sendMessage(notice, { triggerTurn: true });
  api.sendUserMessage("wake");
  assert.deepEqual(notices, [[{
    ...notice, content: "[output](https://storage.invalid/output)",
  }, { triggerTurn: false }]]);
  assert.deepEqual(wakes, []);
  paused = false;
  api.sendMessage(notice, { triggerTurn: true });
  api.sendUserMessage("wake", { deliverAs: "steer" });
  assert.deepEqual(wakes, [["wake", { deliverAs: "steer" }]]);
  assert.deepEqual(notices[1], [{ ...notice, content: "[output](https://storage.invalid/output)" }, { triggerTurn: true }]);
});

test("the persistent native host uses interactive settlement without changing other hook contexts", async () => {
  const handlers = new Map<string, ExtensionHandler<AgentEndEvent>>();
  const api = nativeSubagentApi({
    on: (event: string, handler: ExtensionHandler<AgentEndEvent>) => {
      handlers.set(event, handler);
      return () => handlers.delete(event);
    },
  } as unknown as ExtensionAPI, {});
  const signal = new AbortController().signal;
  const original = { hasUI: false, signal } as ExtensionContext;
  let ended: ExtensionContext | undefined;
  let started: ExtensionContext | undefined;
  api.on("agent_end", (_event, context) => { ended = context; });
  api.on("agent_start", (_event, context) => { started = context; });
  const event = { type: "agent_end", messages: [] } as AgentEndEvent;
  await handlers.get("agent_end")!(event, original);
  await handlers.get("agent_start")!(event, original);
  assert.equal(ended?.hasUI, true);
  assert.equal(ended?.signal, signal);
  assert.equal(started, original);
  assert.equal(original.hasUI, false);
});
