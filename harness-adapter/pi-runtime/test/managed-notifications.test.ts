import assert from "node:assert/strict";
import test from "node:test";
import type { ExtensionAPI, SessionEntry } from "@earendil-works/pi-coding-agent";
import { managedSubagentApi } from "../src/pi-session.js";
import { notificationSettled } from "../src/agent-runtime.js";

test("managed plugin notifications cannot trigger or append duplicate parent work", () => {
  const calls: any[] = [];
  const api = managedSubagentApi({sendMessage: (...args: any[]) => calls.push(args), sendUserMessage: (...args: any[]) => calls.push(args)} as unknown as ExtensionAPI);
  api.sendMessage({customType: "subagent-notify", content: "raw child data", display: true}, {triggerTurn: true});
  api.sendUserMessage("native wake");
  assert.equal(calls.length, 0);
  api.sendMessage({customType: "other-plugin-message", content: "notice", display: false}, {triggerTurn: true});
  assert.equal(calls.length, 1);
  assert.equal(calls[0][1].triggerTurn, false);
});

test("interrupted native notifications are resumable while settled receipts deduplicate", () => {
  const entry = (status?: string) => ({type: "custom", customType: "pyromind.notification.finished", data: {runId: "notice", status}} as unknown as SessionEntry);
  assert.equal(notificationSettled([], "notice"), undefined);
  assert.equal(notificationSettled([entry("cancelled")], "notice"), undefined);
  assert.equal(notificationSettled([entry("completed")], "notice"), "completed");
  assert.equal(notificationSettled([entry("failed")], "notice"), "failed");
  assert.equal(notificationSettled([entry()], "notice"), "completed");
  assert.equal(notificationSettled([entry("completed")], "other"), undefined);
});
