import assert from "node:assert/strict";
import test from "node:test";
import type { AgentSessionEvent } from "@earendil-works/pi-coding-agent";
import { PiEventNormalizer, sanitizeJson } from "../src/pi-events.js";

test("event payloads redact credentials and normalize unsupported details", () => {
  assert.deepEqual(sanitizeJson({ cookie: "secret", nested: { api_key: "key" } }), { cookie: "[REDACTED]", nested: { api_key: "[REDACTED]" } });
});

test("agent_end is lifecycle-only and never emits a terminal event", () => {
  const normalizer = new PiEventNormalizer("s1", "r1");
  const event = { type: "agent_end", messages: [] } as unknown as AgentSessionEvent;
  assert.deepEqual(normalizer.translate(event), []);
});

test("failed tools retain structured error codes, with fallback for legacy or invalid details", () => {
  const normalizer = new PiEventNormalizer("s1", "r1");
  for (const errorCode of ["ENUM_VALUE_INVALID", "", "  ", 0, null, undefined]) {
    const details = { error_code: errorCode, retryable: false, errors: [{ field: "gpu_product" }] };
    const event = normalizer.translate({
      type: "tool_execution_end", toolCallId: "call-1", toolName: "business_tool",
      result: { content: [{ type: "text", text: "unsupported GPU" }], details }, isError: true,
    } as unknown as AgentSessionEvent)[0]!;
    assert.equal(event.kind, "tool.failed");
    assert.equal(event.payload.error_code, errorCode === "ENUM_VALUE_INVALID" ? errorCode : "tool_execution_failed");
    assert.equal((event.payload.details as Record<string, unknown>).retryable, false);
    assert.deepEqual((event.payload.details as Record<string, unknown>).errors, details.errors);
  }
  const legacy = normalizer.translate({
    type: "tool_execution_end", toolCallId: "old-call", toolName: "read",
    result: { content: [], details: undefined }, isError: true,
  } as unknown as AgentSessionEvent)[0]!;
  assert.equal(legacy.payload.error_code, "tool_execution_failed");
  assert.equal(legacy.payload.details, null);
});

test("tool completion reports generic arguments without workflow semantics", () => {
  const normalizer = new PiEventNormalizer("s1", "r1");
  normalizer.translate({
    type: "tool_execution_start",
    toolCallId: "call-1",
    toolName: "write",
    args: { path: "public_data/workflow_canvas/workflow.py", content: "dsl" },
  } as unknown as AgentSessionEvent);
  const events = normalizer.translate({
    type: "tool_execution_end",
    toolCallId: "call-1",
    toolName: "write",
    result: { content: [], details: undefined },
    isError: false,
  } as unknown as AgentSessionEvent);

  assert.equal(events.length, 1);
  assert.equal(events[0]!.kind, "tool.completed");
  assert.deepEqual(events[0]!.payload.arguments, {
    path: "public_data/workflow_canvas/workflow.py",
    content: "dsl",
  });
  assert.equal("resource_type" in events[0]!.payload, false);
});

test("steered identical messages keep command identity separate from callback run", () => {
  const normalizer = new PiEventNormalizer("s", "callback:graph:unique", ["command-1", "command-2"]);
  const ids = [];
  for (const commandId of ["command-1", "command-2"]) {
    const message = { role: "user", content: "same text", timestamp: 1 };
    const started = normalizer.translate({ type: "message_start", message } as AgentSessionEvent)[0]!;
    const completed = normalizer.translate({ type: "message_end", message } as AgentSessionEvent)[0]!;
    assert.equal(started.payload.command_id, commandId);
    assert.equal(completed.payload.command_id, commandId);
    assert.equal(completed.runId, "callback:graph:unique");
    ids.push(completed.payload.message_id);
  }
  assert.notEqual(ids[0], ids[1]);
});
