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
