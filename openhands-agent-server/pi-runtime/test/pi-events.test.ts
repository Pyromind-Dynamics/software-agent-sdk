import assert from "node:assert/strict";
import test from "node:test";
import type { AgentEvent } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Usage, UserMessage } from "@earendil-works/pi-ai";
import { PiEventNormalizer, sanitizeJson } from "../src/pi-events.ts";

const usage: Usage = {
  input: 10,
  output: 4,
  cacheRead: 3,
  cacheWrite: 0,
  totalTokens: 17,
  cost: { input: 0.1, output: 0.2, cacheRead: 0.01, cacheWrite: 0, total: 0.31 },
};

function ids(): () => string {
  let next = 0;
  return () => `id-${++next}`;
}

test("normalizes a streamed Pi run without provider-specific payloads", () => {
  const normalizer = new PiEventNormalizer("session-1", "run-1", "/workspace", ids());
  const user: UserMessage = { role: "user", content: "hello", timestamp: 1 };
  const assistant: AssistantMessage = {
    role: "assistant",
    content: [{ type: "text", text: "done" }],
    api: "openai-responses",
    provider: "openai",
    model: "gpt-5.5",
    usage,
    stopReason: "stop",
    timestamp: 2,
  };

  const events: AgentEvent[] = [
    { type: "agent_start" },
    { type: "message_start", message: user },
    { type: "message_end", message: user },
    { type: "message_start", message: assistant },
    {
      type: "message_update",
      message: assistant,
      assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "do", partial: assistant },
    },
    { type: "tool_execution_start", toolCallId: "call-1", toolName: "bash", args: { command: "pwd" } },
    {
      type: "tool_execution_update",
      toolCallId: "call-1",
      toolName: "bash",
      args: { command: "pwd" },
      partialResult: { content: [{ type: "text", text: "/workspace" }] },
    },
    {
      type: "tool_execution_end",
      toolCallId: "call-1",
      toolName: "bash",
      result: { content: [{ type: "text", text: "/workspace" }], details: { exitCode: 0 } },
      isError: false,
    },
    { type: "message_end", message: assistant },
    { type: "agent_end", messages: [user, assistant] },
  ];

  const output = events.flatMap((event) => normalizer.translate(event));
  assert.deepEqual(
    output.map((event) => event.kind),
    [
      "agent.started",
      "message.started",
      "message.completed",
      "message.started",
      "message.delta",
      "tool.started",
      "tool.progress",
      "tool.completed",
      "message.completed",
      "usage.updated",
      "agent.completed",
    ],
  );
  assert.deepEqual(output[5]?.payload.arguments, { command: "pwd" });
  assert.deepEqual(output[9]?.payload, {
    input_tokens: 10,
    output_tokens: 4,
    cached_tokens: 3,
    cost_usd: 0.31,
  });
  assert.equal(JSON.stringify(output).includes("openai-responses"), false);
  assert.equal(JSON.stringify(output).includes("gpt-5.5"), false);
});

test("maps model errors and cancellation to one terminal event", () => {
  for (const [stopReason, expected] of [
    ["error", "agent.failed"],
    ["aborted", "agent.cancelled"],
  ] as const) {
    const normalizer = new PiEventNormalizer(
      "session-1",
      `run-${stopReason}`,
      "/workspace",
      ids(),
    );
    const assistant: AssistantMessage = {
      role: "assistant",
      content: [],
      api: "openai-responses",
      provider: "openai",
      model: "gpt-5.5",
      usage,
      stopReason,
      errorMessage: stopReason,
      timestamp: 2,
    };
    normalizer.translate({ type: "message_start", message: assistant });
    const output = [
      ...normalizer.translate({ type: "message_end", message: assistant }),
      ...normalizer.translate({ type: "agent_end", messages: [assistant] }),
    ];
    assert.equal(output.at(-1)?.kind, expected);
    assert.equal(output.filter((event) => event.kind.startsWith("agent.")).length, 1);
  }
});

test("redacts credentials and bounds unknown Pi details", () => {
  assert.deepEqual(
    sanitizeJson({ api_key: "secret", nested: { Authorization: "bearer", value: 1 } }),
    { api_key: "[REDACTED]", nested: { Authorization: "[REDACTED]", value: 1 } },
  );
  assert.equal((sanitizeJson("x".repeat(70_000)) as string).length, 64 * 1024);
});

test("omits non-object tool details from product events", () => {
  const normalizer = new PiEventNormalizer("session-1", "run-1", "/workspace", ids());
  const output = normalizer.translate({
    type: "tool_execution_end",
    toolCallId: "call-1",
    toolName: "read",
    result: { content: [], details: undefined },
    isError: false,
  });

  assert.deepEqual(output[0]?.payload, {
    tool_call_id: "call-1",
    tool_name: "read",
    content: [],
  });
});

test("emits a neutral workflow resource update after a successful mutation", () => {
  const normalizer = new PiEventNormalizer("session-1", "run-1", "/workspace", ids());
  normalizer.translate({
    type: "tool_execution_start",
    toolCallId: "call-1",
    toolName: "edit",
    args: { path: "public_data/workflow_canvas/workflow.py", edits: [] },
  });

  const output = normalizer.translate({
    type: "tool_execution_end",
    toolCallId: "call-1",
    toolName: "edit",
    result: { content: [{ type: "text", text: "done" }] },
    isError: false,
  });

  assert.deepEqual(
    output.map((event) => event.kind),
    ["tool.completed", "resource.updated"],
  );
  assert.equal(output[1]?.payload.resource_type, "workflow");
  assert.equal(output[1]?.payload.version, output[1]?.eventId);
});
