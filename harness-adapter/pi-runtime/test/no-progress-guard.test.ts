import assert from "node:assert/strict";
import test from "node:test";
import type {
  ExtensionAPI,
  ToolCallEvent,
  ToolCallEventResult,
  ToolResultEvent,
  ToolResultEventResult,
} from "@earendil-works/pi-coding-agent";
import { createNoProgressGuardExtension } from "../src/no-progress-guard.js";

type CallHandler = (event: ToolCallEvent) => ToolCallEventResult | undefined;
type ResultHandler = (event: ToolResultEvent) => ToolResultEventResult | undefined;

function guardHandlers(): { call: CallHandler; result: ResultHandler } {
  const handlers: Record<string, unknown> = {};
  const pi = {
    on(event: string, handler: unknown) {
      handlers[event] = handler;
    },
  } as unknown as ExtensionAPI;
  const extension = createNoProgressGuardExtension();
  const factory = typeof extension === "function" ? extension : extension.factory;
  factory(pi);
  return {
    call: handlers.tool_call as CallHandler,
    result: handlers.tool_result as ResultHandler,
  };
}

const FIND = { command: "find storage/ -name workflow.py" };

function callEvent(
  toolCallId: string,
  input: Record<string, unknown> = FIND,
  toolName = "terminal",
): ToolCallEvent {
  return { type: "tool_call", toolCallId, toolName, input };
}

function resultEvent(
  toolCallId: string,
  text: string,
  input: Record<string, unknown> = FIND,
  toolName = "terminal",
): ToolResultEvent {
  return {
    type: "tool_result",
    toolCallId,
    toolName,
    input,
    content: [{ type: "text", text }],
    isError: false,
    details: undefined,
  };
}

function textOf(result: ToolResultEventResult | undefined): string {
  return (result?.content ?? [])
    .map((block) => (block.type === "text" ? block.text : ""))
    .join("");
}

test("an identical call with an identical result is warned, then blocked", () => {
  const guard = guardHandlers();
  assert.equal(guard.call(callEvent("c1")), undefined);
  assert.equal(guard.result(resultEvent("c1", "")), undefined);

  assert.equal(guard.call(callEvent("c2")), undefined);
  assert.match(textOf(guard.result(resultEvent("c2", ""))), /no new information/);

  assert.equal(guard.call(callEvent("c3")), undefined);
  assert.equal(guard.result(resultEvent("c3", "")), undefined);

  const blocked = guard.call(callEvent("c4"));
  assert.equal(blocked?.block, true);
  assert.match(blocked?.reason ?? "", /already returned the same result/);
});

test("a call that keeps producing new results is never blocked", () => {
  const guard = guardHandlers();
  for (let index = 1; index <= 6; index += 1) {
    const toolCallId = `c${index}`;
    assert.equal(guard.call(callEvent(toolCallId)), undefined);
    assert.equal(
      guard.result(resultEvent(toolCallId, `processed ${index}/6`)),
      undefined,
    );
  }
});

test("polling tools keep their repeat budget", () => {
  const guard = guardHandlers();
  const input = { output_dir: "public_data/runs/1" };
  for (let index = 1; index <= 6; index += 1) {
    const toolCallId = `c${index}`;
    assert.equal(guard.call(callEvent(toolCallId, input, "df_check_progress")), undefined);
    assert.equal(
      guard.result(resultEvent(toolCallId, "running 30%", input, "df_check_progress")),
      undefined,
    );
  }
});

test("argument key order does not create a new call", () => {
  const guard = guardHandlers();
  assert.equal(guard.call(callEvent("c1", { command: "ls", cwd: "." })), undefined);
  assert.equal(guard.result(resultEvent("c1", "", { command: "ls", cwd: "." })), undefined);
  assert.equal(guard.call(callEvent("c2", { cwd: ".", command: "ls" })), undefined);
  assert.match(
    textOf(guard.result(resultEvent("c2", "", { cwd: ".", command: "ls" }))),
    /no new information/,
  );
});

test("a blocked call does not count as another observation", () => {
  const guard = guardHandlers();
  for (let index = 1; index <= 3; index += 1) {
    const toolCallId = `c${index}`;
    guard.call(callEvent(toolCallId));
    guard.result(resultEvent(toolCallId, ""));
  }
  assert.equal(guard.call(callEvent("c4"))?.block, true);
  assert.equal(guard.result(resultEvent("c4", "")), undefined);
  assert.equal(guard.call(callEvent("c5"))?.block, true);
});
