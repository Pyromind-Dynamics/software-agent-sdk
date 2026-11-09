import assert from "node:assert/strict";
import test from "node:test";
import type {
  ExtensionAPI,
  ToolResultEvent,
  ToolResultEventResult,
} from "@earendil-works/pi-coding-agent";
import type { ObservationRecorder } from "../src/observability/recorder.js";
import { createToolResultExtension } from "../src/tool-results.js";

type ResultHandler = (event: ToolResultEvent) => ToolResultEventResult | undefined;
type HookRecord = { name: string; event: string; input: Record<string, unknown> };

function handlers(): {
  result: ResultHandler;
  records: HookRecord[];
} {
  const records: HookRecord[] = [];
  const recorder = {
    hook(
      name: string,
      event: string,
      input: Record<string, unknown>,
      _callId: string,
      invoke: () => unknown,
    ) {
      records.push({ name, event, input });
      return invoke();
    },
  } as unknown as ObservationRecorder;
  const registered: Record<string, unknown> = {};
  const pi = {
    on(event: string, handler: unknown) {
      registered[event] = handler;
    },
  } as unknown as ExtensionAPI;
  createToolResultExtension(recorder)(pi);
  return {
    result: registered.tool_result as ResultHandler,
    records,
  };
}

test("failed tool categories are marked as errors and recorded", () => {
  const { result, records } = handlers();
  const output = result({
    type: "tool_result",
    toolCallId: "call-1",
    toolName: "terminal",
    input: {},
    content: [{ type: "text", text: "failed" }],
    isError: false,
    details: { resultCategory: "failure" },
  });

  assert.deepEqual(output, { isError: true });
  assert.equal(records.length, 1);
  assert.equal(records[0].input.decision, "set_is_error_true");
  assert.equal(records[0].input.is_error_before, false);
});

test("successful tool results are left unchanged and recorded as no-op", () => {
  const { result, records } = handlers();
  const output = result({
    type: "tool_result",
    toolCallId: "call-2",
    toolName: "terminal",
    input: {},
    content: [{ type: "text", text: "ok" }],
    isError: false,
    details: { resultCategory: "success" },
  });

  assert.equal(output, undefined);
  assert.equal(records.length, 1);
  assert.equal(records[0].input.decision, "leave_unchanged");
});
