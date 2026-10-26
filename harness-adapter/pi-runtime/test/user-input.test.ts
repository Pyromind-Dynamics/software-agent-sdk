import assert from "node:assert/strict";
import test from "node:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { SessionManager, type AgentSession } from "@earendil-works/pi-coding-agent";
import { PiAgentRuntime } from "../src/agent-runtime.js";
import type { JsonObject, JsonValue } from "../src/protocol.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";
import {
  createAskUserTool,
  InputWaitRegistry,
} from "../src/user-input.js";

test("registry tracks active waits and late cancels", () => {
  const waits = new InputWaitRegistry();
  assert.equal(waits.hasActive(), false);
  assert.equal(waits.begin("input:1"), true);
  assert.equal(waits.hasActive(), true);
  waits.end("input:1");
  assert.equal(waits.hasActive(), false);
  // If cancellation arrives before registration, do not create a dangling wait.
  waits.markCancelled("input:2");
  assert.equal(waits.begin("input:2"), false);
  assert.equal(waits.hasActive(), false);
});

function recordingPeer(
  handler?: (method: string, params: JsonObject) => JsonValue,
): JsonlRpcPeer & { calls: Array<{ method: string; params: JsonObject }> } {
  const calls: Array<{ method: string; params: JsonObject }> = [];
  return {
    calls,
    request: async (method: string, params: JsonObject) => {
      calls.push({ method, params });
      return handler ? handler(method, params) : null;
    },
    emit: () => {},
  } as unknown as JsonlRpcPeer & { calls: Array<{ method: string; params: JsonObject }> };
}

const toolSpec = {
  description: "Ask the user.",
  inputSchema: { type: "object" } as JsonObject,
};

test("ask_user tool returns the answer as JSON content", async () => {
  const waits = new InputWaitRegistry();
  const peer = recordingPeer((_method, params) => {
    assert.equal(typeof params.request_id, "string");
    return { status: "answered", answers: [{ question_id: "q1", text: "csv" }] };
  });
  const tool = createAskUserTool(peer, toolSpec, waits, () => true);
  const result = await tool.execute("call-1", { questions: [{ id: "q1", prompt: "?" }] });
  const payload = JSON.parse(result.content[0]!.type === "text" ? (result.content[0] as { text: string }).text : "{}");
  assert.equal(payload.status, "answered");
  assert.deepEqual(payload.answers, [{ question_id: "q1", text: "csv" }]);
  assert.equal(waits.hasActive(), false);
});

test("ask_user enriches choices using both question and option ids without changing the RPC answer", async () => {
  const answer = {
    status: "answered",
    answers: [
      { question_id: "format", option_id: "b" },
      { question_id: "scope", option_id: "b" },
      { question_id: "name", text: "我的周报" },
    ],
  };
  const original = structuredClone(answer);
  const peer = recordingPeer(() => answer);
  const tool = createAskUserTool(peer, toolSpec, new InputWaitRegistry(), () => true);
  const result = await tool.execute("call-1", { questions: [
    { id: "format", prompt: "格式？", options: [
      { id: "a", label: "CSV" }, { id: "b", label: "PDF（推荐）" },
    ] },
    { id: "scope", prompt: "范围？", options: [{ id: "b", label: "仅摘要" }] },
    { id: "name", prompt: "名称？", options: [] },
  ] });
  const expected = {
    status: "answered",
    answers: [
      { question_id: "format", option_id: "b", option_label: "PDF（推荐）" },
      { question_id: "scope", option_id: "b", option_label: "仅摘要" },
      { question_id: "name", text: "我的周报" },
    ],
  };
  assert.equal(result.content[0]?.type, "text");
  assert.deepEqual(JSON.parse((result.content[0] as { text: string }).text), expected);
  assert.deepEqual(result.details, { user_input: expected });
  assert.deepEqual(answer, original);
});

for (const status of ["cancelled", "interrupted"]) {
  test(`ask_user preserves ${status} responses`, async () => {
    const answer = { status, reason: "Execution stopped." };
    const tool = createAskUserTool(recordingPeer(() => answer), toolSpec, new InputWaitRegistry(), () => true);
    const result = await tool.execute("call-1", { questions: [
      { id: "q1", prompt: "?", options: [{ id: "a", label: "A" }] },
    ] });
    assert.deepEqual(JSON.parse((result.content[0] as { text: string }).text), answer);
    assert.deepEqual(result.details, { user_input: answer });
  });
}

test("ask_user tool refuses gated contexts", async () => {
  const peer = recordingPeer();
  const tool = createAskUserTool(peer, toolSpec, new InputWaitRegistry(), () => false);
  await assert.rejects(
    () => tool.execute("call-1", { questions: [] }),
    /not available in this execution context/,
  );
  assert.equal(peer.calls.length, 0);
});

test("aborted ask_user asks Python to clean up its wait", async () => {
  const waits = new InputWaitRegistry();
  const peer = recordingPeer(() => {
    throw new Error("request aborted");
  });
  const tool = createAskUserTool(peer, toolSpec, waits, () => true);
  await assert.rejects(
    () => tool.execute("call-1", { questions: [{ id: "q1", prompt: "?" }] }),
    /request aborted/,
  );
  await new Promise((resolve) => setImmediate(resolve));
  const cancel = peer.calls.find((call) => call.method === "input.cancel");
  assert.ok(cancel);
  const requestId = peer.calls[0]!.params.request_id;
  assert.equal(cancel!.params.request_id, requestId);
  assert.ok(typeof requestId === "string");
  assert.equal(waits.hasActive(), false);
  assert.equal(waits.begin(requestId), true);
  waits.end(requestId);
});

test("prompt admission rejects while an input wait is active", async () => {
  const runtime = new PiAgentRuntime({ emit: () => {} } as unknown as JsonlRpcPeer);
  const session = {
    sessionManager: SessionManager.inMemory(),
    isStreaming: false,
    prompt: async () => {},
  } as unknown as AgentSession;
  const state = runtime as unknown as {
    session: AgentSession;
    sessionId: string;
    userInput: InputWaitRegistry;
  };
  state.session = session;
  state.sessionId = "session-1";
  state.userInput.begin("input:active");
  await assert.rejects(
    () => runtime.handle("prompt", { run_id: "run-1", content: [{ type: "text", text: "hi" }] }),
    /input_pending/,
  );
  state.userInput.end("input:active");
  const accepted = await runtime.handle("prompt", {
    run_id: "run-1",
    content: [{ type: "text", text: "hi" }],
  });
  assert.equal(accepted.accepted, true);
});

test("run.suspend cancels queued stages and aborts the session", async () => {
  const runtime = new PiAgentRuntime({ emit: () => {} } as unknown as JsonlRpcPeer);
  const aborted: string[] = [];
  const session = {
    sessionManager: SessionManager.inMemory(),
    isStreaming: false,
    abort: async () => {
      aborted.push("session");
    },
    waitForIdle: async () => {},
  } as unknown as AgentSession;
  const state = runtime as unknown as {
    session: AgentSession;
    sessionId: string;
    executionSuspended: boolean;
    stages: Map<string, {
      executionId: string;
      request: JsonObject;
      state: "queued" | "running" | "finished";
      promise: Promise<JsonValue>;
      resolve: (result: JsonValue) => void;
    }>;
  };
  state.session = session;
  state.sessionId = "session-1";
  let resolveStage!: (result: JsonValue) => void;
  const promise = new Promise<JsonValue>((resolve) => {
    resolveStage = resolve;
  });
  state.stages.set("stage-1", {
    executionId: "stage:x",
    request: {},
    state: "queued",
    promise,
    resolve: resolveStage,
  });
  const result = await runtime.handle("run.suspend", {});
  assert.deepEqual(result, { suspended: true });
  assert.deepEqual(aborted, ["session"]);
  assert.equal(state.executionSuspended, true);
  assert.deepEqual(await promise, { status: "cancelled", execution_id: "stage:x" });
});

function danglingAskUserSession(): AgentSession {
  const sessionManager = SessionManager.inMemory();
  const assistant: AgentMessage = {
    role: "assistant",
    content: [
      {
        type: "toolCall",
        id: "call-dangling",
        name: "ask_user",
        arguments: { questions: [{ id: "q1", prompt: "?" }] },
      },
    ],
    api: "openai-completions",
    provider: "openai",
    model: "test-model",
    usage: { input: 0, output: 0 } as never,
    stopReason: "toolCalls",
    timestamp: Date.now(),
  } as unknown as AgentMessage;
  sessionManager.appendMessage(assistant);
  return { sessionManager } as unknown as AgentSession;
}

test("recovery repair appends one interrupted result per dangling ask_user", () => {
  const runtime = new PiAgentRuntime({ emit: () => {} } as unknown as JsonlRpcPeer);
  const session = danglingAskUserSession();
  const repair = (
    runtime as unknown as {
      repairInterruptedInputs: (
        session: AgentSession,
        records: JsonValue | undefined,
      ) => void;
    }
  ).repairInterruptedInputs.bind(runtime);
  repair(session, [
    { request_id: "input-1", operation_id: "call-dangling" },
  ]);
  const branch = session.sessionManager.getBranch();
  const results = branch.filter(
    (entry) =>
      entry.type === "message" &&
      (entry as { message?: { role?: string; toolCallId?: string } }).message?.role ===
        "toolResult",
  );
  assert.equal(results.length, 1);
  // Repeated coordination must not append duplicate records.
  repair(session, [
    { request_id: "input-1", operation_id: "call-dangling" },
  ]);
  assert.equal(
    session.sessionManager.getBranch().filter(
      (entry) =>
        entry.type === "message" &&
        (entry as { message?: { role?: string } }).message?.role === "toolResult",
    ).length,
    1,
  );
});

test("sessions without ask_user calls are left untouched by repair", () => {
  const runtime = new PiAgentRuntime({ emit: () => {} } as unknown as JsonlRpcPeer);
  const session = {
    sessionManager: SessionManager.inMemory(),
  } as unknown as AgentSession;
  session.sessionManager.appendMessage({
    role: "user",
    content: "hello",
    timestamp: Date.now(),
  } as never);
  const before = session.sessionManager.getBranch().length;
  (runtime as unknown as {
    repairInterruptedInputs: (
      session: AgentSession,
      records: JsonValue | undefined,
    ) => void;
  }).repairInterruptedInputs(session, []);
  assert.equal(session.sessionManager.getBranch().length, before);
});

test("prompt reservation serializes workflow synchronization with new questions", async () => {
  const waits = new InputWaitRegistry();
  const peer = recordingPeer(() => ({ status: "answered", answers: [{ question_id: "q", text: "ok" }] }));
  const tool = createAskUserTool(peer, toolSpec, waits, () => true);
  waits.reservePrompt("message-1");
  const question = tool.execute("call-1", { questions: [{ id: "q", prompt: "?" }] });
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.equal(peer.calls.length, 0, "input cannot start during admitted workflow synchronization");
  waits.releasePrompt("message-1");
  await question;
  assert.equal(peer.calls[0]?.method, "input.request");
  waits.begin("active");
  assert.throws(() => waits.reservePrompt("message-2"), { code: "input_pending" });
});

test("cancellation interrupts a question waiting for prompt admission", async () => {
  const waits = new InputWaitRegistry();
  const peer = recordingPeer();
  const controller = new AbortController();
  waits.reservePrompt("message-1");
  const tool = createAskUserTool(peer, toolSpec, waits, () => true);
  const question = tool.execute("call-1", { questions: [{ id: "q", prompt: "?" }] }, controller.signal);
  controller.abort();
  await assert.rejects(question);
  assert.equal(waits.hasActive(), false);
  assert.equal(peer.calls.length, 0);
  waits.releasePrompt("message-1");
});

for (const suspendFirst of [true, false]) {
test(`notifications survive suspension ${suspendFirst ? "before" : "after"} scheduling`, async () => {
  const runtime = new PiAgentRuntime({ emit: () => {} } as unknown as JsonlRpcPeer);
  const notifications: string[] = [];
  const session = {
    sessionManager: SessionManager.inMemory(), isStreaming: false, messages: [],
    prompt: async () => {},
    abort: async () => {}, waitForIdle: async () => {},
    sendCustomMessage: async () => { notifications.push("delivered"); },
  } as unknown as AgentSession;
  const state = runtime as unknown as {
    session: AgentSession; sessionId: string; executionSuspended: boolean;
  };
  state.session = session; state.sessionId = "session"; state.executionSuspended = suspendFirst;
  assert.deepEqual(await runtime.handle("notify", { run_id: "background", content: "done" }), { accepted: true, queued: suspendFirst });
  if (!suspendFirst) await runtime.handle("run.suspend", {});
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.deepEqual(notifications, []);
  assert.equal(session.sessionManager.getBranch().some((entry) =>
    entry.type === "custom" && entry.customType === "pyromind.notification.finished"), false);
  await runtime.handle("prompt", { run_id: "user", content: [{ type: "text", text: "continue" }] });
  await new Promise<void>((resolve) => setImmediate(resolve));
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.deepEqual(notifications, ["delivered"]);
});
}

test("a rejected steer does not lift execution suspension", async () => {
  const runtime = new PiAgentRuntime({ emit: () => {} } as unknown as JsonlRpcPeer);
  const state = runtime as unknown as { session: AgentSession; executionSuspended: boolean };
  state.session = { isStreaming: true, steer: async () => { throw new Error("rejected"); } } as unknown as AgentSession;
  state.executionSuspended = true;
  await assert.rejects(runtime.handle("steer", { run_id: "user", content: [{ type: "text", text: "hi" }] }), /rejected/);
  assert.equal(state.executionSuspended, true);
});

test("an accepted steer does not lift a newer execution suspension", async () => {
  const runtime = new PiAgentRuntime({ emit: () => {} } as unknown as JsonlRpcPeer);
  const state = runtime as unknown as { session: AgentSession; executionSuspended: boolean };
  let accept!: () => void;
  state.session = {
    isStreaming: true,
    steer: () => new Promise<void>((resolve) => { accept = resolve; }),
    abort: async () => {}, waitForIdle: async () => {},
  } as unknown as AgentSession;
  const message = runtime.handle("steer", { run_id: "user", content: [{ type: "text", text: "hi" }] });
  await runtime.handle("run.suspend", {});
  accept();
  assert.deepEqual(await message, { accepted: true, steered: true });
  assert.equal(state.executionSuspended, true);
});
