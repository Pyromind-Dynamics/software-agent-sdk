import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough, Writable } from "node:stream";
import test from "node:test";
import type { AddressInfo } from "node:net";
import type { JsonObject, JsonValue, RunnerEvent } from "../src/protocol.js";
import { JsonlRpcPeer } from "../src/rpc-peer.js";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { PiAgentRuntime } from "../src/agent-runtime.js";

const QUESTIONS = [
  {
    id: "duplicate_policy",
    prompt: "How should duplicate rows be handled?",
    options: [
      { id: "keep_first", label: "Keep the first occurrence" },
      { id: "drop_all", label: "Drop every duplicate" },
    ],
  },
  { id: "output_name", prompt: "Output file name?", options: [] },
];

test("ask_user waits for the answer and continues within the same run", async (context) => {
  const previousTimeout = process.env.PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS;
  process.env.PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS = "500";
  context.after(() => {
    if (previousTimeout === undefined) delete process.env.PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS;
    else process.env.PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS = previousTimeout;
  });
  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-user-input-")));
  const workspace = join(root, "conversation");
  const skillsDirectory = join(root, "skills");
  await mkdir(join(workspace, "public_data"), { recursive: true });
  await mkdir(join(workspace, "pi", "terminal-output"), { recursive: true });
  await mkdir(skillsDirectory);

  const requests: Array<{ url: string; body: JsonObject }> = [];
  const server = createServer(async (request, response) => {
    const body = await readJson(request);
    requests.push({ url: request.url ?? "", body });
    if (requests.length === 1) {
      // First, the model asks a structured question.
      sendToolCall(
        response,
        "ask_user",
        { questions: QUESTIONS },
        "call-ask",
      );
      return;
    }
    // Then the answer enters the tool result and the model finishes in the same run.
    const messages = body.messages as JsonValue[];
    const toolResult = JSON.stringify(
      (messages as Array<{ role?: string; content?: unknown }>).find(
        (message) => message.role === "tool",
      ) ?? {},
    );
    assert.equal(
      toolResult.includes("keep_first"),
      true,
      `tool result must carry the user's answers: ${toolResult}`,
    );
    assert.equal(toolResult.includes("Keep the first occurrence"), true);
    assert.equal(toolResult.includes("cleaned_data.csv"), true);
    sendText(response, "Question answered; proceeding.");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  context.after(() => server.close());
  const address = server.address() as AddressInfo;

  const events: RunnerEvent[] = [];
  const inputRequests: JsonObject[] = [];
  let answerGate: ((value: JsonValue) => void) | undefined;
  const peer = {
    request: async (method: string, params: JsonObject): Promise<JsonValue> => {
      if (method === "input.request") {
        inputRequests.push(params);
        // No further model request occurs before the answer; wait for Python to reply.
        return new Promise<JsonValue>((resolve) => {
          answerGate = resolve;
        });
      }
      throw new Error(`unexpected peer request: ${method}`);
    },
    emit: (event: RunnerEvent): void => {
      events.push(event);
    },
  } as unknown as JsonlRpcPeer;
  const runtime = new PiAgentRuntime(peer);
  context.after(() => disposeRuntime(runtime));

  await runtime.handle("start", {
    session_id: "user-input-session",
    workspace_root: workspace,
    terminal_backend: "os-sandbox",
    session_path: join(workspace, "pi", "session.jsonl"),
    skills_directory: skillsDirectory,
    skill_roots: [],
    system_prompt: "Ask the user when a decision is theirs.",
    model: {
      provider: "openai",
      id: "custom-chat-model",
      api_key: "test-key",
      base_url: `http://127.0.0.1:${address.port}/v1`,
    },
    tools: [],
    user_input: {
      enabled: true,
      tool: {
        name: "ask_user",
        description: "Ask the user structured questions.",
        input_schema: {
          type: "object",
          properties: {
            questions: {
              type: "array",
              minItems: 1,
              maxItems: 3,
              items: {
                type: "object",
                properties: {
                  id: { type: "string" },
                  prompt: { type: "string" },
                  options: {
                    type: "array",
                    items: {
                      type: "object",
                      properties: {
                        id: { type: "string" },
                        label: { type: "string" },
                        description: { type: "string" },
                      },
                      required: ["id", "label"],
                    },
                  },
                },
                required: ["id", "prompt", "options"],
              },
            },
          },
          required: ["questions"],
          additionalProperties: false,
        },
      },
    },
  });

  let finishedRun: RunnerEvent | undefined;
  const finished = new Promise<void>((resolve) => {
    const done = () => {
      if (finishedRun) resolve();
    };
    (peer as unknown as { emit: (event: RunnerEvent) => void }).emit = (
      event: RunnerEvent,
    ) => {
      events.push(event);
      if (event.kind === "run.finished") {
        finishedRun = event;
        done();
      }
    };
    void done;
  });

  await runtime.handle("prompt", {
    run_id: "run-1",
    content: [{ type: "text", text: "Clean the dataset." }],
  });

  // No second model request occurs before the answer; wait until ask_user is waiting.
  const waits = (
    runtime as unknown as { userInput: { hasActive(): boolean } }
  ).userInput;
  const deadline = Date.now() + 10_000;
  while (
    inputRequests.length < 1 &&
    Date.now() < deadline
  ) {
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
  }
  assert.equal(inputRequests.length, 1);
  assert.equal(waits.hasActive(), true);
  assert.equal(requests.length, 1, "model must not be re-prompted before the answer");
  assert.equal(inputRequests[0]!.tool_call_id, "call-ask");
  const asked = inputRequests[0]!.questions as JsonValue[];
  assert.equal(asked.length, 2);

  await new Promise<void>((resolve) => setTimeout(resolve, 750));
  assert.equal(waits.hasActive(), true, "human wait outlives the model idle budget");
  assert.equal(finishedRun, undefined);
  assert.equal(requests.length, 1);

  // The user answers, ask_user returns a tool result, and the same run continues.
  answerGate?.({
    status: "answered",
    answers: [
      { question_id: "duplicate_policy", option_id: "keep_first" },
      { question_id: "output_name", text: "cleaned_data.csv" },
    ],
  });

  let timeout: NodeJS.Timeout | undefined;
  await Promise.race([
    finished,
    new Promise<never>((_resolve, reject) => {
      timeout = setTimeout(() => reject(new Error("runner timed out")), 15_000);
      timeout.unref();
    }),
  ]).finally(() => clearTimeout(timeout));

  assert.equal(finishedRun?.kind, "run.finished");
  assert.equal(
    finishedRun?.payload.outcome &&
      typeof finishedRun.payload.outcome === "object" &&
      !Array.isArray(finishedRun.payload.outcome)
      ? finishedRun.payload.outcome.status
      : undefined,
    "completed",
  );
  assert.equal(requests.length, 2);
  assert.deepEqual(
    requests.map((request) => request.url),
    ["/v1/chat/completions", "/v1/chat/completions"],
  );

  // The tool declaration is sent with the first request; later requests include its result.
  const firstTools = requests[0]!.body.tools as JsonValue[];
  const askTool = (firstTools as Array<{ function?: { name?: string } }>).find(
    (tool) => tool.function?.name === "ask_user",
  );
  assert.ok(askTool, "ask_user must be declared to the model");
  const sessionLog = await readFile(join(workspace, "pi", "session.jsonl"), "utf8");
  assert.equal(sessionLog.includes("cleaned_data.csv"), true);

});

test("ask_user refuses prompts while a wait is active", async (context) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-user-input-2-")));
  const workspace = join(root, "conversation");
  await mkdir(join(workspace, "public_data"), { recursive: true });
  await mkdir(join(workspace, "pi", "terminal-output"), { recursive: true });
  await mkdir(join(root, "skills"));

  let modelRequests = 0;
  const server = createServer(async (request, response) => {
    await readJson(request);
    modelRequests++;
    sendToolCall(response, "ask_user", { questions: QUESTIONS.slice(0, 1) }, "call-ask", {
      id: "call-write", name: "write",
      args: { path: "public_data/should-not-exist.txt", content: "unexpected" },
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  context.after(() => server.close());
  const address = server.address() as AddressInfo;

  const events: RunnerEvent[] = [];
  const incoming = new PassThrough();
  const outgoing = new Writable({
    write(chunk, _encoding, done) {
      for (const line of chunk.toString().trim().split("\n")) {
        const frame = JSON.parse(line);
        if (frame.type === "pi.event") events.push(frame);
        if (frame.type === "request" && frame.method === "input.cancel") {
          incoming.write(JSON.stringify({ protocolVersion: frame.protocolVersion,
            type: "response", requestId: frame.requestId, result: { cancelled: true } }) + "\n");
        }
      }
      done();
    },
  });
  const peer = new JsonlRpcPeer(incoming, outgoing);
  const listening = peer.listen(async () => null);
  const runtime = new PiAgentRuntime(peer);
  context.after(async () => {
    await disposeRuntime(runtime);
    incoming.end();
    await listening;
  });
  await runtime.handle("start", {
    session_id: "user-input-session",
    workspace_root: workspace,
    terminal_backend: "os-sandbox",
    session_path: join(workspace, "pi", "session.jsonl"),
    skills_directory: join(root, "skills"),
    skill_roots: [],
    system_prompt: "Ask the user.",
    model: {
      provider: "openai",
      id: "custom-chat-model",
      api_key: "test-key",
      base_url: `http://127.0.0.1:${address.port}/v1`,
    },
    tools: [],
    user_input: {
      enabled: true,
      tool: {
        name: "ask_user",
        description: "Ask the user structured questions.",
        input_schema: { type: "object" },
      },
    },
  });
  const pending = runtime.handle("prompt", {
    run_id: "run-1",
    content: [{ type: "text", text: "Ask me something." }],
  });
  // Wait until ask_user is actually waiting.
  const waits = (
    runtime as unknown as { userInput: { hasActive(): boolean } }
  ).userInput;
  const deadline = Date.now() + 10_000;
  while (!waits.hasActive() && Date.now() < deadline) {
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
  }
  assert.equal(waits.hasActive(), true);
  // Reject a new ordinary prompt while a question is pending.
  await assert.rejects(
    () =>
      runtime.handle("prompt", {
        run_id: "run-2",
        content: [{ type: "text", text: "Another question." }],
      }),
    /input_pending/,
  );
  await assert.rejects(
    runtime.handle("prompt.prepare", { run_id: "run-2", content: [{ type: "text", text: "hi" }] }),
    { code: "input_pending" },
  );
  await runtime.handle("run.suspend", {});
  await pending;
  await new Promise<void>((resolve) => setImmediate(resolve));
  const finished = events.find((event) => event.kind === "run.finished");
  assert.equal((finished?.payload.outcome as JsonObject)?.status, "cancelled");
  assert.equal(waits.hasActive(), false);
  assert.equal(modelRequests, 1, "cancel must not request another model response");
  await assert.rejects(readFile(join(workspace, "public_data", "should-not-exist.txt")), { code: "ENOENT" });
});

test("stalled model arguments exhaust a bounded retry budget and allow a new turn", { timeout: 15_000 }, async (context) => {
  const settings = { PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS: "500",
    PYROMIND_PI_MODEL_MAX_RETRIES: "1", PYROMIND_PI_MODEL_RETRY_BASE_DELAY_MS: "0" };
  for (const [key, value] of Object.entries(settings)) {
    const previous = process.env[key];
    process.env[key] = value;
    context.after(() => {
      if (previous === undefined) delete process.env[key];
      else process.env[key] = previous;
    });
  }
  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-input-stall-")));
  const workspace = join(root, "conversation");
  await mkdir(join(workspace, "public_data"), { recursive: true });
  await mkdir(join(workspace, "pi", "terminal-output"), { recursive: true });
  await mkdir(join(root, "skills"));
  let requests = 0;
  let lastMessages: Array<{ role?: string; tool_call_id?: string }> = [];
  const server = createServer(async (request, response) => {
    const body = await readJson(request);
    lastMessages = body.messages as typeof lastMessages;
    requests++;
    if (requests <= 2 || requests === 4 || requests === 5) {
      response.writeHead(200, { "content-type": "text/event-stream" });
      response.write(`data: ${JSON.stringify(chunk({ role: "assistant", tool_calls: [
        { index: 0, id: "incomplete", type: "function", function: { name: "ask_user", arguments: '{"questions":[' } },
      ] }, null))}\n\n`);
      const heartbeat = setInterval(() => response.write(": heartbeat\n\n"), 30);
      response.on("close", () => clearInterval(heartbeat));
    } else if (requests === 3) {
      sendToolCall(response, "ask_user", { questions: QUESTIONS }, "complete");
    } else sendText(response, "Recovered after answer.");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  context.after(() => { server.closeAllConnections(); server.close(); });
  const events: RunnerEvent[] = [];
  let inputCount = 0;
  const runtime = new PiAgentRuntime({
    emit: (event: RunnerEvent) => events.push(event),
    request: async (method: string) => {
      assert.equal(method, "input.request");
      inputCount++;
      return { status: "answered", answers: [{ question_id: "duplicate_policy", option_id: "keep_first" },
        { question_id: "output_name", text: "cleaned_data.csv" }] };
    },
  } as unknown as JsonlRpcPeer);
  context.after(() => disposeRuntime(runtime));
  await runtime.handle("start", {
    session_id: "stall-session", workspace_root: workspace, terminal_backend: "os-sandbox",
    session_path: join(workspace, "pi", "session.jsonl"), skills_directory: join(root, "skills"), skill_roots: [],
    system_prompt: "Ask if needed.", tools: [],
    model: { provider: "openai", id: "local-stall", api_key: "test",
      base_url: `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1` },
    user_input: { enabled: true, tool: { description: "Ask.", input_schema: { type: "object" } } },
  });
  const waitForFinished = async (runId: string) => {
    const deadline = Date.now() + 8_000;
    while (Date.now() < deadline) {
      const event = events.find((entry) => entry.kind === "run.finished" && entry.runId === runId);
      if (event) {
        await new Promise<void>((resolve) => setImmediate(resolve));
        return event.payload.outcome as JsonObject;
      }
      await new Promise<void>((resolve) => setTimeout(resolve, 20));
    }
    throw new Error("model timeout failed to finish the run");
  };
  await runtime.handle("prompt", { run_id: "stalled", content: [{ type: "text", text: "Ask." }] });
  const failed = await waitForFinished("stalled");
  assert.equal(failed.status, "failed");
  assert.match(String(failed.message), /Model stream idle timeout/);
  assert.equal(requests, 2, "one initial attempt plus one session retry");
  assert.equal(inputCount, 0, "incomplete tool calls must never execute");
  await runtime.handle("prompt", { run_id: "next", content: [{ type: "text", text: "Continue." }] });
  assert.equal((await waitForFinished("next")).status, "failed");
  assert.equal(inputCount, 1);
  assert.equal(requests, 5, "retrying after a tool result must not execute the tool again");
  await runtime.handle("prompt", { run_id: "last", content: [{ type: "text", text: "Finish." }] });
  assert.equal((await waitForFinished("last")).status, "completed");
  assert.equal(inputCount, 1);
  assert.equal(requests, 6);
  assert.equal(lastMessages.filter((message) => message.role === "tool" && message.tool_call_id === "complete").length, 1);
});

async function disposeRuntime(runtime: PiAgentRuntime): Promise<void> {
  const state = runtime as unknown as {
    session?: AgentSession; subagents?: { close(): Promise<void> };
  };
  state.session?.dispose();
  await state.subagents?.close();
}

async function readJson(request: IncomingMessage): Promise<JsonObject> {
  const body = await new Promise<string>((resolve, reject) => {
    let data = "";
    request.on("data", (chunk: Buffer) => {
      data += chunk.toString("utf8");
    });
    request.on("end", () => resolve(data));
    request.on("error", reject);
  });
  return JSON.parse(body) as JsonObject;
}

function sendToolCall(
  response: ServerResponse,
  name: string,
  args: JsonObject,
  id: string,
  following?: { id: string; name: string; args: JsonObject },
): void {
  sendEvents(response, [
    chunk({ role: "assistant" }, null),
    chunk(
      {
        tool_calls: [
          { index: 0, id, type: "function", function: { name, arguments: "" } },
        ],
      },
      null,
    ),
    chunk(
      {
        tool_calls: [{ index: 0, function: { arguments: JSON.stringify(args) } }],
      },
      null,
    ),
    ...(following ? [chunk({ tool_calls: [{ index: 1, id: following.id, type: "function",
      function: { name: following.name, arguments: JSON.stringify(following.args) } }] }, null)] : []),
    chunk({}, "tool_calls"),
  ]);
}

function sendText(
  response: ServerResponse,
  text: string,
  finishReason: JsonValue = "stop",
): void {
  sendEvents(response, [
    chunk({ role: "assistant" }, null),
    chunk({ content: text }, null),
    chunk({}, finishReason),
  ]);
}

function chunk(delta: JsonObject, finishReason: JsonValue): JsonObject {
  return {
    id: "chatcmpl-test",
    object: "chat.completion.chunk",
    created: 1,
    model: "custom-chat-model",
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  };
}

function sendEvents(response: ServerResponse, events: JsonObject[]): void {
  response.writeHead(200, { "content-type": "text/event-stream" });
  for (const event of events) response.write(`data: ${JSON.stringify(event)}\n\n`);
  response.end("data: [DONE]\n\n");
}
