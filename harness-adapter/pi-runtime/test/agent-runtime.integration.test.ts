import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { AddressInfo } from "node:net";
import type { JsonObject, JsonValue, RunnerEvent } from "../src/protocol.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";
import { PiAgentRuntime } from "../src/agent-runtime.js";

test("AgentSession completes the workflow-generation tool loop over chat completions", async (context) => {
  const workspace = await mkdtemp(join(tmpdir(), "pi-runtime-test-"));
  const skill = join(workspace, "generate-workflow-dsl");
  const knowledge = join(workspace, "knowledge-resource");
  await mkdir(join(skill, "references"), { recursive: true });
  await mkdir(knowledge);
  await writeFile(join(skill, "SKILL.md"), [
    "---",
    "name: generate-workflow-dsl",
    "description: Generate and validate a workflow DSL.",
    "---",
    "Read references/workflow-contracts.md, generate and validate the workflow.",
    "",
  ].join("\n"), "utf8");
  await writeFile(join(skill, "references", "workflow-contracts.md"), "A workflow is valid when it defines workflow.\n", "utf8");

  const requests: Array<{ url: string; body: JsonObject }> = [];
  const server = createServer(async (request, response) => {
    const body = await readJson(request);
    requests.push({ url: request.url ?? "", body });
    if (requests.length === 1) sendToolCall(response, "read", { path: join(skill, "SKILL.md") }, "call-skill");
    else if (requests.length === 2) sendToolCall(response, "read", { path: join(skill, "references", "workflow-contracts.md") }, "call-contract");
    else if (requests.length === 3) sendToolCall(response, "write", {
      path: "public_data/workflow_canvas/workflow.py",
      content: "workflow = SFTWorkflow()\n",
    }, "call-write");
    else if (requests.length === 4) sendToolCall(response, "validate_workflow_dsl", {
      dsl_path: "public_data/workflow_canvas/workflow.py",
    }, "call-validate");
    else sendText(response, "SFT workflow generated and validated.");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  context.after(() => server.close());
  const address = server.address() as AddressInfo;

  const events: RunnerEvent[] = [];
  const toolRequests: JsonObject[] = [];
  let resolveFinished: (() => void) | undefined;
  const finished = new Promise<void>((resolve) => { resolveFinished = resolve; });
  const peer = {
    request: async (method: string, params: JsonObject): Promise<JsonValue> => {
      assert.equal(method, "tool.execute");
      toolRequests.push(params);
      return {
        is_error: false,
        content: [{ type: "text", text: "workflow valid" }],
        details: { valid: true },
      };
    },
    emit: (event: RunnerEvent): void => {
      events.push(event);
      if (event.kind === "run.finished") resolveFinished?.();
    },
  } as unknown as JsonlRpcPeer;
  const runtime = new PiAgentRuntime(peer);

  await runtime.handle("start", {
    session_id: "integration-session",
    workspace_root: workspace,
    terminal_backend: "local",
    session_path: join(workspace, "pi", "session.jsonl"),
    skill_root: skill,
    knowledge_root: knowledge,
    system_prompt: "Generate an SFT workflow by following the skill, then validate it.",
    model: {
      provider: "openai",
      id: "custom-chat-model",
      api_key: "test-key",
      base_url: `http://127.0.0.1:${address.port}/v1`,
    },
    tools: [{
      name: "validate_workflow_dsl",
      description: "Validate a workflow DSL file.",
      input_schema: {
        type: "object",
        properties: { dsl_path: { type: "string" } },
        required: ["dsl_path"],
        additionalProperties: false,
      },
    }],
  });
  await runtime.handle("prompt", {
    run_id: "run-1",
    content: [{ type: "text", text: "Generate an SFT workflow." }],
  });

  let timeout: NodeJS.Timeout | undefined;
  await Promise.race([
    finished,
    new Promise<never>((_resolve, reject) => {
      timeout = setTimeout(() => reject(new Error("runner timed out")), 5_000);
      timeout.unref();
    }),
  ]).finally(() => clearTimeout(timeout));

  const finishedEvents = events.filter((event) => event.kind === "run.finished");
  assert.equal(finishedEvents.length, 1, JSON.stringify({ events, requests }, null, 2));
  assert.equal(finishedEvents[0]!.payload.outcome &&
    typeof finishedEvents[0]!.payload.outcome === "object" &&
    !Array.isArray(finishedEvents[0]!.payload.outcome)
      ? finishedEvents[0]!.payload.outcome.status
      : undefined, "completed");
  assert.equal(events.some((event) => event.kind === "tool.completed"), true);
  const writeCompleted = events.find((event) =>
    event.kind === "tool.completed" && event.payload.tool_name === "write"
  );
  assert.equal(
    writeCompleted?.payload.arguments &&
      typeof writeCompleted.payload.arguments === "object" &&
      !Array.isArray(writeCompleted.payload.arguments)
      ? writeCompleted.payload.arguments.path
      : undefined,
    "public_data/workflow_canvas/workflow.py",
  );
  assert.equal(requests.length, 5);
  assert.deepEqual(requests.map((request) => request.url), Array(5).fill("/v1/chat/completions"));
  for (const request of requests) {
    assert.equal(Object.hasOwn(request.body, "max_tokens"), false);
    assert.equal(Object.hasOwn(request.body, "max_completion_tokens"), false);
  }
  assert.equal(JSON.stringify(requests[1]!.body).includes("output_text"), false);
  const messages = requests[1]!.body.messages;
  assert.equal(Array.isArray(messages), true);
  assert.equal((messages as JsonValue[]).some((message) => (
    typeof message === "object" && message !== null && !Array.isArray(message) && message.role === "tool"
  )), true);
  assert.equal(toolRequests.length, 1);
  assert.equal(toolRequests[0]!.tool_name, "validate_workflow_dsl");
  assert.equal(
    await readFile(join(workspace, "public_data", "workflow_canvas", "workflow.py"), "utf8"),
    "workflow = SFTWorkflow()\n",
  );
  const sessionLog = await readFile(join(workspace, "pi", "session.jsonl"), "utf8");
  assert.equal(sessionLog.includes("SFT workflow generated and validated."), true);
  const firstRequest = JSON.stringify(requests[0]!.body);
  assert.equal(firstRequest.includes("<available_skills>"), true);
  assert.equal(firstRequest.includes(join(skill, "SKILL.md")), true);
});

test("a persistent length stop finishes as output_truncated, never completed", async (context) => {
  const server = createServer(async (request, response) => {
    await readJson(request);
    sendText(response, "I", "length");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  context.after(() => server.close());
  const address = server.address() as AddressInfo;
  const workspace = await mkdtemp(join(tmpdir(), "pi-runtime-length-"));
  const events: RunnerEvent[] = [];
  let resolveFinished: (() => void) | undefined;
  const finished = new Promise<void>((resolve) => { resolveFinished = resolve; });
  const peer = {
    request: async (): Promise<JsonValue> => { throw new Error("unexpected Python request"); },
    emit: (event: RunnerEvent): void => {
      events.push(event);
      if (event.kind === "run.finished") resolveFinished?.();
    },
  } as unknown as JsonlRpcPeer;
  const runtime = new PiAgentRuntime(peer);

  await runtime.handle("start", {
    session_id: "length-session",
    workspace_root: workspace,
    terminal_backend: "local",
    session_path: join(workspace, "pi", "session.jsonl"),
    skill_root: workspace,
    system_prompt: "Answer the user.",
    model: {
      provider: "openai",
      id: "deepseek-v4-flash-0731",
      api_key: "test-key",
      base_url: `http://127.0.0.1:${address.port}/v1`,
      context_window: 200_000,
    },
    tools: [],
  });
  await runtime.handle("prompt", {
    run_id: "run-length",
    content: [{ type: "text", text: "Give a complete answer." }],
  });
  let timeout: NodeJS.Timeout | undefined;
  await Promise.race([
    finished,
    new Promise<never>((_resolve, reject) => {
      timeout = setTimeout(() => reject(new Error("length runner timed out")), 5_000);
      timeout.unref();
    }),
  ]).finally(() => clearTimeout(timeout));

  const terminal = events.filter((event) => event.kind === "run.finished");
  assert.equal(terminal.length, 1);
  const outcome = terminal[0]!.payload.outcome as JsonObject;
  assert.equal(outcome.status, "failed");
  assert.equal(outcome.stop_reason, "length");
  assert.equal(outcome.error_code, "output_truncated");
});

async function readJson(request: IncomingMessage): Promise<JsonObject> {
  let body = "";
  for await (const chunk of request) body += String(chunk);
  return JSON.parse(body) as JsonObject;
}

function sendToolCall(response: ServerResponse, name: string, args: JsonObject, id: string): void {
  sendEvents(response, [
    chunk({ role: "assistant" }, null),
    chunk({ tool_calls: [{ index: 0, id, type: "function", function: { name, arguments: "" } }] }, null),
    chunk({ tool_calls: [{ index: 0, function: { arguments: JSON.stringify(args) } }] }, null),
    chunk({}, "tool_calls"),
  ]);
}

function sendText(response: ServerResponse, text: string, finishReason: JsonValue = "stop"): void {
  sendEvents(response, [chunk({ role: "assistant" }, null), chunk({ content: text }, null), chunk({}, finishReason)]);
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
