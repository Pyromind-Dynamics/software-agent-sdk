import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, realpath, symlink, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { AddressInfo } from "node:net";
import type { JsonObject, JsonValue, RunnerEvent } from "../src/protocol.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";
import { PiAgentRuntime } from "../src/agent-runtime.js";

test("AgentSession resolves an uncovered workflow node through the knowledge index without terminal", async (context) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-runtime-test-")));
  const workspace = join(root, "conversation");
  const skillsDirectory = join(root, "configured-skills");
  const skill = join(skillsDirectory, "custom-workflow");
  const extraSkill = join(root, "extra-skill");
  const knowledge = join(root, "knowledge-resource");
  await mkdir(join(workspace, "public_data"), { recursive: true });
  await mkdir(join(workspace, "pi", "terminal-output"), { recursive: true });
  await mkdir(join(skill, "references"), { recursive: true });
  await mkdir(join(knowledge, "nodes", "CustomCommandNode"), { recursive: true });
  await writeFile(join(knowledge, "index.md"), "[CustomCommandNode](nodes/CustomCommandNode/CustomCommandNode.md)\n", "utf8");
  await writeFile(join(knowledge, "nodes", "CustomCommandNode", "CustomCommandNode.md"),
    "CustomCommandNode runs a shell command on the platform; required input: command.\n", "utf8");
  await mkdir(extraSkill);
  await writeFile(join(extraSkill, "SKILL.md"), "---\nname: extra-skill\ndescription: An explicitly configured skill.\n---\nExtra skill.\n");
  const shadowedSkill = join(skillsDirectory, "extra-skill");
  await mkdir(shadowedSkill);
  await writeFile(join(shadowedSkill, "SKILL.md"), "---\nname: extra-skill\ndescription: base-shadowed-description\n---\nBase skill.\n");
  const escapedSkill = join(root, "escaped-skill");
  await mkdir(escapedSkill);
  await writeFile(join(escapedSkill, "SKILL.md"), "---\nname: escaped-skill\ndescription: escaped-skill-description\n---\nOutside skill.\n");
  await symlink(escapedSkill, join(skillsDirectory, "escaped-skill"));
  await writeFile(join(skill, "SKILL.md"), [
    "---",
    "name: custom-workflow",
    "description: Generate and validate a workflow DSL.",
    "---",
    "Read references/workflow-contracts.md first. If the node is absent, read knowledge/index.md and its linked node page.",
    "Generate and validate the workflow. Do not use terminal to discover resources or execute the node command.",
    "",
  ].join("\n"), "utf8");
  await writeFile(join(skill, "references", "workflow-contracts.md"), "Common training nodes only.\n", "utf8");

  const workflowDsl = 'custom = CustomCommandNode(command=\'echo "hello world"\', cpu=1, memory=1, gpu_count=1, gpu_product="NVIDIA-H100-80GB-HBM3")\n';

  const requests: Array<{ url: string; body: JsonObject }> = [];
  const server = createServer(async (request, response) => {
    const body = await readJson(request);
    requests.push({ url: request.url ?? "", body });
    if (requests.length === 1) sendToolCall(response, "read", { path: ".agents/skills/custom-workflow/SKILL.md" }, "call-skill");
    else if (requests.length === 2) sendToolCall(response, "read", { path: "./.agents/skills/custom-workflow/references/workflow-contracts.md" }, "call-contract");
    else if (requests.length === 3) sendToolCall(response, "read", { path: "knowledge/index.md" }, "call-index");
    else if (requests.length === 4) sendToolCall(response, "read", {
      path: "knowledge/nodes/CustomCommandNode/CustomCommandNode.md",
    }, "call-node");
    else if (requests.length === 5) sendToolCall(response, "write", {
      path: "public_data/workflow_canvas/workflow.py",
      content: workflowDsl,
    }, "call-write");
    else if (requests.length === 6) sendToolCall(response, "validate_workflow_dsl", {
      dsl_path: "public_data/workflow_canvas/workflow.py",
    }, "call-validate");
    else sendText(response, "CustomCommandNode workflow generated and validated.");
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
    terminal_backend: "os-sandbox",
    session_path: join(workspace, "pi", "session.jsonl"),
    skills_directory: skillsDirectory,
    skill_roots: [{ name: "extra-skill", path: extraSkill }],
    knowledge_root: knowledge,
    system_prompt: "Generate a single-node CustomCommandNode workflow by following the skill, then validate it.",
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
    content: [{ type: "text", text: "Generate a CustomCommandNode workflow that runs echo hello world." }],
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
  assert.equal(finishedEvents[0]!.payload.workflow_dsl, workflowDsl);
  assert.equal(typeof finishedEvents[0]!.payload.checkpoint_entry_id, "string");
  const assistantStarts = events.filter((event) => event.kind === "message.started" && event.payload.role === "assistant");
  assert.equal(new Set(assistantStarts.map((event) => event.payload.message_id)).size, assistantStarts.length);
  for (const start of assistantStarts) {
    const deltas = events.filter((event) => event.kind === "message.delta" && event.payload.message_id === start.payload.message_id);
    const completed = events.find((event) => event.kind === "message.completed" && event.payload.message_id === start.payload.message_id);
    assert.ok(completed);
    const blocks = completed.payload.content as Array<{ type: string; text?: string }>;
    assert.equal(deltas.map((event) => event.payload.text).join(""), blocks.filter((block) => block.type === "text").map((block) => block.text).join(""));
  }
  assert.equal(finishedEvents[0]!.payload.outcome &&
    typeof finishedEvents[0]!.payload.outcome === "object" &&
    !Array.isArray(finishedEvents[0]!.payload.outcome)
      ? finishedEvents[0]!.payload.outcome.status
      : undefined, "completed");
  assert.equal(events.some((event) => event.kind === "tool.completed"), true);
  assert.equal(events.some((event) => event.kind === "tool.started" && event.payload.tool_name === "terminal"), false);
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
  assert.equal(requests.length, 7);
  assert.deepEqual(requests.map((request) => request.url), Array(7).fill("/v1/chat/completions"));
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
    workflowDsl,
  );
  const sessionLog = await readFile(join(workspace, "pi", "session.jsonl"), "utf8");
  assert.equal(sessionLog.includes("CustomCommandNode workflow generated and validated."), true);
  const firstRequest = JSON.stringify(requests[0]!.body);
  assert.equal(firstRequest.includes("<available_skills>"), true);
  assert.equal(firstRequest.includes(join(skill, "SKILL.md")), true);
  assert.equal(firstRequest.includes(join(extraSkill, "SKILL.md")), true);
  assert.equal(firstRequest.includes("base-shadowed-description"), false);
  assert.equal(firstRequest.includes("escaped-skill-description"), false);
  assert.equal(JSON.stringify(requests[1]!.body).includes("Read references/workflow-contracts.md"), true);
  assert.equal(JSON.stringify(requests[2]!.body).includes("Common training nodes only."), true);
  assert.equal(JSON.stringify(requests[3]!.body).includes("[CustomCommandNode](nodes/CustomCommandNode/CustomCommandNode.md)"), true);
  assert.equal(JSON.stringify(requests[4]!.body).includes("CustomCommandNode runs a shell command on the platform"), true);

  const secondFinished = new Promise<void>((resolve) => { resolveFinished = resolve; });
  await runtime.handle("prompt", {
    run_id: "run-2",
    content: [{ type: "text", text: "Confirm the workflow is ready." }],
  });
  await Promise.race([
    secondFinished,
    new Promise<never>((_resolve, reject) => {
      timeout = setTimeout(() => reject(new Error("second turn timed out")), 5_000);
      timeout.unref();
    }),
  ]).finally(() => clearTimeout(timeout));
  assert.equal(requests.length, 8);
  for (const request of requests) {
    const messages = request.body.messages as Array<{ role: string; content: string }>;
    const system = messages.find((message) => message.role === "system" || message.role === "developer");
    assert.ok(system);
    assert.ok(system.content.includes("Generate a single-node CustomCommandNode workflow by following the skill, then validate it."));
    assert.ok(system.content.includes("<available_skills>"));
    assert.ok(system.content.includes("Paths starting with knowledge/ or .agents/skills/ are runtime resource aliases: pass them unchanged to file tools, without prepending the skill directory."));
    assert.ok(system.content.includes("For other relative paths referenced by a skill file, resolve them against the skill directory"));
    assert.ok(!system.content.includes("When a skill file references a relative path, resolve it against the skill directory"));
  }
});

test("a persistent length stop finishes as output_truncated, never completed", async (context) => {
  const server = createServer(async (request, response) => {
    await readJson(request);
    sendText(response, "I", "length");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  context.after(() => server.close());
  const address = server.address() as AddressInfo;
  const root = await mkdtemp(join(tmpdir(), "pi-runtime-length-"));
  const workspace = join(root, "conversation");
  const skill = join(root, "skill");
  await mkdir(join(workspace, "public_data"), { recursive: true });
  await mkdir(join(workspace, "pi", "terminal-output"), { recursive: true });
  await mkdir(skill);
  await writeFile(join(skill, "SKILL.md"), [
    "---",
    "name: test-skill",
    "description: Test skill.",
    "---",
    "Test instructions.",
    "",
  ].join("\n"), "utf8");
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
    terminal_backend: "os-sandbox",
    session_path: join(workspace, "pi", "session.jsonl"),
    skill_root: skill,
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
