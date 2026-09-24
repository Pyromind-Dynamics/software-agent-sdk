import assert from "node:assert/strict";
import { mkdir, mkdtemp, realpath, writeFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { PiAgentRuntime } from "../src/agent-runtime.js";
import type { JsonObject, JsonValue, RunnerEvent } from "../src/protocol.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";

test("sandbox write recovers when the control plane replaces the container", async (context) => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "pi-sandbox-write-")));
  const workspace = join(root, "conversation");
  const skillsDirectory = join(root, "skills");
  await mkdir(join(workspace, "public_data"), { recursive: true });
  await mkdir(join(workspace, "pi", "terminal-output"), { recursive: true });
  await mkdir(skillsDirectory, { recursive: true });
  await writeFile(
    join(skillsDirectory, "SKILL.md"),
    "---\nname: demo\ndescription: A demo skill.\n---\nDemo.\n",
    "utf8",
  );

  const uploads: string[] = [];
  const chunkLengths: Array<string | undefined> = [];
  const fileApi = createServer((request, response) => {
    const url = request.url ?? "";
    uploads.push(`${request.method} ${url}`);
    if (request.method === "PUT") chunkLengths.push(request.headers["content-length"]);
    if (url.includes("/files/chunks/init")) {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ data: { upload_id: "upload-1" } }));
      return;
    }
    response.writeHead(200, { "content-type": "application/json" });
    response.end("{}");
  });
  await new Promise<void>((resolve) => fileApi.listen(0, "127.0.0.1", resolve));
  context.after(() => fileApi.close());
  const liveBaseUrl = `http://127.0.0.1:${(fileApi.address() as AddressInfo).port}`;
  const deadBaseUrl = await unusedBaseUrl();

  let modelCalls = 0;
  const model = createServer(async (request, response) => {
    await readJson(request);
    modelCalls += 1;
    if (modelCalls === 1) {
      sendToolCall(
        response,
        "write",
        { path: "public_data/out.txt", content: "hello" },
        "call-write",
      );
      return;
    }
    sendText(response, "written");
  });
  await new Promise<void>((resolve) => model.listen(0, "127.0.0.1", resolve));
  context.after(() => model.close());

  const ensures: JsonObject[] = [];
  let resolveFinished: (() => void) | undefined;
  const finished = new Promise<void>((resolve) => {
    resolveFinished = resolve;
  });
  const peer = {
    request: async (method: string, params: JsonObject): Promise<JsonValue> => {
      if (method !== "sandbox.ensure") throw new Error(`unexpected request: ${method}`);
      ensures.push(params);
      // The first bundle is the container the platform has already replaced.
      const baseUrl = ensures.length === 1 ? deadBaseUrl : liveBaseUrl;
      return {
        base_url: baseUrl,
        ws_base_url: baseUrl,
        sandbox_id: "sbx-1",
        api_key: "key",
        workspace_path: "/target-workspace/.pyromind-agent/conv-1",
        storage_path: "/target-workspace",
      };
    },
    emit: (event: RunnerEvent): void => {
      if (event.kind === "run.finished") resolveFinished?.();
    },
  } as unknown as JsonlRpcPeer;
  const runtime = new PiAgentRuntime(peer);

  await runtime.handle("start", {
    session_id: "sandbox-write",
    workspace_root: workspace,
    terminal_backend: "sandbox",
    session_path: join(workspace, "pi", "session.jsonl"),
    skills_directory: skillsDirectory,
    skill_roots: [],
    system_prompt: "Write the requested file.",
    model: {
      provider: "openai",
      id: "custom-chat-model",
      api_key: "test-key",
      base_url: `http://127.0.0.1:${(model.address() as AddressInfo).port}/v1`,
    },
    tools: [],
  });
  await runtime.handle("prompt", {
    run_id: "run-1",
    content: [{ type: "text", text: "Write public_data/out.txt." }],
  });

  let timeout: NodeJS.Timeout | undefined;
  await Promise.race([
    finished,
    new Promise<never>((_resolve, reject) => {
      timeout = setTimeout(() => reject(new Error("runner timed out")), 10_000);
    }),
  ]).finally(() => clearTimeout(timeout));

  assert.deepEqual(ensures, [{ refresh: false }, { refresh: true }]);
  assert.deepEqual(uploads, [
    "POST /sandboxes/sbx-1/files/chunks/init?total_size=5&total_chunks=1",
    "PUT /sandboxes/sbx-1/files/chunks/upload-1/part?chunk_index=0",
    "POST /sandboxes/sbx-1/files/chunks/upload-1/complete" +
      "?path=%2Ftarget-workspace%2F.pyromind-agent%2Fconv-1%2Fpublic_data%2Fout.txt" +
      "&total_chunks=1",
  ]);
  assert.deepEqual(chunkLengths, ["5"]);
});

async function unusedBaseUrl(): Promise<string> {
  const server = createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  await new Promise<void>((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
  return `http://127.0.0.1:${port}`;
}

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

function sendText(response: ServerResponse, text: string): void {
  sendEvents(response, [
    chunk({ role: "assistant" }, null),
    chunk({ content: text }, null),
    chunk({}, "stop"),
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
