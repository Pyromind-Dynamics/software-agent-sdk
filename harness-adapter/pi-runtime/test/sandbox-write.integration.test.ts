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
import {
  TERMINAL_BEGIN_MARKER,
  TERMINAL_END_MARKER,
  type SandboxTerminalConnector,
  type SandboxTerminalSocket,
} from "../src/sandbox-terminal.js";

test("sandbox write travels the terminal and recovers a replaced container", async (context) => {
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

  // The file API stays mounted so the test can prove the write no longer
  // touches it: every file request is a failure of the terminal transport.
  const uploads: string[] = [];
  const fileApi = createServer((request, response) => {
    uploads.push(`${request.method} ${request.url ?? ""}`);
    response.writeHead(500, { "content-type": "application/json" });
    response.end("{}");
  });
  await new Promise<void>((resolve) => fileApi.listen(0, "127.0.0.1", resolve));
  context.after(() => fileApi.close());
  const liveBaseUrl = `http://127.0.0.1:${(fileApi.address() as AddressInfo).port}`;
  const deadBaseUrl = await unusedBaseUrl();
  const liveTerminalPrefix = liveBaseUrl.replace("http://", "ws://");

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
  const deadSockets: FakeSocket[] = [];
  const liveSockets: FakeSocket[] = [];
  const connect: SandboxTerminalConnector = (url) => {
    if (!url.startsWith(liveTerminalPrefix)) {
      // The platform already replaced the container behind the first bundle:
      // every socket for it dies during the handshake.
      const dead = new FakeSocket();
      deadSockets.push(dead);
      queueMicrotask(() => dead.close());
      return dead;
    }
    const socket = new FakeSocket();
    liveSockets.push(socket);
    socket.onSend = (line) => {
      const ack = /'__PM_ACK__' '([0-9a-f]{18}):([^']+)'/.exec(line);
      if (ack) {
        socket.emit(`__PM_ACK__${ack[1]}:${ack[2]}:ok\n`);
        return;
      }
      const token = /([0-9a-f]{18})\s*$/.exec(line)?.[1];
      assert.ok(token, "watch line carries the run token");
      socket.emit(`${TERMINAL_BEGIN_MARKER}${token}\n`);
      socket.emit(`${TERMINAL_END_MARKER}${token}:0\n`);
    };
    queueMicrotask(() => socket.open());
    return socket;
  };

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
  const runtime = new PiAgentRuntime(peer, { terminalConnect: connect });

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
  assert.deepEqual(uploads, [], "the HTTP file API is never called");
  assert.equal(liveSockets.length, 1, "the write is replayed on the replacement terminal");
  assert.equal(deadSockets.length, 3, "the stale bundle is retried before it is dropped");
  const script = decodeStagedScript(liveSockets[0]!.sent);
  assert.match(script, /mkdir -p [^\n]*conv-1\/public_data/);
  assert.match(script, /aGVsbG8=/, "the payload travels as base64 over the TTY");
  assert.match(script, /openssl base64 -d -A/);
  assert.ok(liveSockets[0]!.sent.every((line) => Buffer.byteLength(line) < 4096),
    "the sandbox PTY caps one canonical line at 4096 bytes");
});

class FakeSocket implements SandboxTerminalSocket {
  readonly sent: string[] = [];
  onSend: ((line: string) => void) | undefined;
  private openHandler: (() => void) | undefined;
  private messageHandler: ((data: string | ArrayBuffer) => void) | undefined;
  private closeHandler: (() => void) | undefined;
  private errorHandler: ((error: unknown) => void) | undefined;

  onOpen(handler: () => void): void {
    this.openHandler = handler;
  }

  onMessage(handler: (data: string | ArrayBuffer) => void): void {
    this.messageHandler = handler;
  }

  onClose(handler: () => void): void {
    this.closeHandler = handler;
  }

  onError(handler: (error: unknown) => void): void {
    this.errorHandler = handler;
  }

  send(data: string): void {
    this.sent.push(data);
    this.onSend?.(data);
  }

  close(): void {
    this.closeHandler?.();
  }

  open(): void {
    this.openHandler?.();
  }

  emit(data: string): void {
    this.messageHandler?.(data);
  }
}

/** Decodes the inline base64 staging of the launch burst back into cmd.sh. */
function decodeStagedScript(lines: string[]): string {
  const encoded = lines
    .flatMap((line) => [...line.matchAll(/printf '%s' '([A-Za-z0-9+/=]+)'/g)])
    .map((match) => match[1]!)
    .join("");
  return Buffer.from(encoded, "base64").toString("utf8");
}

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
