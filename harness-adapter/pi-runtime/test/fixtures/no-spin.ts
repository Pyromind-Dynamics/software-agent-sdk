import { mkdtemp, mkdir } from "node:fs/promises";
import { createServer, type ServerResponse } from "node:http";
import { join } from "node:path";
import type { TestContext } from "node:test";
import type { JsonObject, RunnerEvent } from "../../src/protocol.js";
import type { JsonlRpcPeer } from "../../src/rpc-peer.js";

export function frame(response: ServerResponse, delta: object, reason: string | null = null) {
  response.write(`data: ${JSON.stringify({ id: "guard-test", object: "chat.completion.chunk",
    choices: [{ index: 0, delta, finish_reason: reason }] })}\n\n`);
}

export function text(response: ServerResponse, content: string) {
  response.writeHead(200, { "content-type": "text/event-stream" });
  frame(response, { role: "assistant", content });
  frame(response, {}, "stop");
  response.end("data: [DONE]\n\n");
}

export function repeat(response: ServerResponse, field: string, onClose: () => void) {
  response.writeHead(200, { "content-type": "text/event-stream" });
  frame(response, { role: "assistant" });
  const timer = setInterval(() => frame(response, { [field]: "Emit. " }), 5);
  response.on("close", () => { clearInterval(timer); onClose(); });
}

export async function eventually(check: () => boolean | Promise<boolean>) {
  const deadline = Date.now() + 15_000;
  while (!await check()) {
    if (Date.now() > deadline) throw new Error("Guard integration timed out");
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

export async function fixture(t: TestContext, respond: (body: any, response: ServerResponse) => void) {
  const root = await mkdtemp("/tmp/pi-no-spin-");
  await mkdir(join(root, "skills"));
  await mkdir(join(root, "public_data"));
  const requests: any[] = [];
  const server = createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    const parsed = JSON.parse(body);
    requests.push(parsed);
    respond(parsed, response);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => { server.closeAllConnections(); server.close(); });
  const address = server.address() as { port: number };
  const config: JsonObject = {
    session_id: "guard-session", workspace_root: root, terminal_backend: "sandbox",
    session_path: join(root, "pi/session.jsonl"), skills_directory: join(root, "skills"), skill_roots: [],
    system_prompt: "Answer the user.",
    model: { provider: "openai", id: "guard-model", api_key: "test-only",
      base_url: `http://127.0.0.1:${address.port}/v1` },
    tools: [{ name: "evidence", description: "Read evidence.", input_schema: { type: "object", properties: {} } }],
  };
  const events: RunnerEvent[] = [];
  const peer = { emit: (event: RunnerEvent) => events.push(event), request: async () => ({
    is_error: false, content: [{ type: "text", text: "PRESERVED_TOOL_EVIDENCE" }],
  }) } as unknown as JsonlRpcPeer;
  return { root, requests, config, events, peer };
}

