import assert from "node:assert/strict";
import { mkdtemp, mkdir, realpath } from "node:fs/promises";
import { createServer } from "node:http";
import { join } from "node:path";
import test from "node:test";
import { createPiSession } from "../src/pi-session.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";

test("background subagents do not hold the parent turn open", { timeout: 15_000 }, async (t) => {
  const root = await realpath(await mkdtemp("/tmp/pi-subagent-settlement-"));
  await mkdir(join(root, "pi", "terminal-output"), { recursive: true });
  await mkdir(join(root, "public_data"), { recursive: true });
  await mkdir(join(root, "skills"), { recursive: true });

  let childStarted!: () => void;
  const started = new Promise<void>((resolve) => { childStarted = resolve; });
  let releaseChild!: () => void;
  const released = new Promise<void>((resolve) => { releaseChild = resolve; });
  const childSystemPrompts: string[] = [];

  const server = createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    const payload = JSON.parse(body);
    const isParent = payload.tools?.some((tool: any) => tool.function.name === "subagent");
    if (!isParent) {
      childSystemPrompts.push(String(payload.messages[0]?.content ?? ""));
      childStarted();
      await released;
    }
    const toolResults = payload.messages.filter((message: any) => message.role === "tool");
    const delta = isParent && toolResults.length === 0
      ? {
          tool_calls: [{
            index: 0,
            id: "delegate-1",
            type: "function",
            function: {
              name: "subagent",
              arguments: JSON.stringify({
                agent: "search",
                task: "Run until the test releases the child",
                async: true,
              }),
            },
          }],
        }
      : { content: isParent ? "Parent finished while child runs" : "Child finished" };
    response.writeHead(200, { "content-type": "text/event-stream" });
    for (const [part, reason] of [
      [{ role: "assistant" }, null],
      [delta, null],
      [{}, isParent && toolResults.length === 0 ? "tool_calls" : "stop"],
    ]) {
      response.write(`data: ${JSON.stringify({
        id: "test",
        object: "chat.completion.chunk",
        created: 1,
        model: payload.model,
        choices: [{ index: 0, delta: part, finish_reason: reason }],
      })}\n\n`);
    }
    response.end("data: [DONE]\n\n");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => {
    releaseChild();
    server.closeAllConnections();
    server.close();
  });

  const peer = {
    emit: () => {},
    request: async () => ({
      is_error: false,
      content: [{ type: "text", text: "unused" }],
    }),
  } as unknown as JsonlRpcPeer;
  const { session, subagents } = await createPiSession({
    session_id: "subagent-settlement",
    workspace_root: root,
    terminal_backend: "os-sandbox",
    session_path: join(root, "pi", "session.jsonl"),
    system_prompt: "Delegate the background task.",
    skills_directory: join(root, "skills"),
    skill_roots: [],
    managed_delegation: true,
    tools: [],
    model: {
      provider: "openai",
      id: "custom-chat-model",
      api_key: "test",
      base_url: `http://127.0.0.1:${(server.address() as { port: number }).port}/v1`,
    },
  }, peer);
  t.after(async () => {
    await subagents.stopAll();
    await session.abort();
    await session.waitForIdle();
    await session.extensionRunner.emit({ type: "session_shutdown", reason: "quit" });
    session.dispose();
    await subagents.close();
  });
  await subagents.attach(session);
  assert.deepEqual(
    await subagents.control({
      action: "authorize",
      role: "search",
      scope: { run_id: "run-123", work_id: "work-1", generation: 1 },
    }),
    { execution_ref: null, status: "authorized", result_text: "" },
  );

  const prompt = session.prompt("Delegate this and return to the user.");
  await started;
  assert.match(childSystemPrompts[0], /public_data\/experiment-runs\/run-123\//);
  assert.match(childSystemPrompts[0], /task\.inputs\.path_contract/);
  const state = await Promise.race([
    prompt.then(() => "idle"),
    new Promise<"blocked">((resolve) => setTimeout(() => resolve("blocked"), 2_000)),
  ]);
  assert.equal(state, "idle");
  assert.equal(session.isIdle, true);
});
