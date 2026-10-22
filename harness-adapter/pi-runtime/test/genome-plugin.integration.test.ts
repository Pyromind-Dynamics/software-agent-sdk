import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { createServer, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { DefaultResourceLoader, SettingsManager } from "@earendil-works/pi-coding-agent";
import { genomeHostAccess } from "../src/genome-host.js";
import { PiAgentRuntime } from "../src/agent-runtime.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";
import type { JsonObject, JsonValue, RunnerEvent } from "../src/protocol.js";

test("SDK loads the package default entry, discovers its host, and retains it across reloads", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "sdk-genome-loader-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const calls: string[] = [];
  const peer = { request: async (method: string) => { calls.push(method); return { assets: [] }; } } as unknown as JsonlRpcPeer;
  const host = genomeHostAccess(peer, true);
  const loader = new DefaultResourceLoader({ cwd: root, agentDir: join(root, "agent"),
    settingsManager: SettingsManager.inMemory({}), eventBus: host.eventBus,
    additionalExtensionPaths: host.additionalExtensionPaths, noExtensions: true,
    noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true });
  for (let iteration = 0; iteration < 2; iteration++) {
    await loader.reload();
    host.assertLoaded(loader);
    const extensions = loader.getExtensions().extensions;
    assert.equal(extensions.length, 1);
    const extension = extensions[0];
    assert.ok(extension.resolvedPath.endsWith("@agentgenome/pi-extension/index.js"));
    assert.equal(extension.path.includes("inline"), false);
    assert.equal(extension.tools.size, 17);
    assert.equal(extension.handlers.has("session_start"), false, "SDK must not initialize a native service");
    assert.equal(extension.commands.has("genome"), false, "SDK must not install a native Python environment");
    await extension.tools.get("genome_list")!.definition.execute("call", {}, undefined, undefined, {} as never);
  }
  assert.deepEqual(calls, ["workflow.invoke", "workflow.invoke"]);

  const missingHostLoader = new DefaultResourceLoader({ cwd: root, agentDir: join(root, "other"),
    settingsManager: SettingsManager.inMemory({}),
    additionalExtensionPaths: host.additionalExtensionPaths, noExtensions: true,
    noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true });
  await missingHostLoader.reload();
  assert.throws(() => genomeHostAccess(peer, true).assertLoaded(missingHostLoader), /did not connect/);

  const disabled = genomeHostAccess(peer, false);
  assert.deepEqual(disabled.additionalExtensionPaths, []);
  const disabledLoader = new DefaultResourceLoader({ cwd: root, agentDir: join(root, "disabled"),
    settingsManager: SettingsManager.inMemory({}), eventBus: disabled.eventBus,
    additionalExtensionPaths: disabled.additionalExtensionPaths, noExtensions: true,
    noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true });
  await disabledLoader.reload();
  assert.equal(disabledLoader.getExtensions().extensions.length, 0);
});

test("SDK forwards a reviewed distillation plan and its confirmation", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "sdk-genome-distill-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const requests: JsonObject[] = [];
  const peer = {
    request: async (method: string, payload: JsonObject): Promise<JsonValue> => {
      assert.equal(method, "workflow.invoke");
      requests.push(payload);
      if (payload.action === "distill") {
        return {
          plan_id: "distill-plan", digest: "distill-digest", status: "planned",
          requires_confirmation: true, review: { steps: [{ step: "clean", kind: "run" }] },
        };
      }
      assert.equal(payload.action, "confirm_distill");
      return {
        capture_id: "capture-distill-plan", status: "draft",
        distill_plan: { plan_id: "distill-plan", digest: "distill-digest", status: "confirmed" },
      };
    },
  } as unknown as JsonlRpcPeer;
  const host = genomeHostAccess(peer, true);
  const loader = new DefaultResourceLoader({
    cwd: root, agentDir: join(root, "agent"),
    settingsManager: SettingsManager.inMemory({}), eventBus: host.eventBus,
    additionalExtensionPaths: host.additionalExtensionPaths, noExtensions: true,
    noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
  });
  await loader.reload();
  host.assertLoaded(loader);
  const extension = loader.getExtensions().extensions[0];
  const distillArgs = { label: "clean.csv", source_traces: [] };
  await extension.tools.get("genome_distill")!.definition.execute(
    "distill-call", distillArgs, undefined, undefined, {} as never,
  );
  await extension.tools.get("genome_confirm_distill")!.definition.execute(
    "confirm-distill-call",
    { plan_id: "distill-plan", digest: "distill-digest" },
    undefined,
    undefined,
    {} as never,
  );
  assert.deepEqual(requests.map((request) => request.action), ["distill", "confirm_distill"]);
  assert.deepEqual(
    requests.map((request) => request.request_id),
    ["distill-call", "confirm-distill-call"],
  );
  assert.deepEqual(requests[0].arguments, distillArgs);
  assert.equal(
    (requests[0].arguments as JsonObject).steps,
    undefined,
    "the first distill call must let the plugin derive candidate steps",
  );
  assert.deepEqual(requests[1].arguments, {
    plan_id: "distill-plan",
    digest: "distill-digest",
  });
});

test("a real SDK Pi turn invokes genome tools through the loaded package and existing RPC host", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "sdk-genome-turn-"));
  await mkdir(join(root, "public_data"));
  await mkdir(join(root, "skills"));
  await mkdir(join(root, "pi/terminal-output"), { recursive: true });
  const modelRequests: JsonObject[] = [], hostRequests: JsonObject[] = [], events: RunnerEvent[] = [];
  const server = createServer(async (request, response) => {
    let text = "";
    for await (const chunk of request) text += chunk;
    modelRequests.push(JSON.parse(text));
    const step = modelRequests.length;
    if (step === 1) reply(response, "genome_list", { action: "list", foo: "bar" }, "list-call");
    else if (step === 2) reply(response, "genome_prepare_run", {
      asset_id: "data-cleaning", version: "1.0.0", params: { data_file: "public_data/input.csv" },
      execution_scope: { mode: "full", description: "用户确认全量执行" },
    }, "prepare-call");
    else if (step === 3) reply(response, "genome_confirm_run", {
      plan_id: "sdk-plan", digest: "plan-digest",
    }, "confirm-call");
    else reply(response);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address() as AddressInfo;
  let finish!: () => void;
  const finished = new Promise<void>((resolve) => { finish = resolve; });
  const peer = {
    request: async (method: string, payload: JsonObject): Promise<JsonValue> => {
      assert.equal(method, "workflow.invoke");
      hostRequests.push(payload);
      if (payload.action === "list") return { assets: [{ id: "data-cleaning", version: "1.0.0" }] };
      if (payload.action === "prepare_run") return {
        plan_id: "sdk-plan", digest: "plan-digest", status: "planned", requires_confirmation: true,
      };
      assert.equal(payload.action, "confirm_run");
      return { plan: { plan_id: "sdk-plan" }, run: { id: "sdk-run", status: "queued" } };
    },
    emit: (event: RunnerEvent) => { events.push(event); if (event.kind === "run.finished") finish(); },
  } as unknown as JsonlRpcPeer;
  const runtime = new PiAgentRuntime(peer);
  t.after(async () => { server.closeAllConnections(); server.close(); await rm(root, { recursive: true, force: true }); });
  await runtime.handle("start", {
    session_id: "plugin-turn", workspace_root: root, terminal_backend: "os-sandbox",
    session_path: join(root, "pi/session.jsonl"), skills_directory: join(root, "skills"),
    skill_roots: [], tools: [], workflows_enabled: true,
    model: { provider: "openai", id: "plugin-test", api_key: "unused", base_url: `http://127.0.0.1:${address.port}/v1` },
  });
  await runtime.handle("prompt", { run_id: "user-command", content: [{ type: "text", text: "复用历史经验清洗 CSV" }] });
  let timeout: NodeJS.Timeout;
  await Promise.race([finished, new Promise<never>((_resolve, reject) => {
    timeout = setTimeout(() => reject(new Error("plugin tool turn did not finish")), 10000);
  })]).finally(() => clearTimeout(timeout));
  assert.deepEqual(hostRequests.map((request) => request.action), ["list", "prepare_run", "confirm_run"]);
  assert.deepEqual(hostRequests.map((request) => request.request_id),
    ["list-call", "prepare-call", "confirm-call"]);
  assert.deepEqual(hostRequests[0].arguments, {});
  assert.deepEqual(hostRequests[1].arguments, {
    asset_id: "data-cleaning", version: "1.0.0", params: { data_file: "public_data/input.csv" },
    execution_scope: { mode: "full", description: "用户确认全量执行" },
  });
  assert.deepEqual(hostRequests[2].arguments, { plan_id: "sdk-plan", digest: "plan-digest" });
  const schemas = modelRequests[0].tools as Array<{ function: { name: string } }>;
  assert.deepEqual(
    schemas.map((tool) => tool.function.name).filter((name) => name.startsWith("genome_")).sort(),
    ["genome_cancel", "genome_capture", "genome_capture_status",
      "genome_confirm_distill", "genome_confirm_run", "genome_distill", "genome_get",
      "genome_list", "genome_prepare_revision", "genome_prepare_run",
      "genome_replay_requeue", "genome_resume", "genome_run", "genome_status",
      "genome_step_result", "genome_takeover", "genome_trace"],
  );
  assert.equal(events.filter((event) => event.kind === "run.finished").length, 1);
  assert.match(JSON.stringify(modelRequests[3].messages), /sdk-run/);
});

function reply(response: ServerResponse, name?: string, args: JsonObject = {}, id?: string): void {
  const deltas: Array<[JsonObject, string | null]> = name ? [
    [{ role: "assistant" }, null],
    [{ tool_calls: [{ index: 0, id, type: "function", function: { name, arguments: "" } }] }, null],
    [{ tool_calls: [{ index: 0, function: { arguments: JSON.stringify(args) } }] }, null],
    [{}, "tool_calls"],
  ] : [[{ role: "assistant", content: "已提交，等待后台执行结果。" }, null], [{}, "stop"]];
  response.writeHead(200, { "content-type": "text/event-stream" });
  for (const [delta, finish_reason] of deltas) response.write(`data: ${JSON.stringify({
    id: "plugin-test", object: "chat.completion.chunk", created: 1, model: "plugin-test",
    choices: [{ index: 0, delta, finish_reason }],
  })}\n\n`);
  response.end("data: [DONE]\n\n");
}
