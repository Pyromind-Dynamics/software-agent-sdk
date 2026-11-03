import assert from "node:assert/strict";
import test from "node:test";
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { InMemorySpanExporter, NodeTracerProvider, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-node";
import { setLangfuseTracerProvider, startObservation } from "@langfuse/tracing";
import {
  ObservationRecorder,
  classifyHookResult,
  redact,
} from "../src/observability/recorder.js";
import { recordFinalProviderPayload } from "../src/pi-session.js";

const exporter = new InMemorySpanExporter();
setLangfuseTracerProvider(new NodeTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] }));

test("tracing stays disabled without an explicit endpoint", async () => {
  const directory = await mkdtemp(join(tmpdir(), "pi-langfuse-config-"));
  const env = { ...process.env, PI_CODING_AGENT_DIR: directory,
    LANGFUSE_PUBLIC_KEY: "local-test-public", LANGFUSE_SECRET_KEY: "local-test-secret" };
  delete env.LANGFUSE_BASE_URL;
  delete env.LANGFUSE_HOST;
  delete env.LANGFUSE_TRACING_ENABLED;
  const module = new URL("../src/observability/langfuse-plugin.ts", import.meta.url).href;
  execFileSync(process.execPath, ["--import", "tsx", "--input-type=module", "-e",
    `const {loadConfig}=await import(${JSON.stringify(module)}); if(loadConfig()!==undefined) process.exit(1);`,
  ], { env, stdio: "pipe" });
});

async function recorder(id: string, directory?: string) {
  return new ObservationRecorder({ conversationId: id,
    directory: directory ?? await mkdtemp(join(tmpdir(), "pi-observability-")),
    secrets: ["private-model-credential"], snapshot: { harness: "pi" },
  });
}

test("observation copies redact credentials without changing execution payload", () => {
  const payload = { messages: [{ content: "Bearer secret-token-1234567890 and private-model-credential" }],
    authorization: "private", api_key: "private", nested: { cookie: "private" }, model: "test" };
  const original = JSON.stringify(payload);
  const clean = JSON.stringify(redact(payload, ["private-model-credential"]));
  assert.equal(JSON.stringify(payload), original);
  for (const secret of ["secret-token-1234567890", "private-model-credential", '"private"']) assert.ok(!clean.includes(secret));
  assert.ok(clean.includes('"model":"test"'));
});

test("redaction keeps tool schema property definitions comparable", () => {
  const tools = [{
    type: "object",
    properties: {
      api_key: { type: "string", description: "credential for the gateway" },
      authorization: { type: "string" },
    },
  }];
  const clean = redact({ tools, api_key: "actual-secret-value" }) as {
    tools: unknown;
    api_key: string;
  };
  assert.deepEqual(clean.tools, tools);
  assert.equal(clean.api_key, "[REDACTED]");
});

test("hook results classify no-op, modification and blocking", () => {
  assert.equal(classifyHookResult(undefined), "no_operation");
  assert.equal(classifyHookResult(null), "no_operation");
  assert.equal(classifyHookResult({ isError: true }), "modified");
  assert.equal(classifyHookResult({ block: true, reason: "repeat" }), "blocked");
});

test("hook wrappers preserve results, execution count and original exceptions", async () => {
  const capture = await recorder("hooks");
  const root = startObservation("hook-test"); capture.attach(root);
  let calls = 0;
  assert.equal(capture.hook("test", "tool_call", {}, "id", () => { calls++; return undefined; }), undefined);
  const result = { block: true, reason: "unchanged" };
  assert.equal(capture.hook("test", "tool_call", {}, "id", () => { calls++; return result; }), result);
  const failure = new Error("original hook failure");
  assert.throws(() => capture.hook("test", "tool_result", {}, "id", () => { calls++; throw failure; }), (error) => error === failure);
  assert.equal(calls, 3); await capture.finish(); root.end();
});

test("recorders save immutable redacted requests and isolate conversations", async () => {
  const directoryA = await mkdtemp(join(tmpdir(), "pi-evidence-a-"));
  const directoryB = await mkdtemp(join(tmpdir(), "pi-evidence-b-"));
  const a = await recorder("a", directoryA), b = await recorder("b", directoryB);
  const rootA = startObservation("root-a"), rootB = startObservation("root-b");
  a.begin("run-a"); b.begin("run-b"); a.attach(rootA); b.attach(rootB);
  a.generation = rootA.startObservation("model-a", {}, { asType: "generation" });
  b.generation = rootB.startObservation("model-b", {}, { asType: "generation" });
  const request = { messages: [{ content: "private-model-credential " + "x".repeat(25000) }] };
  a.request(request); b.request({ messages: [{ content: "conversation-b" }] });
  request.messages[0].content = "changed-after-request";
  await Promise.all([a.finish(), b.finish()]);
  const contentsA = await Promise.all((await readdir(directoryA)).map(p => readFile(join(directoryA, p), "utf8")));
  const contentsB = await Promise.all((await readdir(directoryB)).map(p => readFile(join(directoryB, p), "utf8")));
  assert.ok(contentsA.some(text => text.includes("x".repeat(25000))));
  assert.ok(contentsA.every(text => !text.includes("private-model-credential") && !text.includes("changed-after-request")));
  assert.ok(contentsB.some(text => text.includes("conversation-b")));
  assert.ok(contentsA.every(text => !text.includes("conversation-b")));
  rootA.end(); rootB.end();
});

test("content storage failure does not fail execution", async () => {
  const capture = await recorder("failure", "\u0000");
  capture.attach(startObservation("storage-failure"));
  capture.fact("unavailable", { value: "safe" });
  await capture.finish();
});

test("final request is recorded after later payload transformations", async () => {
  const directory = await mkdtemp(join(tmpdir(), "pi-transform-"));
  const capture = new ObservationRecorder({
    conversationId: "transform",
    directory,
    secrets: [],
    snapshot: {},
  });
  const root = startObservation("transform-root");
  capture.begin("run");
  capture.attach(root);
  capture.generation = root.startObservation("model", {}, { asType: "generation" });

  type Runtime = Parameters<typeof recordFinalProviderPayload>[0];
  let captured: { onPayload?: (payload: unknown, model: unknown) => Promise<unknown> };
  const runtime = {
    streamSimple: (
      _model: unknown,
      _context: unknown,
      options: { onPayload?: (payload: unknown, model: unknown) => Promise<unknown> },
    ) => {
      captured = options;
      return "sent";
    },
  } as unknown as Runtime;
  recordFinalProviderPayload(runtime, capture);

  const result = (runtime as unknown as {
    streamSimple: (model: unknown, context: unknown, options: unknown) => unknown;
  }).streamSimple(null, null, {
    onPayload: async (payload: { messages: unknown[] }) => ({
      ...payload,
      messages: [{ role: "user", content: "post-processed" }],
    }),
  });
  assert.equal(result, "sent");
  await captured!.onPayload!({ messages: [{ role: "user", content: "original" }] }, null);
  await capture.finish();
  root.end();

  const contents = await Promise.all(
    (await readdir(directory)).map((name) => readFile(join(directory, name), "utf8")),
  );
  assert.ok(contents.some((text) => text.includes("post-processed")));
  assert.ok(contents.every((text) => !text.includes('"original"')));
});

test("credential text is redacted in tool messages, JSON and header arrays", () => {
  const payload = {
    messages: [{ content: 'Cookie: session=DEMO_COOKIE_12345\nAuthorization: Basic ZGVtbzpwYXNzd29yZA==\n{"auth_token":"DEMO_AUTH_TOKEN"}' }],
    authorization: ["Basic ZGVtbzpwYXNzd29yZA=="],
  };
  const original = JSON.stringify(payload);
  const clean = JSON.stringify(redact(payload));
  for (const secret of ["DEMO_COOKIE_12345", "ZGVtbzpwYXNzd29yZA==", "DEMO_AUTH_TOKEN"]) assert.ok(!clean.includes(secret));
  assert.equal(JSON.stringify(payload), original);
});

test("large fact input is bounded and refused evidence marks the turn partial", async () => {
  const capture = await recorder("large-fact");
  const root = startObservation("large-root");
  capture.begin("large-run"); capture.attach(root);
  const text = "x".repeat(2 * 1024 * 1024 + 1);
  const preview = capture.preview({ content: text }) as { preview: { content: string }; truncated: boolean };
  assert.equal(preview.truncated, true);
  assert.ok(preview.preview.content.length <= 20_050);
  assert.ok(preview.preview.content.endsWith("[truncated]"));
  capture.fact("oversized", { content: text });
  await capture.finish();
  const span = root.otelSpan as unknown as { attributes: Record<string, string> };
  assert.equal(span.attributes["langfuse.observation.metadata.capture_state"], "partial");
  assert.ok(span.attributes["langfuse.observation.metadata.gap_reasons"].includes("content_budget"));
  root.end();
});

test("tool evidence and normal hook checks stay on their tool while snapshots remain children", async () => {
  const directory = await mkdtemp(join(tmpdir(), "pi-tool-evidence-"));
  const capture = new ObservationRecorder({
    conversationId: "tool-evidence", directory, snapshot: {}, secrets: [],
    readFile: async () => Buffer.from("original file"),
  });
  const root = startObservation("evidence-root");
  capture.begin("run"); capture.attach(root);
  capture.hook("no-progress-guard", "tool_call", { prior_repeats: 0 }, "read-id", () => undefined);
  const tool = root.startObservation("Tool: read", {}, { asType: "tool" });
  capture.toolStarted("read-id", tool);
  await capture.captureFile("public_data/input.txt", "initial", "read-id");
  capture.hook("tool-result-status", "tool_result", {}, "read-id", () => undefined);
  capture.toolResult("read", { path: "public_data/input.txt" }, [{ type: "text", text: "original file" }], false, "read-id");
  capture.toolEnded("read-id"); capture.endObservation(tool);
  await capture.finish(); root.end();
  const spans = exporter.getFinishedSpans().filter(span => span.spanContext().traceId === root.otelSpan.spanContext().traceId);
  assert.deepEqual(spans.map(span => span.name).sort(), ["Artifact snapshot: before", "Harness snapshot", "Tool: read", "Turn outcome", "evidence-root"].sort());
  const snapshot = spans.find(span => span.name === "Artifact snapshot: before")!;
  assert.equal(snapshot.parentSpanContext?.spanId, tool.otelSpan.spanContext().spanId);
  const recordedTool = spans.find(span => span.name === "Tool: read")!;
  for (const field of ["file_evidence", "hook_checks_evidence"]) {
    const evidence = JSON.parse(String(recordedTool.attributes[`langfuse.observation.metadata.${field}`]));
    assert.equal(evidence.status, "saved");
    const saved = JSON.parse(await readFile(join(directory, `${evidence.sha256}.json`), "utf8"));
    if (field === "file_evidence") assert.equal(saved.returned_content[0].text, "original file");
    else assert.equal(saved.hook_checks.length, 2);
  }
});

test("model records expose message structure, references and skill matches without helper spans", async () => {
  const directory = await mkdtemp(join(tmpdir(), "pi-request-evidence-"));
  const capture = await recorder("request", directory);
  const root = startObservation("request-root");
  capture.begin("run"); capture.attach(root);
  const tool = root.startObservation("Tool: read", {}, { asType: "tool" });
  capture.toolStarted("skill-id", tool);
  capture.toolResult("read", { path: ".agents/skills/example/SKILL.md" }, [{ type: "text", text: "skill instructions" }], false, "skill-id");
  capture.toolEnded("skill-id"); capture.endObservation(tool);
  const generation = root.startObservation("LLM Call", {}, { asType: "generation" });
  capture.generation = generation;
  const payload = { model: "example", messages: [{ role: "system", content: "skill instructions" }], tools: [{ name: "read" }] };
  capture.request(payload); capture.endObservation(generation);
  await capture.finish(); root.end();
  const spans = exporter.getFinishedSpans().filter(span => span.spanContext().traceId === root.otelSpan.spanContext().traceId);
  assert.ok(!spans.some(span => ["Evidence storage", "Skill read", "Skill request evidence", "File operation evidence"].includes(span.name)));
  const recorded = spans.find(span => span.name === "LLM Call")!;
  assert.deepEqual(JSON.parse(String(recorded.attributes["langfuse.observation.input"])), payload.messages);
  const matches = JSON.parse(String(recorded.attributes["langfuse.observation.metadata.skill_context"]));
  assert.equal(matches[0].returned_text_present, true);
  assert.equal(matches[0].read_evidence.tool_call_id, "skill-id");
  const evidence = JSON.parse(String(recorded.attributes["langfuse.observation.metadata.request_evidence"]));
  assert.equal(evidence.status, "saved");
  assert.deepEqual(JSON.parse(await readFile(join(directory, `${evidence.sha256}.json`), "utf8")), payload);
});

test("artifact snapshots distinguish missing files and unsupported binary content", async () => {
  const directory = await mkdtemp(join(tmpdir(), "pi-artifact-"));
  const capture = new ObservationRecorder({
    conversationId: "artifacts", directory, secrets: [], snapshot: {},
    readFile: async (path) => {
      if (path.endsWith("missing")) {
        throw Object.assign(new Error("missing"), { code: "ENOENT" });
      }
      return Buffer.from([0xff, 0x00]);
    },
  });
  const root = startObservation("artifacts");
  capture.begin("run"); capture.attach(root);
  await capture.captureFile("public_data/missing", "final", "one");
  await capture.captureFile("public_data/binary", "final", "two");
  await capture.finish(); root.end();
  const saved = await Promise.all((await readdir(directory)).map(p => readFile(join(directory, p), "utf8")));
  assert.ok(saved.some(text => text.includes('"exists":false')));
  assert.ok(saved.some(text => text.includes('"content_omitted":"unsupported_binary_or_partial_utf8"')));
  assert.ok(saved.some(text => text.includes('"gap_reasons":["artifact_content_unsupported"]')));
});
