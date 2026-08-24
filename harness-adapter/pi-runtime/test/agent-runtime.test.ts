import assert from "node:assert/strict";
import test from "node:test";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { Api, AssistantMessage, Model, Models } from "@earendil-works/pi-ai";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { PiAgentRuntime } from "../src/agent-runtime.js";
import { resolveModel } from "../src/pi-model.js";
import { normalizePiOutcome } from "../src/pi-outcome.js";
import type { JsonObject, JsonValue, RunnerEvent } from "../src/protocol.js";
import type { JsonlRpcPeer } from "../src/rpc-peer.js";

function model(id: string, api: Api = "openai-responses"): Model<Api> {
  return {
    id,
    name: id,
    api,
    provider: "openai",
    baseUrl: "https://api.openai.com/v1",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128_000,
    maxTokens: 16_384,
  };
}

function models(catalog: Model<Api> | undefined, templates: Model<Api>[]): Models {
  return {
    getModel: () => catalog,
    getModels: () => templates,
  } as unknown as Models;
}

test("unknown model on a custom OpenAI-compatible URL defaults to chat completions", () => {
  const resolved = resolveModel(models(undefined, [model("gpt-5")]), {
    provider: "openai",
    modelId: "deepseek-v4-flash-0731",
    baseUrl: "http://localhost:8000/v1",
    contextWindow: 200_000,
  });

  assert.equal(resolved.api, "openai-completions");
  assert.equal(resolved.provider, "openai");
  assert.equal(resolved.id, "deepseek-v4-flash-0731");
  assert.equal(resolved.baseUrl, "http://localhost:8000/v1");
  assert.equal(resolved.contextWindow, 200_000);
  assert.equal(resolved.maxTokens, 200_000);
  assert.deepEqual(resolved.cost, { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
});

test("explicit API overrides catalog and fallback protocols", () => {
  const catalog = model("gpt-5", "openai-responses");
  assert.equal(resolveModel(models(catalog, [catalog]), {
    provider: "openai",
    modelId: "gpt-5",
    api: "openai-completions",
  }).api, "openai-completions");

  assert.equal(resolveModel(models(undefined, [catalog]), {
    provider: "openai",
    modelId: "custom-responses-model",
    baseUrl: "http://localhost:8000/v1",
    api: "openai-responses",
  }).api, "openai-responses");
});

test("legacy catalog sessions on custom URLs also default to chat completions", () => {
  const catalog = model("custom-model", "openai-responses");
  const resolved = resolveModel(models(catalog, [catalog]), {
    provider: "openai",
    modelId: "custom-model",
    baseUrl: "http://localhost:8000/v1",
  });

  assert.equal(resolved.api, "openai-completions");
});

test("official OpenAI URL preserves the catalog template protocol", () => {
  const resolved = resolveModel(models(undefined, [model("gpt-5")]), {
    provider: "openai",
    modelId: "future-openai-model",
    baseUrl: "https://api.openai.com/v1",
  });

  assert.equal(resolved.api, "openai-responses");
});

test("unknown model without a base URL remains an error", () => {
  assert.throws(
    () => resolveModel(models(undefined, [model("gpt-5")]), {
      provider: "openai",
      modelId: "unknown-model",
    }),
    /unknown Pi model: openai\/unknown-model/,
  );
});

test("known models retain catalog metadata while accepting an explicit context window", () => {
  const catalog = model("known-model");
  const resolved = resolveModel(models(catalog, [catalog]), {
    provider: "openai",
    modelId: "known-model",
    contextWindow: 200_000,
  });

  assert.equal(resolved.contextWindow, 200_000);
  assert.equal(resolved.maxTokens, 16_384);
});

test("final Pi stop reasons normalize without treating length as success", () => {
  assert.deepEqual(normalizePiOutcome([assistant("stop")]), {
    status: "completed", stop_reason: "stop",
  });
  assert.deepEqual(normalizePiOutcome([assistant("toolUse")]), {
    status: "completed", stop_reason: "toolUse",
  });
  assert.deepEqual(normalizePiOutcome([assistant("length")]), {
    status: "failed",
    stop_reason: "length",
    error_code: "output_truncated",
    message: "Model output was truncated",
  });
  assert.equal(normalizePiOutcome([assistant("aborted")]).status, "cancelled");
  assert.equal(normalizePiOutcome([assistant("deferred")]).status, "suspended");
  assert.equal(normalizePiOutcome([assistant("error")]).status, "failed");
  assert.equal(
    normalizePiOutcome(
      [assistant("stop")],
      assistant("length") as AssistantMessage,
    ).error_code,
    "output_truncated",
  );
});

test("idle external notification triggers an independent hidden Pi turn", async () => {
  const events: RunnerEvent[] = [];
  const peer = {
    emit: (event: RunnerEvent) => events.push(event),
    request: async (): Promise<JsonValue> => ({}),
  } as unknown as JsonlRpcPeer;
  const runtime = new PiAgentRuntime(peer);
  const calls: Array<{ message: unknown; options: unknown }> = [];
  const session = {
    isStreaming: false,
    messages: [assistant("stop")],
    sendCustomMessage: async (message: unknown, options: unknown) => {
      calls.push({ message, options });
    },
  } as unknown as AgentSession;
  const state = runtime as unknown as {
    session: AgentSession;
    sessionId: string;
  };
  state.session = session;
  state.sessionId = "session-1";

  const result = await runtime.handle("notify", {
    run_id: "callback:task-1:succeeded",
    content: "<system_reminder>done</system_reminder>",
    details: { task_id: "task-1" } as JsonObject,
  });
  await new Promise((resolve) => setImmediate(resolve));

  assert.deepEqual(result, { accepted: true, queued: false });
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0]!.options, { triggerTurn: true });
  assert.equal(events.at(-1)?.kind, "run.finished");
});

function assistant(stopReason: AssistantMessage["stopReason"]): AgentMessage {
  return {
    role: "assistant",
    content: [],
    api: "openai-completions",
    provider: "openai",
    model: "test-model",
    stopReason,
    usage: {
      input: 0, output: 0, cacheRead: 0, cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    timestamp: Date.now(),
  } as AssistantMessage;
}
