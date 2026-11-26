import assert from "node:assert/strict";
import test from "node:test";
import type { Context } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { createPiModelRuntime } from "../src/pi-model.js";

for (const maxTokens of [undefined, 2048, 8192]) {
  const reasoning = maxTokens === undefined ? undefined : "high";
  test(`custom endpoint preserves mandatory reasoning (${reasoning ?? "default"}) with output limit ${maxTokens ?? "unset"}`, async () => {
    const { modelRuntime, model } = await createPiModelRuntime({
      provider: "openrouter",
      modelId: "z-ai/glm-5.3-flash",
      apiKey: "test-key",
      baseUrl: "https://openrouter.ai/api/v1",
    });
    let payload: Record<string, unknown> | undefined;
    const answer = await modelRuntime.streamSimple(model, {
      messages: [{ role: "user", content: "你好", timestamp: Date.now() }],
    }, {
      reasoning,
      maxTokens,
      fetch: async (_url, init) => {
        payload = JSON.parse(String(init?.body));
        if ((payload?.reasoning as { effort?: string } | undefined)?.effort === "none") {
          return Response.json({ error: { message: "Reasoning is mandatory for this endpoint and cannot be disabled." } }, { status: 400 });
        }
        return new Response('data: {"choices":[{"index":0,"delta":{"content":"你好！"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', {
          headers: { "content-type": "text/event-stream" },
        });
      },
    }).result();
    assert.equal(answer.stopReason, "stop", answer.errorMessage);
    assert.deepEqual(answer.content, [{ type: "text", text: "你好！" }]);
    assert.equal(model.thinkingLevelMap?.off, null);
    assert.deepEqual(payload?.reasoning, reasoning ? { effort: "high" } : undefined);
    assert.equal(payload?.max_tokens ?? payload?.max_completion_tokens, maxTokens);
  });
}

test("custom endpoint registration retains non-reasoning models", async () => {
  const { model } = await createPiModelRuntime({
    provider: "openai",
    modelId: "gpt-4o",
    apiKey: "test-key",
    baseUrl: "http://localhost:8000/v1",
  });
  assert.equal(model.reasoning, false);
  assert.equal(model.api, "openai-completions");
});

for (const endpoint of ["https://openrouter.ai/api/v1", "https://api.openai.com/v1"]) {
  test(`output cap reaches the request payload for ${endpoint}`, async (context) => {
    const previous = process.env.PYROMIND_PI_MAX_OUTPUT_TOKENS;
    process.env.PYROMIND_PI_MAX_OUTPUT_TOKENS = "1024";
    context.after(() => {
      if (previous === undefined) delete process.env.PYROMIND_PI_MAX_OUTPUT_TOKENS;
      else process.env.PYROMIND_PI_MAX_OUTPUT_TOKENS = previous;
    });
    const { modelRuntime, model } = await createPiModelRuntime({
      provider: "openai", modelId: "gpt-4o", apiKey: "sk-test",
      baseUrl: endpoint, api: "openai-completions", contextWindow: 128_000,
    });
    assert.equal(model.contextWindow, 128_000);
    assert.equal(model.maxTokens, 1024);
    for (const requested of [undefined, 512, 131_072]) {
      let payload: Record<string, unknown> | undefined;
      const answer = await modelRuntime.streamSimple(model, {
        messages: [{ role: "user", content: "hello", timestamp: Date.now() }],
      }, {
        maxTokens: requested,
        fetch: async (_url, init) => {
          payload = JSON.parse(String(init?.body));
          const response = endpoint.includes("api.openai.com")
            ? 'event: response.completed\ndata: {"type":"response.completed","response":{"id":"test","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1}}}\n\n'
            : 'data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';
          return new Response(response, {
            headers: { "content-type": "text/event-stream" },
          });
        },
      }).result();
      assert.equal(answer.stopReason, "stop", answer.errorMessage);
      assert.equal(payload?.max_tokens ?? payload?.max_completion_tokens ?? payload?.max_output_tokens, requested === 512 ? 512 : 1024);
    }
  });
}

test("invalid output caps fail before a model request", async (context) => {
  const previous = process.env.PYROMIND_PI_MAX_OUTPUT_TOKENS;
  context.after(() => {
    if (previous === undefined) delete process.env.PYROMIND_PI_MAX_OUTPUT_TOKENS;
    else process.env.PYROMIND_PI_MAX_OUTPUT_TOKENS = previous;
  });
  for (const value of ["0", "-1", "1.5", "NaN", "Infinity", "9007199254740992"]) {
    process.env.PYROMIND_PI_MAX_OUTPUT_TOKENS = value;
    await assert.rejects(createPiModelRuntime({
      provider: "openai", modelId: "gpt-4o", apiKey: "test-key",
    }), /PYROMIND_PI_MAX_OUTPUT_TOKENS/);
  }
});

test("insufficient credits remain a model error without an explicit output cap", async () => {
  const { modelRuntime, model } = await createPiModelRuntime({
    provider: "openrouter", modelId: "z-ai/glm-5.3-flash", apiKey: "test-key",
    baseUrl: "https://openrouter.ai/api/v1",
  });
  let requests = 0;
  const answer = await modelRuntime.streamSimple(model, {
    messages: [{ role: "user", content: "hello", timestamp: Date.now() }],
  }, { maxRetries: 0, fetch: async () => {
    requests++;
    return Response.json({ error: { message: "Insufficient credits", code: 402 } }, { status: 402 });
  } }).result();
  assert.equal(answer.stopReason, "error");
  assert.match(answer.errorMessage ?? "", /402/);
  assert.equal(requests, 1);
});

test("DeepSeek gateway replays reasoning across consecutive tool calls", async () => {
  const { modelRuntime, model } = await createPiModelRuntime({
    provider: "openai", modelId: "deepseek/deepseek-v4-flash-0731",
    apiKey: "test-key", baseUrl: "http://gateway.invalid/v1",
  });
  assert.equal(model.contextWindow, 1_000_000);
  assert.equal(model.maxTokens, 384_000);
  const context: Context = {
    messages: [{ role: "user", content: "Check both records.", timestamp: 1 }],
    tools: [{ name: "lookup", description: "Look up a record", parameters: Type.Object({}) }],
  };
  for (const [index, maxTokens] of [undefined, 300_000, 500_000].entries()) {
    const answer = await modelRuntime.streamSimple(model, context, {
      ...(index > 0 ? { reasoning: "xhigh" as const } : {}), maxTokens, maxRetries: 0,
      fetch: async (_url, init) => {
        const body = JSON.parse(String(init?.body));
        assert.deepEqual(body.thinking, { type: "enabled" });
        assert.equal(body.reasoning_effort, index === 0 ? "high" : "max");
        assert.equal(body.max_tokens, [256_000, 300_000, 384_000][index]);
        const history = body.messages.filter((message: { role: string }) => message.role === "assistant");
        assert.deepEqual(history.map((message: { reasoning_content?: string }) => message.reasoning_content),
          ["look up the first record", ""].slice(0, index));
        const delta = index === 2 ? { content: "done" } : {
          ...(index === 0 ? { reasoning_content: "look up the first record" } : {}),
          tool_calls: [{ index: 0, id: `call-${index}`, type: "function", function: { name: "lookup", arguments: "{}" } }],
        };
        const event = { choices: [{ index: 0, delta, finish_reason: index === 2 ? "stop" : "tool_calls" }] };
        return new Response(`data: ${JSON.stringify(event)}\n\ndata: [DONE]\n\n`, {
          headers: { "content-type": "text/event-stream" },
        });
      },
    }).result();
    assert.equal(answer.stopReason, index === 2 ? "stop" : "toolUse", answer.errorMessage);
    if (index < 2) context.messages.push(answer, {
      role: "toolResult", toolCallId: `call-${index}`, toolName: "lookup",
      content: [{ type: "text", text: "found" }], isError: false, timestamp: 2 + index,
    });
  }
});

test("DeepSeek routes retain the selected endpoint, model identity and modalities", async () => {
  for (const config of [
    { provider: "deepseek", modelId: "deepseek-v4-pro" },
    { provider: "deepseek", modelId: "deepseek-v4-flash" },
    { provider: "openai", modelId: "deepseek-v4.1-flash-1017", baseUrl: "http://gateway.invalid/v1" },
    { provider: "openrouter", modelId: "deepseek/deepseek-v4-flash", baseUrl: "https://openrouter.ai/api/v1" },
  ]) {
    const { model } = await createPiModelRuntime({ ...config, apiKey: "test-key" });
    assert.equal(model.id, config.modelId);
    assert.equal(model.baseUrl, config.baseUrl ?? "https://api.deepseek.com");
    assert.equal(model.compat?.thinkingFormat, config.provider === "openrouter" ? "openrouter" : "deepseek");
    assert.equal(model.compat?.requiresReasoningContentOnAssistantMessages, true);
    assert.deepEqual(model.input, config.modelId.includes("v4.1") ? ["text", "image"] : ["text"]);
  }
});
