import assert from "node:assert/strict";
import test from "node:test";
import { createPiModelRuntime } from "../src/pi-model.js";

for (const reasoning of [undefined, "high"] as const) {
  test(`custom endpoint preserves mandatory reasoning (${reasoning ?? "default"}) in greeting requests`, async () => {
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
      ...(reasoning ? { maxTokens: 2048 } : {}),
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
    assert.equal(payload?.max_tokens ?? payload?.max_completion_tokens, reasoning ? 2048 : undefined);
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
