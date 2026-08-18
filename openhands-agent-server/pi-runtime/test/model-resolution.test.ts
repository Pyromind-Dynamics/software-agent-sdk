import assert from "node:assert/strict";
import test from "node:test";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";
import { resolveConfiguredModel } from "../src/agent-runtime.ts";

test("derives unknown compatible model ids from a known provider", async () => {
  const credentials = new InMemoryCredentialStore();
  await credentials.modify("deepseek", async () => ({ type: "api_key", key: "test" }));
  const models = builtinModels({ credentials });

  const model = resolveConfiguredModel(models, {
    provider: "deepseek",
    modelId: "deepseek-v4-flash-0731",
    baseUrl: "https://openrouter.example/v1",
  });

  assert.equal(model.id, "deepseek-v4-flash-0731");
  assert.equal(model.name, "deepseek-v4-flash-0731");
  assert.equal(model.api, "openai-completions");
  assert.equal(model.baseUrl, "https://openrouter.example/v1");
});

test("still rejects unknown models without an explicit compatible endpoint", () => {
  const models = builtinModels({ credentials: new InMemoryCredentialStore() });

  assert.throws(
    () => resolveConfiguredModel(models, {
      provider: "deepseek",
      modelId: "not-in-the-catalog",
      baseUrl: undefined,
    }),
    /Unknown Pi model/,
  );
});
