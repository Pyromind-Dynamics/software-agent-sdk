import assert from "node:assert/strict";
import test from "node:test";
import { modelRetrySettings } from "../src/pi-session.js";

test("model retries default to a small budget on both layers", () => {
  assert.deepEqual(modelRetrySettings({}), {
    enabled: true,
    maxRetries: 2,
    baseDelayMs: 2_000,
    provider: { maxRetries: 2 },
  });
});

test("model retries come from the environment when configured", () => {
  assert.deepEqual(modelRetrySettings({
    PYROMIND_PI_MODEL_MAX_RETRIES: " 4 ",
    PYROMIND_PI_MODEL_RETRY_BASE_DELAY_MS: "500",
  }), {
    enabled: true,
    maxRetries: 4,
    baseDelayMs: 500,
    provider: { maxRetries: 4 },
  });
  assert.equal(modelRetrySettings({ PYROMIND_PI_MODEL_MAX_RETRIES: "0" }).enabled, false);
  assert.throws(() => modelRetrySettings({ PYROMIND_PI_MODEL_MAX_RETRIES: "many" }),
    /must be a non-negative integer/);
  assert.throws(() => modelRetrySettings({ PYROMIND_PI_MODEL_RETRY_BASE_DELAY_MS: "-1" }),
    /must be a non-negative integer/);
});
