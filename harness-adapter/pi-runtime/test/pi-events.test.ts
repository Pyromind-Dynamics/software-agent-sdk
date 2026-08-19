import assert from "node:assert/strict";
import test from "node:test";
import { sanitizeJson } from "../src/pi-events.js";

test("event payloads redact credentials and normalize unsupported details", () => {
  assert.deepEqual(sanitizeJson({ cookie: "secret", nested: { api_key: "key" } }), { cookie: "[REDACTED]", nested: { api_key: "[REDACTED]" } });
});
