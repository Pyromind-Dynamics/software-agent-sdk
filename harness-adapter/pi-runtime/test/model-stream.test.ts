import assert from "node:assert/strict";
import { createServer, type ServerResponse } from "node:http";
import test from "node:test";
import { modelIdleTimeoutMs } from "../src/model-stream.js";
import { createPiModelRuntime } from "../src/pi-model.js";

test("model idle timeout has a bounded default and an explicit disable switch", () => {
  assert.equal(modelIdleTimeoutMs({}), 120_000);
  assert.equal(modelIdleTimeoutMs({ PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS: "0" }), 0);
  for (const value of ["-1", "NaN", "1.5", "2147483648"]) {
    assert.throws(() => modelIdleTimeoutMs({ PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS: value }));
  }
});

function frame(response: ServerResponse, delta: object, reason: string | null = null): void {
  response.write(`data: ${JSON.stringify({ id: "local", object: "chat.completion.chunk",
    choices: [{ index: 0, delta, finish_reason: reason }] })}\n\n`);
}

for (const scenario of ["heartbeat", "text", "thinking", "arguments", "cancel", "disabled"] as const) {
  test(`custom model stream: ${scenario}`, { timeout: 10_000 }, async (context) => {
    const previous = process.env.PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS;
    process.env.PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS = scenario === "disabled" ? "0" : "500";
    context.after(() => {
      if (previous === undefined) delete process.env.PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS;
      else process.env.PYROMIND_PI_MODEL_IDLE_TIMEOUT_MS = previous;
    });
    let requests = 0;
    let heartbeats = 0;
    const generates = scenario === "text" || scenario === "thinking" || scenario === "arguments";
    const server = createServer((request, response) => {
      request.resume();
      response.writeHead(200, { "content-type": "text/event-stream" });
      requests++;
      if (requests > 1) {
        frame(response, { role: "assistant", content: "recovered" });
        frame(response, {}, "stop");
        response.end("data: [DONE]\n\n");
        return;
      }
      frame(response, { role: "assistant", ...(!generates || scenario === "arguments"
        ? { tool_calls: [{ index: 0, id: "call-1", type: "function", function: { name: "ask_user", arguments: "" } }] }
        : {}) });
      let ticks = 0;
      const argumentsParts = ['{"questions":[', '{"id":"q",', '"prompt":"?",', '"options":[]}', "]}"];
      const timer = setInterval(() => {
        response.write(": heartbeat\n\n");
        heartbeats++;
        if (!generates) return;
        ticks++;
        if (scenario === "thinking") frame(response, { reasoning_content: "considering " });
        else if (scenario === "text") frame(response, { content: "Please choose. " });
        else frame(response, { tool_calls: [{ index: 0, function: { arguments: argumentsParts[ticks - 1] } }] });
        if (ticks === 5) {
          frame(response, {}, scenario === "arguments" ? "tool_calls" : "stop");
          clearInterval(timer);
          response.end("data: [DONE]\n\n");
        }
      }, generates ? 150 : 30);
      response.on("close", () => clearInterval(timer));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    context.after(() => { server.closeAllConnections(); server.close(); });
    const address = server.address() as { port: number };
    const { modelRuntime, model } = await createPiModelRuntime({ provider: "openai",
      modelId: "local-timeout-test", apiKey: "test", baseUrl: `http://127.0.0.1:${address.port}/v1` });
    const controller = new AbortController();
    const cancel = scenario === "cancel" || scenario === "disabled"
      ? setTimeout(() => controller.abort(), scenario === "disabled" ? 800 : 250)
      : undefined;
    context.after(() => clearTimeout(cancel));
    const transcript = { messages: [{ role: "user" as const, content: "test", timestamp: Date.now() }] };
    const answer = await modelRuntime.streamSimple(model, transcript,
      { signal: controller.signal, timeoutMs: 100, maxRetries: 0 }).result();
    assert.ok(heartbeats > 0);
    if (scenario === "heartbeat") {
      assert.equal(answer.stopReason, "error");
      assert.match(answer.errorMessage!, /Model stream idle timeout/);
      assert.equal(controller.signal.aborted, false);
      assert.equal(requests, 1, "guard must not retry or execute a tool");
      const next = await modelRuntime.streamSimple(model, transcript).result();
      assert.equal(next.stopReason, "stop");
      assert.equal(next.content[0]?.type === "text" && next.content[0].text, "recovered");
    } else if (scenario === "arguments") {
      assert.equal(answer.stopReason, "toolUse");
      const tool = answer.content.find((block) => block.type === "toolCall");
      assert.equal(tool?.type === "toolCall" && tool.arguments.questions[0].id, "q");
    } else if (generates) {
      assert.equal(answer.stopReason, "stop");
      assert.ok(answer.content.some((block) => block.type === scenario));
    } else {
      assert.equal(answer.stopReason, "aborted");
      assert.doesNotMatch(answer.errorMessage ?? "", /idle timeout/);
    }
  });
}
