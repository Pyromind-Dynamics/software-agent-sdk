import assert from "node:assert/strict";
import test from "node:test";
import { deliverStage } from "../src/stage-delivery.js";
import { fixture, frame, text } from "./fixtures/no-spin.js";

test("stage child reads evidence and submits with its own tool set", async t => {
  const calls: string[] = [];
  const { config, peer } = await fixture(t, (body, response) => {
    const next = ["read_evidence", "submit_report"][calls.length];
    if (!next) return text(response, "Delivered");
    assert.deepEqual(body.tools.map((v: any) => v.function.name).sort(),
      ["list_evidence", "read_evidence", "submit_report"]);
    const args = next === "read_evidence" ? { path: "/workspace/metrics.json" }
      : { report: { schema_version: 1 }, outputs: {} };
    response.writeHead(200, { "content-type": "text/event-stream" });
    frame(response, { role: "assistant", tool_calls: [{ index: 0, id: next, type: "function",
      function: { name: next, arguments: JSON.stringify(args) } }] });
    frame(response, {}, "tool_calls");
    response.end("data: [DONE]\n\n");
  });
  peer.request = async (name) => {
    calls.push(name);
    return name === "submit_report" ? { accepted: true } : { text: "score=0.5" };
  };
  assert.deepEqual(await deliverStage({ model: config.model, role: "train_analysis", evidence: {} }, peer), { accepted: true });
  assert.deepEqual(calls, ["read_evidence", "submit_report"]);
});
