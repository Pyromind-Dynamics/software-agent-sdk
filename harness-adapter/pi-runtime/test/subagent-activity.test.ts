import assert from "node:assert/strict";
import test from "node:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync, symlinkSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { activityWindow, SubagentActivity } from "../src/subagent-activity.js";

test("native records become ordered messages and paired tool operations without metadata or thinking", () => {
  const result = activityWindow([
    {recordType: "message", runId: "a", timestamp: 1, message: {role: "system", content: "PRIVATE_SYSTEM"}},
    {recordType: "message", runId: "a", timestamp: 2, message: {role: "assistant", content: [
      {type: "thinking", thinking: "PRIVATE_THINKING"},
      {type: "text", text: "Reading samples"},
      {type: "toolCall", id: "read-1", name: "read", arguments: {path: "/data.jsonl"}},
    ]}},
    {recordType: "tool_start", runId: "a", toolCallId: "read-1", toolName: "read", argsPayload: '{"path":"/data.jsonl"}'},
    {recordType: "message", runId: "a", message: {role: "toolResult", toolCallId: "read-1", toolName: "read", content: [{type: "text", text: "Four samples"}]}},
    {recordType: "tool_end", runId: "a", toolCallId: "read-1", toolName: "read", isError: false},
    {recordType: "message", runId: "a", timestamp: 3, message: {role: "assistant", content: [{type: "text", text: "**Done**"}]}, cwd: "/private/metadata"},
  ]);
  const items = result.items as any[];
  assert.deepEqual(items.map(item => item.kind), ["message", "operation", "message"]);
  assert.equal(items[1].name, "read");
  assert.deepEqual(items[1].arguments, {path: "/data.jsonl"});
  assert.equal(items[1].status, "completed");
  assert.deepEqual(items[1].output, [{type: "text", text: "Four samples"}]);
  assert.equal(items[2].content[0].text, "**Done**");
  assert.doesNotMatch(JSON.stringify(result), /PRIVATE|cwd|private\/metadata|thinking/);
});

test("active tool calls retain their identity and failure results preserve errors", () => {
  const rows = [
    {recordType: "tool_start", runId: "a", toolCallId: "call", toolName: "terminal", argsPayload: '{"command":"exit 7"}'},
  ];
  const before = activityWindow(rows).items as any[];
  assert.equal(before[0].status, "running");
  const after = activityWindow([...rows, {recordType: "message", runId: "a", message: {role: "toolResult", toolCallId: "call", toolName: "terminal", isError: true, content: [{type: "text", text: "exit 7"}]}}]).items as any[];
  assert.equal(after[0].item_id, before[0].item_id);
  assert.equal(after[0].status, "failed");
  assert.equal(after[0].output[0].text, "exit 7");
});

test("structured windows keep complete recent entries and stay under the transport budget", () => {
  const rows = Array.from({length: 300}, (_, index) => ({recordType: "message", timestamp: index, message: {
    role: "assistant", content: [{type: "text", text: `${index}:` + "证据".repeat(1000)}],
  }}));
  const result = activityWindow(rows);
  assert.equal(result.truncated, true);
  assert.ok((result.items as any[]).length <= 128);
  assert.ok(Buffer.byteLength(JSON.stringify(result)) < 66000);
  assert.match((result.items as any[]).at(-1).content[0].text, /^299:/);
});

test("reader trusts plugin ownership, confines artifact paths and tolerates a partial final record", () => {
  const root = mkdtempSync(join(tmpdir(), "child-activity-"));
  try {
    const state = join(root, "subagents"), artifacts = join(root, "artifacts");
    mkdirSync(join(state, "async-subagent-runs", "child"), {recursive: true});
    mkdirSync(join(state, "sessions"), {recursive: true});
    mkdirSync(artifacts);
    const sessionFile = join(state, "sessions", "child.jsonl"), transcriptPath = join(artifacts, "transcript.jsonl");
    writeFileSync(sessionFile, '{"type":"session","id":"native-child"}\n');
    const statusPath = join(state, "async-subagent-runs", "child", "status.json");
    const save = (owner: string, path = transcriptPath) => writeFileSync(statusPath, JSON.stringify({
      lifecycleArtifactVersion: 3, runId: "child", sessionId: owner, steps: [{sessionFile, transcriptPath: path}],
    }));
    const rows = [
      {version: 1, runId: "other", recordType: "message", message: {role: "assistant", content: [{type: "text", text: "WRONG_RUN"}]}},
      {version: 1, runId: "child", recordType: "message", message: {role: "assistant", content: [{type: "text", text: "Correct child"}]}},
    ];
    writeFileSync(transcriptPath, rows.map(row => JSON.stringify(row)).join("\n") + '\n{"incomplete":');
    save("parent");
    const reader = new SubagentActivity(state, artifacts, () => ["parent"]);
    assert.match(JSON.stringify(reader.read("child")), /Correct child/);
    assert.doesNotMatch(JSON.stringify(reader.read("child")), /WRONG_RUN/);
    assert.equal(reader.read("../child"), undefined);
    save("other-parent");
    assert.equal(reader.read("child"), undefined);
    const foreign = join(root, "outside.jsonl");
    writeFileSync(foreign, JSON.stringify(rows[1]));
    const link = join(artifacts, "outside.jsonl");
    symlinkSync(foreign, link);
    save("parent", link);
    assert.equal(reader.read("child"), undefined);
  } finally {rmSync(root, {recursive: true, force: true});}
});
